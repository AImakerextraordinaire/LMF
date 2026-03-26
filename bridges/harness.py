"""
ANIMA Living Memory Field - Bridge Integration Harness
Connects Bridge 1 (Input), Bridge 2 (Memory), and Bridge 3 (Output) to a transformer model + LMF.

This is the "wiring diagram" — it hooks into the model's forward pass to:
0. Arm Bridge 2 with current field state for mid-layer injection
2. Capture hidden states after the model processes input (Bridge 2 injects during forward) (→ Bridge 1 → Field)
3. Inject field state as logit bias during generation (Field → Bridge 3 → logits)

Usage:
    model = AutoModelForCausalLM.from_pretrained(...)
    lmf = LivingMemoryField(config)
    
    harness = BridgeHarness(model, lmf)
    harness.process_and_generate(input_ids, attention_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import time

from .input_bridge import InputBridge
from .output_bridge import OutputBridge
from .memory_bridge import MemoryBridge


class BridgeHarness(nn.Module):
    """
    Wires Bridge 1 + Bridge 2 + Bridge 3 between a transformer model and the LMF.
    
    Lifecycle per generation step:
        1. Model processes input tokens → hidden states
        2. Bridge 1: last hidden state → attention pool → project → field perturbation
        3. LMF: process perturbation → evolve field state → maybe form memory
        4. Model generates next token logits
        5. Bridge 3: field state → transform → lm_head → logit bias
        6. Combined logits = model logits + logit bias
        7. Sample/argmax from combined logits
        
    The model is FROZEN. Only the bridges and LMF are trainable.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lmf: nn.Module,
        input_bridge: Optional[InputBridge] = None,
        output_bridge: Optional[OutputBridge] = None,
        bridge_device: str = "cpu",
    ):
        super().__init__()
        
        self.model = model
        self.lmf = lmf
        self.bridge_device = torch.device(bridge_device)
        
        # Get model config
        hidden_dim = model.config.hidden_size
        
        # === Bridge 1: Input ===
        self.input_bridge = input_bridge or InputBridge(
            hidden_dim=hidden_dim,
            bottleneck_dim=64,
            alpha=0.1,
        )
        self.input_bridge = self.input_bridge.to(self.bridge_device)
        
        # === Bridge 3: Output ===
        self.output_bridge = output_bridge or OutputBridge(
            hidden_dim=hidden_dim,
            transform_dim=128,
            gamma=0.1,
        )
        self.output_bridge = self.output_bridge.to(self.bridge_device)
        
        # === Bridge 2: Memory (LoRA mid-layer injection) ===
        self.memory_bridge = MemoryBridge(
            field_dim=hidden_dim,
            hidden_dim=hidden_dim,
            target_layers=[9, 10, 11, 12, 13, 14],
            rank=32,
            alpha=64.0,
        )
        # Register hooks on model layers (injectors migrate to layer devices lazily)
        self.memory_bridge.register_hooks(model)
        
        # === Connect output bridge to model's lm_head ===
        # Accelerate may offload lm_head to meta device. We can't call it directly.
        # Solution: capture the REAL weight during a forward pass via hook,
        # or extract it if it's already materialized.
        self._lm_head_weight = None
        self._extract_lm_head_weight()
        
        # === Freeze base model ===
        for param in model.parameters():
            param.requires_grad = False
        
        # === Metrics ===
        self._step_count = 0
        self._timing = {}
    
    def _extract_lm_head_weight(self):
        """
        Extract lm_head weight, handling accelerate's meta tensor offloading.
        
        When accelerate offloads layers to CPU/disk, the weight tensor is replaced
        with a meta tensor (no data). During forward passes, accelerate materializes
        the real weight temporarily. We use a forward hook to capture it.
        """
        lm_head = self.model.lm_head
        weight = lm_head.weight
        
        if weight.device.type != 'meta':
            # Weight is already materialized — just clone it
            self._lm_head_weight = weight.data.clone().to(
                device=self.bridge_device, dtype=torch.float32
            )
            self._lm_head_weight.requires_grad = False
            print(f"  lm_head weight extracted directly: {self._lm_head_weight.shape} on {self.bridge_device}")
        else:
            # Weight is on meta device — accelerate offloaded it.
            # Use accelerate's own hook to materialize the weight directly.
            print("  lm_head is on meta device (accelerate offloaded). Extracting...")
            
            extracted = False
            
            # Method 1: Use accelerate's hook to materialize weight in-place
            if hasattr(lm_head, '_hf_hook'):
                hook = lm_head._hf_hook
                try:
                    # pre_forward materializes offloaded weights onto the execution device
                    hook.pre_forward(lm_head)
                    if lm_head.weight.device.type != 'meta':
                        self._lm_head_weight = lm_head.weight.data.clone().cpu().float()
                        extracted = True
                        print(f"  Extracted via accelerate pre_forward hook")
                    # Let accelerate clean up (may offload again)
                    try:
                        hook.post_forward(lm_head, None)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"  pre_forward hook failed: {e}")
            
            # Method 2: Check accelerate's weights_map
            if not extracted and hasattr(lm_head, '_hf_hook'):
                hook = lm_head._hf_hook
                if hasattr(hook, 'weights_map') and hook.weights_map is not None:
                    try:
                        import numpy as np
                        weight_data = hook.weights_map.get('weight', None)
                        if weight_data is not None:
                            if isinstance(weight_data, torch.Tensor):
                                self._lm_head_weight = weight_data.clone().cpu().float()
                            else:
                                self._lm_head_weight = torch.tensor(weight_data, dtype=torch.float32)
                            extracted = True
                            print(f"  Extracted from accelerate weights_map")
                    except Exception as e:
                        print(f"  weights_map extraction failed: {e}")
            
            # Method 3: Check offload folder
            if not extracted:
                import os
                offload_dir = "offload_temp"
                possible_paths = [
                    os.path.join(offload_dir, "lm_head.weight.dat"),
                    os.path.join(offload_dir, "lm_head", "weight.dat"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        try:
                            import numpy as np
                            data = np.fromfile(path, dtype=np.float16 if 'bf16' not in path else np.float16)
                            # lm_head shape: [vocab_size, hidden_dim] = [201088, 2880]
                            data = data.reshape(self.model.config.vocab_size, self.model.config.hidden_size)
                            self._lm_head_weight = torch.from_numpy(data).float()
                            extracted = True
                            print(f"  Extracted from offload file: {path}")
                            break
                        except Exception as e:
                            print(f"  Offload file load failed ({path}): {e}")
            
            if not extracted:
                raise RuntimeError(
                    "Failed to extract lm_head weight from meta device. "
                    "Tried: pre_forward hook, weights_map, offload folder."
                )
            
            self._lm_head_weight.requires_grad = False
            print(f"  lm_head weight shape: {self._lm_head_weight.shape} on {self.bridge_device}")
        
        # Create a plain Linear on CPU with the captured weight (for output bridge)
        lm_head_cpu = nn.Linear(
            self._lm_head_weight.shape[1], 
            self._lm_head_weight.shape[0], 
            bias=False,
        )
        lm_head_cpu.weight = nn.Parameter(self._lm_head_weight, requires_grad=False)
        self.output_bridge.set_lm_head(lm_head_cpu)
    
    def capture_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        enable_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the model forward and capture hidden states + logits.
        
        Args:
            enable_grad: If True, don't wrap in no_grad (needed for Bridge 2 training)
        
        Returns:
            last_hidden: [batch, seq_len, hidden_dim]
            logits: [batch, seq_len, vocab_size]
        """
        if enable_grad:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        
        last_hidden = outputs.hidden_states[-1]  # [B, S, 2880]
        logits = outputs.logits                       # [B, S, vocab_size]
        
        # Ensure tensors are on real devices (accelerate may leave them on meta)
        if last_hidden.device.type == 'meta':
            raise RuntimeError(f"Hidden states are on meta device — accelerate offloading issue")
        if logits.device.type == 'meta':
            raise RuntimeError(f"Logits are on meta device — accelerate offloading issue")
        
        return last_hidden, logits
    
    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        emotional_context: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        enable_grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full Bridge 1 → LMF → Bridge 3 cycle.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: [batch, seq_len]
            emotional_context: Optional emotional state for regulatory layer
            return_debug: Include intermediate values
            
        Returns:
            dict with:
                'logits': Combined logits [batch, seq_len, vocab_size]
                'model_logits': Original model logits
                'logit_bias': Memory-influenced bias
                'field_state': Current field state after update
                'significance': Bridge 1's significance estimate
        """
        t0 = time.time()
        
        # === Step 0: Arm Bridge 2 with current field state ===
        # Bridge 2 uses the PREVIOUS step's field state to influence
        # the current forward pass. This is correct: accumulated memory
        # should influence processing of new input.
        field_norm = self.lmf.field_state.norm().item()
        if field_norm > 1e-6:
            self.memory_bridge.arm(
                field_state=self.lmf.field_state,
                significance=min(field_norm / 5.0, 1.0),  # Normalize by typical field magnitude
                training=enable_grad,
            )
        
        # === Step 1: Model forward (Bridge 2 hooks inject during this) ===
        last_hidden, model_logits = self.capture_hidden_states(
            input_ids, attention_mask, enable_grad=enable_grad,
        )
        
        # Disarm Bridge 2 to prevent stale injection
        bridge2_stats = self.memory_bridge.get_injection_stats()
        self.memory_bridge.disarm()
        t_model = time.time()
        
        # === Step 2: Bridge 1 — hidden states → field perturbation ===
        # Move hidden states to bridge device
        hidden_on_bridge = last_hidden.to(self.bridge_device, dtype=torch.float32)
        mask_on_bridge = attention_mask.to(self.bridge_device) if attention_mask is not None else None
        
        bridge1_out = self.input_bridge(
            hidden_on_bridge, 
            mask_on_bridge,
            return_components=return_debug,
        )
        t_bridge1 = time.time()
        
        # === Step 3: LMF processes the perturbation ===
        # Take first batch element (LMF is single-state for now)
        perturbation = bridge1_out['scaled_perturbation'][0]  # [2880]
        perturbation_on_field = perturbation.to(self.lmf.field_state.device)
        
        self.lmf.process_input(
            input_embedding=perturbation_on_field,
            emotional_context=emotional_context,
        )
        t_lmf = time.time()
        
        # === Step 4: Bridge 3 — field state → logit bias ===
        field_on_bridge = self.lmf.field_state.to(self.bridge_device, dtype=torch.float32)
        
        bridge3_out = self.output_bridge(
            field_state=field_on_bridge,
            model_logits=model_logits,
            return_components=return_debug,
        )
        t_bridge3 = time.time()
        
        # === Timing ===
        self._timing = {
            'model_forward_ms': (t_model - t0) * 1000,
            'bridge1_ms': (t_bridge1 - t_model) * 1000,
            'lmf_process_ms': (t_lmf - t_bridge1) * 1000,
            'bridge3_ms': (t_bridge3 - t_lmf) * 1000,
            'total_ms': (t_bridge3 - t0) * 1000,
        }
        self._step_count += 1
        
        result = {
            'logits': bridge3_out.get('combined_logits', model_logits),
            'model_logits': model_logits,
            'logit_bias': bridge3_out['logit_bias'],
            'field_state': self.lmf.field_state.clone(),
            'significance': bridge1_out['significance'],
            'timing': self._timing,
        }
        
        if return_debug:
            result['hidden_states'] = last_hidden
            result['bridge1'] = bridge1_out
            result['bridge2'] = bridge2_stats
            result['bridge3'] = bridge3_out
            result['bridge2_gates'] = self.memory_bridge.get_gate_values()
            result['field_status'] = self.lmf.get_status()
        
        return result
    
    def get_status(self) -> dict:
        """Get harness status for monitoring."""
        return {
            'step_count': self._step_count,
            'timing': self._timing,
            'field_status': self.lmf.get_status(),
            'input_bridge_params': self.input_bridge.get_param_count(),
            'output_bridge_params': self.output_bridge.get_param_count(),
            'memory_bridge_params': self.memory_bridge.get_param_count(),
            'memory_bridge_gates': self.memory_bridge.get_gate_values(),
            'bridge_device': str(self.bridge_device),
            'effective_gamma': self.output_bridge.effective_gamma,
        }
    
    def __repr__(self):
        ib = self.input_bridge.get_param_count()
        ob = self.output_bridge.get_param_count()
        return (
            f"BridgeHarness(\n"
            f"  model={self.model.config.model_type},\n"
            f"  input_bridge={ib['total']:,} params,\n"
            f"  output_bridge={ob['total_new']:,} new params "
            f"({ob['total_reused']:,} reused from lm_head),\n"
            f"  bridge_device={self.bridge_device},\n"
            f"  gamma={self.output_bridge.effective_gamma:.3f},\n"
            f"  steps={self._step_count}\n"
            f")"
        )
