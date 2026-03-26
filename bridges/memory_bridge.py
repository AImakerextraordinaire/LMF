"""
ANIMA Living Memory Field - Bridge 2: Memory Bridge
=====================================================

Injects LMF field state into transformer mid-layer representations via
LoRA-style low-rank adapters. The model is frozen; only the adapters are trainable.

Architecture:
    For each target layer i in [9..14]:
        field_state (2880) -> down_proj (2880 -> rank) -> GELU -> up_proj (rank -> 2880)
        layer_output += perturbation * (alpha / rank) * significance_gate

    Hooks registered on model.model.layers[i] to inject during forward pass.
    Perturbation is broadcast across sequence length (field state is per-input).

Design Principles:
    - LoRA-style: low-rank keeps parameter count small (~1.1M for 6 layers at rank 32)
    - Significance-gated: injection strength scales with current field significance
    - Device-aware: each injector lives on same device as its target layer
    - Per-layer learned gate: each layer learns how much field influence to accept
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import warnings


class LoRAInjector(nn.Module):
    """
    Single LoRA adapter for one transformer layer.
    
    Transforms field_state into a perturbation for the layer's residual stream.
    
    Parameters:
        field_dim: Dimension of the LMF field state (2880)
        hidden_dim: Dimension of the transformer hidden state (2880) 
        rank: LoRA rank (default 32)
        alpha: LoRA alpha scaling (default 64.0)
        dropout: Dropout on the down-projected representation
    """
    
    def __init__(
        self,
        field_dim: int = 2880,
        hidden_dim: int = 2880,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        
        self.field_dim = field_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha
        self.layer_idx = layer_idx
        self.scaling = alpha / rank
        
        # LoRA projections: field_state -> low rank -> hidden perturbation
        self.down_proj = nn.Linear(field_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_dim, bias=False)
        
        # Per-layer learned gate: how much field influence this layer accepts
        # Initialized near zero so injection starts gentle
        self.gate = nn.Parameter(torch.tensor(0.0))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize: down_proj with small normal, up_proj with zeros
        # This gives zero initial perturbation (standard LoRA init)
        nn.init.kaiming_normal_(self.down_proj.weight, a=0.01)
        nn.init.zeros_(self.up_proj.weight)
    
    def forward(
        self,
        field_state: torch.Tensor,
        significance: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute perturbation for this layer.
        
        Args:
            field_state: [field_dim] current LMF field state
            significance: scalar gate from significance detector [0, 1]
            
        Returns:
            perturbation: [hidden_dim] to add to layer output
        """
        # field_state -> low-rank representation
        low_rank = self.down_proj(field_state)  # [rank]
        low_rank = F.gelu(low_rank)
        low_rank = self.dropout(low_rank)
        
        # low-rank -> hidden-dim perturbation
        perturbation = self.up_proj(low_rank)  # [hidden_dim]
        
        # Scale by LoRA alpha/rank, significance, and learned gate
        gate_value = torch.sigmoid(self.gate)
        perturbation = perturbation * self.scaling * significance * gate_value
        
        return perturbation
    
    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MemoryBridge(nn.Module):
    """
    Bridge 2: Injects LMF field state into transformer mid-layers.
    
    Creates LoRA injectors for target layers and registers forward hooks
    to add field-derived perturbations during the model's forward pass.
    
    The bridge is "armed" before each forward pass with the current field state
    and significance, then "disarmed" after to prevent stale injection.
    
    Parameters:
        field_dim: LMF field dimension (2880)
        hidden_dim: Model hidden dimension (2880)
        target_layers: Which layers to inject into (default [9, 10, 11, 12, 13, 14])
        rank: LoRA rank (default 32)
        alpha: LoRA alpha scaling (default 64.0)
        dropout: Dropout rate (default 0.0)
    """
    
    def __init__(
        self,
        field_dim: int = 2880,
        hidden_dim: int = 2880,
        target_layers: Optional[List[int]] = None,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.field_dim = field_dim
        self.hidden_dim = hidden_dim
        self.target_layers = target_layers or [9, 10, 11, 12, 13, 14]
        self.rank = rank
        self.alpha = alpha
        
        # Create one LoRA injector per target layer
        self.injectors = nn.ModuleDict({
            str(layer_idx): LoRAInjector(
                field_dim=field_dim,
                hidden_dim=hidden_dim,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                layer_idx=layer_idx,
            )
            for layer_idx in self.target_layers
        })
        
        # State for hook injection (set before forward pass, cleared after)
        self._armed = False
        self._field_state = None       # Current field state to inject
        self._significance = 0.0       # Current significance level
        self._hooks = []               # Registered hook handles
        self._injection_stats = {}     # Per-layer injection magnitudes (for debugging)
        self._training_mode = False     # When True, gradients flow through injectors
    
    def arm(
        self,
        field_state: torch.Tensor,
        significance: float = 1.0,
        training: bool = False,
    ):
        """
        Arm the bridge with current field state before model forward pass.
        
        Call this BEFORE running model(input_ids, ...) so the hooks
        have the field state ready for injection.
        """
        self._field_state = field_state.detach()
        self._significance = significance
        self._armed = True
        self._training_mode = training
        self._injection_stats = {}
    
    def disarm(self):
        """
        Disarm after forward pass. Prevents stale injection on next call.
        """
        self._armed = False
        self._field_state = None
        self._significance = 0.0
        self._training_mode = False
    
    def register_hooks(self, model: nn.Module):
        """
        Register forward hooks on target transformer layers.
        
        Detects the layer structure automatically:
            - model.model.layers[i]  (LLaMA, Mistral, GPT-oss)
            - model.transformer.h[i] (GPT-2, GPT-NeoX)
            - model.gpt_neox.layers[i] (GPT-NeoX variant)
        """
        # Remove any existing hooks
        self.remove_hooks()
        
        # Find the layers
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            layer_path = "model.model.layers"
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            layer_path = "model.transformer.h"
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layers = model.gpt_neox.layers
            layer_path = "model.gpt_neox.layers"
        else:
            raise ValueError(
                f"Could not find transformer layers. "
                f"Model type: {type(model).__name__}. "
                f"Top-level attributes: {[a for a in dir(model) if not a.startswith('_')]}"
            )
        
        num_layers = len(layers)
        print(f"  Bridge 2: Found {num_layers} layers at {layer_path}")
        
        # Validate target layers
        for idx in self.target_layers:
            if idx >= num_layers:
                raise ValueError(f"Target layer {idx} >= num_layers {num_layers}")
        
        # Register hooks
        for layer_idx in self.target_layers:
            hook = layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)
        
        print(f"  Bridge 2: Hooks registered on layers {self.target_layers}")
        return self
    
    def _make_hook(self, layer_idx: int):
        """
        Create a forward hook closure for a specific layer.
        
        The hook adds the field-derived perturbation to the layer output.
        Layer output format varies by architecture but is typically:
            (hidden_states, ...) or just hidden_states
        """
        def hook_fn(module, input, output):
            if not self._armed or self._field_state is None:
                return output
            
            # Get the injector for this layer
            injector = self.injectors[str(layer_idx)]
            
            # Determine the hidden states tensor from the output
            # Most architectures return (hidden_states, ...) tuple
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            # Move field state to same device as hidden states
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            field_on_device = self._field_state.to(device=device, dtype=torch.float32)
            
            # Move injector to same device (lazy migration)
            if next(injector.parameters()).device != device:
                injector = injector.to(device)
                self.injectors[str(layer_idx)] = injector
            
            # Compute perturbation
            # When training_mode is True, gradients flow through injector params
            # When False, wrapped in no_grad for inference efficiency
            if self._training_mode:
                perturbation = injector(field_on_device, self._significance)
                perturbation = perturbation.to(dtype=dtype)
                perturbation = perturbation.unsqueeze(0).unsqueeze(0)
                perturbation = perturbation.expand_as(hidden_states)
            else:
                with torch.no_grad():
                    perturbation = injector(field_on_device, self._significance)
                    perturbation = perturbation.to(dtype=dtype)
                    perturbation = perturbation.unsqueeze(0).unsqueeze(0)
                    perturbation = perturbation.expand_as(hidden_states)
            
            # Add to hidden states
            modified = hidden_states + perturbation
            
            # Track injection magnitude
            self._injection_stats[layer_idx] = {
                'perturbation_norm': perturbation[0, 0].float().norm().item(),
                'hidden_norm': hidden_states[0, 0].float().norm().item(),
                'ratio': (perturbation[0, 0].float().norm() / 
                         hidden_states[0, 0].float().norm().clamp(min=1e-8)).item(),
                'gate': torch.sigmoid(injector.gate).item(),
            }
            
            if rest is not None:
                return (modified,) + rest
            else:
                return modified
        
        return hook_fn
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def get_injection_stats(self) -> Dict:
        """Get per-layer injection statistics from the last forward pass."""
        return self._injection_stats.copy()
    
    def get_param_count(self) -> int:
        """Total trainable parameters across all injectors."""
        return sum(p.numel() for p in self.parameters())
    
    def get_gate_values(self) -> Dict[int, float]:
        """Get current learned gate values for each layer."""
        return {
            int(k): torch.sigmoid(v.gate).item() 
            for k, v in self.injectors.items()
        }
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()
