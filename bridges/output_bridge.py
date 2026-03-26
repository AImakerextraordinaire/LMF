"""
ANIMA Living Memory Field - Bridge 3: Output Bridge
Field state → Logit bias (memory-influenced generation)

This bridge takes the current field state and produces a bias over the
vocabulary that nudges token generation. Memories don't replace the model's
predictions — they shift probabilities toward tokens that are more consistent
with the system's experiential history.

Architecture insight (from probe):
  - lm_head: Linear(2880 → 201088) — already maps hidden_dim to vocab
  - embed_tokens: Embedding(201088, 2880) — maps vocab to hidden_dim
  - Both use the same dimensional space as the field state
  
Design: REUSE lm_head weights (0 new projection params for vocab mapping)
  
  The field state lives in the same 2880-dim space as transformer hidden states.
  So we can:
    1. Learn a small transform on the field state (field → "pseudo hidden state")
    2. Pass it through the existing lm_head to get logit bias
    3. Scale by gamma to control influence strength
    
  This is elegant because:
    - The lm_head already knows the mapping from hidden_dim → token probabilities
    - We're asking "if this field state were a hidden state, what tokens would it predict?"
    - That prediction becomes our memory-influenced bias
    - 0 new params for the 2880→201088 mapping (vs 579M for a separate linear)

Signal flow:
    field_state [2880]
      → field_transform (small learned MLP) → [2880]
      → lm_head (frozen, reused) → [201088]  
      → * gamma → logit_bias [201088]
      
    final_logits = model_logits + logit_bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class OutputBridge(nn.Module):
    """
    Bridge 3: Living Memory Field → Token Generation
    
    Transforms the field state into a logit bias that influences generation.
    Uses the model's own lm_head for the heavy lifting (vocab projection).
    
    The field_transform learns to map from "field state space" to 
    "lm_head-compatible hidden state space". These are the same dimensionality
    but may have different distributions — the transform bridges that gap.
    
    Gamma controls influence strength:
      - gamma=0.0: field has no effect on generation (pure base model)
      - gamma=0.1: subtle memory influence (recommended starting point)
      - gamma=0.5: strong memory influence 
      - gamma=1.0: memory bias equal to model logits (too strong, probably)
    """
    
    def __init__(
        self,
        hidden_dim: int = 2880,
        transform_dim: int = 128,
        gamma: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.temperature = temperature
        
        # Reference to model's lm_head (set during integration, not owned)
        self._lm_head = None
        
        # === Field state transform ===
        # Maps field state to lm_head-compatible representation
        # Small MLP with residual connection
        # The residual means untrained bridge ≈ identity (field state passes through)
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, transform_dim),
            nn.GELU(),
            nn.Linear(transform_dim, hidden_dim),
        )
        self.transform_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid → starts at 0.5
        self.transform_norm = nn.LayerNorm(hidden_dim)
        
        # === Optional: learned gamma (can be fixed or trainable) ===
        # Start with fixed gamma, can make trainable later
        self.log_gamma = nn.Parameter(
            torch.tensor(math.log(gamma)), 
            requires_grad=True,  # Trainable - learns optimal influence strength
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for near-zero initial influence."""
        # Transform: small init so initial output is small
        nn.init.xavier_uniform_(self.transform[0].weight, gain=0.1)
        nn.init.zeros_(self.transform[0].bias)
        nn.init.zeros_(self.transform[2].weight)  # Zero init → residual dominates
        nn.init.zeros_(self.transform[2].bias)
    
    def set_lm_head(self, lm_head: nn.Linear):
        """
        Register the model's lm_head for reuse.
        
        Called once during integration setup. The lm_head is NOT owned by this
        module — it stays frozen as part of the base model.
        """
        self._lm_head = lm_head
        # Freeze lm_head weights — gradients flow through but don't update
        for param in self._lm_head.parameters():
            param.requires_grad = False
    
    @property
    def effective_gamma(self) -> float:
        """Current influence strength."""
        return torch.exp(self.log_gamma).item()
    
    def forward(
        self,
        field_state: torch.Tensor,
        model_logits: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute memory-influenced logit bias.
        
        Args:
            field_state: Current LMF state [2880] or [batch, 2880]
            model_logits: Optional base model logits for combined output
                         [batch, seq_len, vocab_size]
            return_components: Return intermediates for debugging
            
        Returns:
            dict with:
                'logit_bias': [batch, vocab_size] or [vocab_size]
                'combined_logits': [batch, seq_len, vocab_size] if model_logits given
                (optional) 'transformed_state', 'raw_lm_output' for debugging
        """
        assert self._lm_head is not None, (
            "lm_head not set. Call output_bridge.set_lm_head(model.lm_head) first."
        )
        
        # Ensure batch dimension
        if field_state.dim() == 1:
            field_state = field_state.unsqueeze(0)  # [1, 2880]
        
        # === Transform field state ===
        # Residual: gate * transform(x) + (1-gate) * x
        gate = torch.sigmoid(self.transform_gate)
        transformed = self.transform(field_state)
        mixed = gate * transformed + (1 - gate) * field_state
        mixed = self.transform_norm(mixed)  # [batch, 2880]
        
        # === Project through lm_head ===
        # Use lm_head WEIGHT directly, not the module.
        # Accelerate's offloading hooks wrap the module and try to move inputs
        # to an "execution device" (which may be meta). Bypassing the module
        # and using F.linear with the raw weight avoids this entirely.
        # Gradients still flow through to the transform (weight is frozen via requires_grad=False).
        lm_weight = self._lm_head.weight  # [vocab_size, hidden_dim]
        lm_bias = getattr(self._lm_head, 'bias', None)
        lm_device = lm_weight.device
        mixed_on_device = mixed.to(device=lm_device, dtype=lm_weight.dtype)
        
        raw_logit_bias = F.linear(mixed_on_device, lm_weight, lm_bias)  # [batch, vocab_size]
        
        # === Scale by gamma ===
        gamma = torch.exp(self.log_gamma)
        logit_bias = gamma * raw_logit_bias  # [batch, vocab_size]
        
        result = {
            'logit_bias': logit_bias,
        }
        
        # === Combine with model logits if provided ===
        if model_logits is not None:
            # Do combination on logit_bias's device (bridge_device / CPU).
            # Accelerate may leave model_logits on meta or mixed devices,
            # so we pull model_logits to us rather than pushing to them.
            model_logits_local = model_logits.to(
                device=logit_bias.device, dtype=logit_bias.dtype
            )
            combined = model_logits_local + logit_bias.unsqueeze(1)  # [B, S, V]
            result['combined_logits'] = combined
        
        if return_components:
            result['transformed_state'] = mixed
            result['raw_lm_output'] = raw_logit_bias
            result['effective_gamma'] = gamma.item()
            result['gate_value'] = gate.item()
        
        return result
    
    def get_param_count(self) -> Dict[str, int]:
        """Parameter budget breakdown."""
        counts = {
            'transform': sum(p.numel() for n, p in self.named_parameters()
                           if 'transform' in n),
            'gamma': 1,
            'lm_head_reused': sum(p.numel() for p in self._lm_head.parameters()) 
                             if self._lm_head else 0,
        }
        counts['total_new'] = counts['transform'] + counts['gamma']
        counts['total_reused'] = counts['lm_head_reused']
        return counts
