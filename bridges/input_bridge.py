"""
ANIMA Living Memory Field - Bridge 1: Input Bridge
Transformer hidden states → Field perturbation

This bridge extracts the transformer's representation of what it just processed
and projects it into a field perturbation. The field receives signal about
"what happened" without needing to re-process the tokens.

Architecture (from probe):
  - Model: GptOssForCausalLM, hidden_size=2880, 24 layers
  - MoE: 32 experts, 4 active per token (router decisions = free significance signal)
  - Hidden states: [batch, seq_len, 2880] at each layer
  - We tap the LAST hidden state (post all 24 layers, pre-lm_head)

Design choices:
  - Low-rank projection (2880 → 64 → 2880): 0.4M params, not 8.3M
  - Sequence pooling: weighted attention pool, not just mean
    (some tokens matter more than others for memory formation)
  - Significance gating: output scaled by a learned significance estimate
    (feeds into field.process_input's significance evaluation)
  - Input scaling (alpha): prevents the field from being overwhelmed
    by large transformer activations on first integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class AttentionPool(nn.Module):
    """
    Learned attention pooling over sequence dimension.
    
    Instead of mean-pooling (all tokens equal) or last-token (only final position),
    this learns which token positions carry the most memory-relevant information.
    
    For a 69-token input, this produces a single [2880] vector that emphasizes
    the tokens most relevant for memory formation.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Single-head attention: project to scalar score per token
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
            
        Returns:
            pooled: [batch, hidden_dim]
        """
        # Compute attention scores: how relevant is each token for memory?
        keys = self.key_proj(hidden_states)  # [batch, seq, dim]
        scores = torch.einsum('d, bsd -> bs', self.query, keys) / self.scale  # [batch, seq]
        
        # Mask padding tokens
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)  # [batch, seq]
        
        # Weighted sum
        pooled = torch.einsum('bs, bsd -> bd', weights, hidden_states)  # [batch, dim]
        
        return pooled


class InputBridge(nn.Module):
    """
    Bridge 1: Transformer → Living Memory Field
    
    Takes the last hidden state from GPT-oss-20b and produces:
    1. A field perturbation vector (what to inject into the field)
    2. A significance estimate (how strongly to inject it)
    
    The perturbation is L2-normalized to match the field's pattern space,
    then scaled by alpha * significance.
    
    Signal flow:
        hidden_states [B, S, 2880]
          → attention_pool → [B, 2880]
          → low_rank_proj → [B, 2880]  (the perturbation)
          → normalize → [B, 2880]
          
        hidden_states [B, S, 2880]
          → attention_pool → [B, 2880]
          → sig_head → [B, 1]  (significance estimate, 0-1)
          
        output = perturbation * alpha * significance
    """
    
    def __init__(
        self,
        hidden_dim: int = 2880,
        bottleneck_dim: int = 64,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        
        # === Sequence pooling ===
        self.pool = AttentionPool(hidden_dim)
        
        # === Low-rank projection ===
        # 2880 → 64 → 2880 = 368,640 params (vs 8.3M for full projection)
        # The bottleneck forces the bridge to learn a compressed "memory-relevant"
        # subspace of the transformer's representation
        self.proj_down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.proj_up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.proj_norm = nn.LayerNorm(hidden_dim)
        
        # === Significance head ===
        # Learns to estimate "how important is this input for memory?"
        # This gives the field's SignificanceDetector a head start —
        # the bridge has already seen the full transformer computation
        self.sig_head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        
        # Initialize projection to near-identity behavior
        # (initially, the bridge passes through a compressed version of the input)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for stable early training."""
        # Low-rank projection: small random init
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)
        
        # Significance head: initialize to output ~0.5
        # (let the system learn what's significant vs not)
        nn.init.xavier_uniform_(self.sig_head[0].weight)
        nn.init.zeros_(self.sig_head[0].bias)
        nn.init.xavier_uniform_(self.sig_head[2].weight)
        nn.init.zeros_(self.sig_head[2].bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process transformer hidden states into field perturbation.
        
        Args:
            hidden_states: Last hidden state [batch, seq_len, 2880]
            attention_mask: Token mask [batch, seq_len]
            return_components: If True, return intermediate values for debugging
            
        Returns:
            dict with:
                'perturbation': [batch, 2880] — normalized field perturbation
                'significance': [batch, 1] — estimated significance (0-1)
                'scaled_perturbation': [batch, 2880] — perturbation * alpha * sig
                (optional) 'pooled', 'pre_norm_proj' for debugging
        """
        # Pool over sequence
        pooled = self.pool(hidden_states, attention_mask)  # [B, 2880]
        
        # Low-rank projection
        compressed = self.proj_down(pooled)      # [B, 64]
        projected = self.proj_up(compressed)     # [B, 2880]
        projected = self.proj_norm(projected)    # [B, 2880]
        
        # Normalize to unit sphere (matches field pattern space)
        perturbation = F.normalize(projected, dim=-1)  # [B, 2880]
        
        # Significance estimate
        significance = self.sig_head(pooled)  # [B, 1]
        
        # Scale: alpha controls maximum injection strength
        # significance modulates within that range
        scaled = perturbation * self.alpha * significance  # [B, 2880]
        
        result = {
            'perturbation': perturbation,
            'significance': significance,
            'scaled_perturbation': scaled,
        }
        
        if return_components:
            result['pooled'] = pooled
            result['compressed'] = compressed
        
        return result
    
    def get_param_count(self) -> Dict[str, int]:
        """Parameter budget breakdown."""
        counts = {
            'attention_pool': sum(p.numel() for p in self.pool.parameters()),
            'projection': sum(p.numel() for n, p in self.named_parameters() 
                            if 'proj' in n),
            'significance_head': sum(p.numel() for p in self.sig_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts
