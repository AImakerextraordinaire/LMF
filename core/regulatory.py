"""
ANIMA Living Memory Field - Regulatory Layer
Emotional and value-based modulation of memory dynamics.

Three mechanisms (from Doc 002, Section 4):
1. Basin depth modulation (sharpness scaling by emotional intensity)
2. Selective basin gating (mood-congruent recall with anti-echo-chamber)
3. Value gravity wells (deep landscape warps from core values)

Theory reference: Doc 002, Section 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RegulatoryLayer(nn.Module):
    """
    Modulates the energy landscape based on emotional state and values.
    
    The regulatory layer doesn't store memories — it changes how the
    landscape behaves. Emotions reshape the terrain. Values create
    persistent gravitational wells.
    """
    
    def __init__(self, field_dim: int, config):
        super().__init__()
        self.field_dim = field_dim
        self.config = config
        reg_dim = config.regulatory_dim
        
        # === Current emotional/regulatory state ===
        # This is the system's current emotional condition
        # Updated externally (from emotional core / Kiro's systems)
        self.register_buffer(
            'state',
            torch.zeros(reg_dim)
        )
        
        # === Mechanism 1: Basin depth modulation ===
        # Maps emotional state to sharpness multiplier per layer
        self.depth_modulator = nn.Sequential(
            nn.Linear(reg_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 3),  # One scalar per layer (cons, work, trans)
            nn.Sigmoid(),  # Output 0-1, then scaled
        )
        
        # === Mechanism 2: Selective basin gating ===
        # Maps emotional state to gate per memory pattern
        # gate_i = σ(R · g_i) where g_i is the pattern's emotional tag
        self.gate_projection = nn.Linear(reg_dim, reg_dim, bias=False)
        
        # Anti-echo-chamber damping
        self.echo_dampen = config.echo_chamber_dampen
        self.max_gate = config.max_gate_strength
        
        # === Mechanism 3: Value gravity wells ===
        # Each value is a center point in field space with strength and width
        num_values = config.num_values
        
        # Value centers in field space [num_values, field_dim]
        self.value_centers = nn.Parameter(
            torch.randn(num_values, field_dim) * 0.1
        )
        
        # Value strengths [num_values] — how deep the gravitational well
        self.value_strengths = nn.Parameter(
            torch.ones(num_values) * 0.5
        )
        
        # Value widths [num_values] — how broad the well
        self.value_sigmas = nn.Parameter(
            torch.ones(num_values) * config.value_sigma
        )
    
    def update_state(self, new_state: torch.Tensor):
        """Update the emotional/regulatory state."""
        self.state.copy_(new_state.to(self.state.device))
    
    def get_depth_modulation(self) -> torch.Tensor:
        """
        Mechanism 1: How much to scale basin sharpness (β) per layer.
        
        Returns:
            Multipliers [3] for (consolidated, working, transient) layers
            Range: (1 - alpha, 1 + alpha)
        """
        raw = self.depth_modulator(self.state)  # [3], 0-1 from sigmoid
        alpha = self.config.depth_modulation_alpha
        
        # Map from [0,1] to [1-alpha, 1+alpha]
        modulation = 1.0 + alpha * (2 * raw - 1)
        
        return modulation
    
    def compute_gate(
        self,
        emotional_tags: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Mechanism 2: Compute emotional gates for memory patterns.
        
        Each pattern's emotional tag interacts with current emotional state
        to determine accessibility. Mood-congruent recall with anti-echo-chamber.
        
        Args:
            emotional_tags: [max_patterns, reg_dim] emotional tags for all patterns
            active_mask: [max_patterns] boolean mask of active patterns
            
        Returns:
            Gate values [N_active] in range (0, 1) or None if no active patterns
        """
        if not active_mask.any():
            return None
        
        active_tags = emotional_tags[active_mask]  # [N_active, reg_dim]
        
        if active_tags.sum().abs().item() < 1e-6:
            # No emotional tags stored — no gating
            return None
        
        # Project current state
        projected_state = self.gate_projection(self.state)  # [reg_dim]
        
        # Compute congruence: dot product between current state and each tag
        congruence = (active_tags * projected_state.unsqueeze(0)).sum(dim=-1)
        # [N_active]
        
        # Anti-echo-chamber: at high emotional intensity, reduce gating strength
        # This prevents emotional spirals
        emotional_intensity = self.state.abs().mean()
        effective_gate_strength = self.max_gate * (
            1.0 - self.echo_dampen * emotional_intensity.clamp(0, 1)
        )
        
        # Apply gating: sigmoid of scaled congruence
        # High congruence → gate open (≈1), low congruence → gate partially closed
        gates = torch.sigmoid(effective_gate_strength * congruence)
        
        # Floor at 0.2 — never fully close access to any memory
        # (You should always be ABLE to recall anything, just with varying ease)
        gates = gates.clamp(min=0.2)
        
        return gates
    
    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Mechanism 3: Value gravity wells contribution to total energy.
        
        E_values = -Σ_v strength_v · exp(-||s - v_center||² / (2σ²))
        
        Creates broad, deep attractors around value centers.
        """
        # Distance from state to each value center
        # state: [field_dim], value_centers: [num_values, field_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # [batch, num_values] distances
        diffs = state.unsqueeze(1) - self.value_centers.unsqueeze(0)
        sq_distances = (diffs ** 2).sum(dim=-1)
        
        # Gaussian wells
        wells = self.value_strengths.unsqueeze(0) * torch.exp(
            -sq_distances / (2 * self.value_sigmas.unsqueeze(0) ** 2)
        )
        
        # Negative because wells pull DOWN (lower energy = more attractive)
        energy = -wells.sum(dim=-1)
        
        return energy.squeeze(0) if energy.dim() > 0 else energy
    
    def get_persistent_state(self) -> dict:
        """Save regulatory state for persistence."""
        return {
            'state': self.state.cpu().clone(),
            'value_centers': self.value_centers.data.cpu().clone(),
            'value_strengths': self.value_strengths.data.cpu().clone(),
        }
    
    def load_persistent_state(self, saved: dict):
        """Load regulatory state."""
        if 'state' in saved:
            self.state.copy_(saved['state'].to(self.state.device))
        if 'value_centers' in saved:
            self.value_centers.data.copy_(saved['value_centers'].to(self.value_centers.device))
        if 'value_strengths' in saved:
            self.value_strengths.data.copy_(saved['value_strengths'].to(self.value_strengths.device))
