"""
ANIMA Living Memory Field - Core Module
The unified energy landscape with multi-timescale memory layers.

This is the heart of the system. The field state evolves continuously,
influenced by multiple layers of memory (structural, consolidated, working,
transient), regulatory modulation, and external input.

Theory reference: Doc 001 (memory substrate), Doc 002 (energy function)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import time
import json
import os

from .memory_layer import MemoryLayer, MemoryPattern
from .significance import SignificanceDetector
from .regulatory import RegulatoryLayer
from .association import AssociationMatrix


class LivingMemoryField(nn.Module):
    """
    The Living Memory Field (LMF).
    
    A persistent, continuous state that evolves over time, shaped by an energy
    landscape defined by multiple layers of memory patterns. This state
    never resets — it is the system's experiential continuity.
    
    The energy function (from Doc 002):
        E_total = E_consolidated + E_working + E_transient 
                + E_regulatory + E_stability
    
    (Structural layer is implicit in the transformer backbone, not here.)
    
    The field state evolves via hybrid SSM + Hopfield dynamics:
        s(t+1) = A·s(t) + B·input(t) + Σ_l memory_pull_l + σ·η
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.field_dim = config.field.field_dim
        
        # === The field state itself ===
        # This is THE persistent state — the ball on the landscape
        # Registered as buffer so it's saved with the model but not trained via backprop
        self.register_buffer(
            'field_state',
            torch.zeros(self.field_dim)
        )
        
        # === Memory layers ===
        # Each layer contributes energy (valleys) to the landscape
        
        # Consolidated: long-term personal memory (deep soil)
        self.consolidated = MemoryLayer(
            pattern_dim=config.consolidated.pattern_dim,
            max_patterns=config.consolidated.max_patterns,
            beta=config.consolidated.beta,
            decay_rate=config.consolidated.decay_rate,
            projection_trainable=config.consolidated.projection_trainable,
            min_depth=config.consolidated.min_depth,
            seed_threshold=config.consolidated.seed_threshold,
        )
        
        # Working: session memory (topsoil)
        self.working = MemoryLayer(
            pattern_dim=config.working.pattern_dim,
            max_patterns=config.working.max_patterns,
            beta=config.working.beta,
            decay_rate=config.working.decay_rate,
            projection_trainable=config.working.projection_trainable,
            min_depth=config.working.min_depth,
            seed_threshold=config.working.seed_threshold,
        )
        
        # Transient: immediate input buffer (snow)
        self.transient = MemoryLayer(
            pattern_dim=config.transient.pattern_dim,
            max_patterns=config.transient.max_patterns,
            beta=config.transient.beta,
            decay_rate=config.transient.decay_rate,
            projection_trainable=config.transient.projection_trainable,
            min_depth=config.transient.min_depth,
            seed_threshold=config.transient.seed_threshold,
        )
        
        # === SSM Evolution Parameters ===
        # A: state transition (how the ball rolls on its own)
        # Parameterized as diag(alpha) + U @ V.T (diagonal + low-rank)
        # - Diagonal: each dimension mostly persists (momentum)
        # - Low-rank: learned cross-dimensional drift directions (trains of thought)
        # This is stable by construction when alpha ∈ (0,1) and rank is small,
        # and costs O(dim * rank) instead of O(dim²) parameters.
        self._A_rank = 16  # Low-rank coupling dimensions
        self.A_diag = nn.Parameter(torch.ones(self.field_dim) * 0.99)
        self.A_low_rank_U = nn.Parameter(torch.randn(self.field_dim, self._A_rank) * 0.001)
        self.A_low_rank_V = nn.Parameter(torch.randn(self.field_dim, self._A_rank) * 0.001)
        
        # Layer-specific retrieval weights (how much each layer pulls)
        self.layer_weights = nn.Parameter(torch.tensor([0.3, 0.5, 0.2]))  # [cons, work, trans]
        
        # === Significance Detector ===
        self.significance = SignificanceDetector(
            field_dim=self.field_dim,
            config=config.significance,
        )
        
        # === Regulatory Layer ===
        self.regulatory = RegulatoryLayer(
            field_dim=self.field_dim,
            config=config.regulatory,
        )
        
        # === Association Matrix ===
        self.associations = AssociationMatrix(
            config=config.association,
        )
        
        # === Stability ===
        self.stability_lambda = config.field.stability_lambda
        self.noise_sigma = config.field.noise_sigma
        self.internal_steps = config.field.internal_steps
        
        # === Consolidation tracking ===
        self._total_steps = 0
        self._last_consolidation = 0
        self.consolidation_interval = config.consolidation.consolidation_interval
        
        # === Anamnesis seed buffer ===
        # Patterns that need to be exported to Anamnesis
        self._seed_buffer: List[dict] = []
        
        # === Metrics tracking ===
        self._metrics = {
            'total_memories_formed': 0,
            'total_memories_decayed': 0,
            'total_consolidations': 0,
            'total_seeds_exported': 0,
        }
    
    @property
    def state(self) -> torch.Tensor:
        """Current field state."""
        return self.field_state
    
    @state.setter
    def state(self, value: torch.Tensor):
        self.field_state.copy_(value)
    
    def compute_total_energy(
        self, 
        state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total energy and per-layer breakdown.
        
        Lower energy = state is near strong memory basins.
        Used for monitoring and visualization, not directly for dynamics.
        """
        if state is None:
            state = self.field_state
        
        energies = {}
        energies['consolidated'] = self.consolidated.compute_energy(state)
        energies['working'] = self.working.compute_energy(state)
        energies['transient'] = self.transient.compute_energy(state)
        energies['stability'] = (self.stability_lambda / 2) * (state ** 2).sum()
        energies['regulatory'] = self.regulatory.compute_energy(state)
        energies['total'] = sum(energies.values())
        
        return energies
    
    def evolve(
        self,
        external_input: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Evolve the field state forward by one or more steps.
        
        This is the core dynamics:
            s(t+1) = A·s(t) + input + Σ_l weight_l · retrieve_l(s) + noise
        
        Args:
            external_input: Optional perturbation from input bridge [field_dim]
            num_steps: Override number of internal steps
            
        Returns:
            Updated field state
        """
        steps = num_steps or self.internal_steps
        s = self.field_state.clone()
        
        # Get regulatory state for modulation
        reg_state = self.regulatory.state
        
        # Get layer weights (softmax to ensure they sum to ~1)
        weights = F.softmax(self.layer_weights, dim=0)
        
        for step in range(steps):
            # === SSM evolution: A·s(t) ===
            # A = diag(alpha) + U @ V.T  (diagonal + low-rank)
            # Clamp diagonal to (0, 1) for guaranteed stability
            alpha_clamped = self.A_diag.clamp(0.01, 0.999)
            s_evolved = alpha_clamped * s + (s @ self.A_low_rank_V) @ self.A_low_rank_U.T
            
            # === External input perturbation ===
            if external_input is not None and step == 0:
                # Only inject input on first step (subsequent steps are settling)
                s_evolved = s_evolved + external_input
            
            # === Memory retrieval (Hopfield step) per layer ===
            # Each layer pulls the state toward its basins
            
            # Get emotional gates from regulatory layer
            cons_gate = self.regulatory.compute_gate(
                self.consolidated.emotional_tags,
                self.consolidated.active_mask,
            )
            work_gate = self.regulatory.compute_gate(
                self.working.emotional_tags,
                self.working.active_mask,
            )
            
            cons_pull = self.consolidated.retrieve(s_evolved, gate=cons_gate)
            work_pull = self.working.retrieve(s_evolved, gate=work_gate)
            trans_pull = self.transient.retrieve(s_evolved)
            
            # Combine with learned weights
            total_pull = (
                weights[0] * cons_pull + 
                weights[1] * work_pull + 
                weights[2] * trans_pull
            )
            
            # === Stability term ===
            # Gentle pull toward origin (prevents runaway)
            stability_pull = -self.stability_lambda * s_evolved
            
            # === Noise (exploration) ===
            noise = self.noise_sigma * torch.randn_like(s_evolved)
            
            # === Combine ===
            s = s_evolved + total_pull + stability_pull + noise
        
        # Update persistent state
        self.field_state.copy_(s.detach())
        self._total_steps += 1
        
        return self.field_state
    
    def process_input(
        self,
        input_embedding: torch.Tensor,
        emotional_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a new input through the field.
        
        This is the main entry point during inference:
        1. Evaluate significance
        2. Perturb field state
        3. Evolve field dynamics
        4. Maybe form new memory
        5. Decay existing memories
        
        Args:
            input_embedding: Embedded input [field_dim]
            emotional_context: Current emotional state [regulatory_dim]
            
        Returns:
            Updated field state
        """
        # Update regulatory state if provided
        if emotional_context is not None:
            self.regulatory.update_state(emotional_context)
        
        # === Significance evaluation ===
        sig_score, sig_components = self.significance.evaluate(
            input_embedding=input_embedding,
            field_state=self.field_state,
            regulatory_state=self.regulatory.state,
            memory_layers=[self.consolidated, self.working, self.transient],
        )
        
        # === Perturb field and evolve ===
        # Scale perturbation by significance
        perturbation = input_embedding * sig_score.clamp(min=0.05)
        
        self.evolve(external_input=perturbation)
        
        # === Memory formation ===
        if sig_score.item() > self.config.significance.formation_threshold:
            self._form_memory(
                input_embedding=input_embedding,
                significance=sig_score.item(),
                emotional_tag=self.regulatory.state.clone(),
            )
        
        # === Store in transient layer (always, for immediate buffer) ===
        self.transient.store_pattern(
            pattern=input_embedding,
            depth=0.5,
            significance=sig_score.item(),
        )
        
        # === Periodic maintenance ===
        if self._total_steps % 10 == 0:
            self._decay_all_layers()
        
        if self._total_steps % self.consolidation_interval == 0:
            self._consolidate()
        
        return self.field_state
    
    def _form_memory(
        self,
        input_embedding: torch.Tensor,
        significance: float,
        emotional_tag: Optional[torch.Tensor] = None,
    ):
        """Form a new memory in the working layer."""
        # Encode: input + current context (field state)
        # The memory is the input IN CONTEXT of the current state
        encoded = F.normalize(
            input_embedding + 0.3 * self.field_state, dim=-1
        )
        
        # Depth scaled by significance and emotional intensity
        depth = self.config.significance.base_depth * significance
        if emotional_tag is not None:
            emotional_intensity = emotional_tag.abs().mean().item()
            depth *= (1.0 + 0.5 * emotional_intensity)
        
        idx = self.working.store_pattern(
            pattern=encoded,
            depth=depth,
            significance=significance,
            emotional_tag=emotional_tag,
        )
        
        if idx is not None:
            self._metrics['total_memories_formed'] += 1
            
            # Track association with recently active patterns
            self.associations.record_activation(
                layer='working',
                pattern_idx=idx,
                field_state=self.field_state,
            )
    
    def _decay_all_layers(self):
        """Apply decay to all memory layers. Collect seeds for Anamnesis."""
        for layer in [self.consolidated, self.working, self.transient]:
            seeds = layer.decay_step()
            if seeds:
                self._seed_buffer.extend(seeds)
                self._metrics['total_seeds_exported'] += len(seeds)
                self._metrics['total_memories_decayed'] += len(seeds)
    
    def _consolidate(self):
        """
        Consolidation: transfer significant working memories to consolidated layer.
        
        This runs periodically (like sleep). It:
        1. Identifies significant working patterns
        2. Abstracts them (slight smoothing)
        3. Transfers to consolidated layer
        4. Clears from working layer
        """
        threshold = self.config.consolidation.consolidation_threshold
        max_per_cycle = self.config.consolidation.max_consolidations_per_cycle
        abstraction = self.config.consolidation.abstraction_factor
        
        candidates = []
        
        for idx in range(self.working.max_patterns):
            meta = self.working.pattern_metadata[idx]
            if meta is None:
                continue
            
            # Score: significance × depth × access_count
            depth = self.working.depths[idx].item()
            score = meta.significance * depth * (1 + meta.access_count * 0.1)
            
            if score > threshold:
                candidates.append((score, idx, meta))
        
        # Sort by score descending, take top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, idx, meta in candidates[:max_per_cycle]:
            # Get the pattern
            pattern = self.working.patterns[idx].clone()
            emotional_tag = self.working.emotional_tags[idx].clone()
            
            # Abstraction: slight smoothing (add small noise, renormalize)
            # This simulates the gist-extraction of consolidation
            if abstraction < 1.0:
                noise_scale = (1.0 - abstraction) * 0.1
                pattern = pattern + noise_scale * torch.randn_like(pattern)
                pattern = F.normalize(pattern, dim=-1)
            
            # Store in consolidated layer
            cons_idx = self.consolidated.store_pattern(
                pattern=pattern,
                depth=self.working.depths[idx].item() * 1.2,  # Boost on consolidation
                significance=meta.significance,
                emotional_tag=emotional_tag,
                value_alignment=meta.value_alignment,
            )
            
            if cons_idx is not None:
                # Mark as consolidated and clear from working
                meta.consolidated = True
                self.working._clear_slot(idx)
                self._metrics['total_consolidations'] += 1
    
    def reconstruct_from_seed(
        self,
        seed_pattern: torch.Tensor,
        target_layer: str = 'consolidated',
        seed_depth: float = 0.5,
    ) -> Optional[int]:
        """
        Reconstruct a memory from an Anamnesis seed.
        
        Injects the seed as a new pattern. If residual landscape deformation
        exists (the basin hasn't fully vanished), the field dynamics will
        enrich the reconstruction with associated context.
        
        Args:
            seed_pattern: The seed vector from Anamnesis
            target_layer: Which layer to inject into
            seed_depth: Initial depth (lower than natural — it's a reconstruction)
            
        Returns:
            Pattern index if successful
        """
        layer = self.consolidated if target_layer == 'consolidated' else self.working
        
        idx = layer.store_pattern(
            pattern=seed_pattern,
            depth=seed_depth,
            significance=0.5,  # Moderate — it's reconstructed, not fresh
        )
        
        return idx
    
    # === Persistence ===
    
    def save_persistent_state(self, path: str):
        """Save the entire field state to disk. Call between sessions."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        state = {
            'field_state': self.field_state.cpu(),
            'consolidated': self.consolidated.get_state_dict_persistent(),
            'working': self.working.get_state_dict_persistent(),
            'transient': self.transient.get_state_dict_persistent(),
            'regulatory': self.regulatory.get_persistent_state(),
            'associations': self.associations.get_persistent_state(),
            'total_steps': self._total_steps,
            'metrics': self._metrics,
            'seed_buffer': self._seed_buffer,
        }
        
        torch.save(state, path)
    
    def load_persistent_state(self, path: str):
        """Load field state from disk. Call at session start."""
        if not os.path.exists(path):
            return  # Fresh start
        
        state = torch.load(path, map_location='cpu', weights_only=False)
        
        self.field_state.copy_(state['field_state'].to(self.field_state.device))
        self.consolidated.load_state_dict_persistent(state['consolidated'])
        self.working.load_state_dict_persistent(state['working'])
        self.transient.load_state_dict_persistent(state['transient'])
        self.regulatory.load_persistent_state(state.get('regulatory', {}))
        self.associations.load_persistent_state(state.get('associations', {}))
        self._total_steps = state.get('total_steps', 0)
        self._metrics = state.get('metrics', self._metrics)
        self._seed_buffer = state.get('seed_buffer', [])
    
    # === Monitoring ===
    
    def get_status(self) -> dict:
        """Get current field status for monitoring/visualization."""
        energies = self.compute_total_energy()
        
        return {
            'field_norm': self.field_state.norm().item(),
            'consolidated_active': self.consolidated.num_active,
            'consolidated_capacity': self.consolidated.max_patterns,
            'working_active': self.working.num_active,
            'working_capacity': self.working.max_patterns,
            'transient_active': self.transient.num_active,
            'total_energy': energies['total'].item(),
            'energy_breakdown': {k: v.item() for k, v in energies.items()},
            'total_steps': self._total_steps,
            'seeds_pending': len(self._seed_buffer),
            'regulatory_state': self.regulatory.state.tolist() if self.regulatory.state is not None else None,
            **self._metrics,
        }
    
    def __repr__(self):
        return (
            f"LivingMemoryField(\n"
            f"  field_dim={self.field_dim},\n"
            f"  consolidated={self.consolidated},\n"
            f"  working={self.working},\n"
            f"  transient={self.transient},\n"
            f"  total_steps={self._total_steps}\n"
            f")"
        )
