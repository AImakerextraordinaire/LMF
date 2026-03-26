"""
ANIMA Living Memory Field - Memory Layer
Implements a single layer of the energy landscape using Modern Hopfield Networks.

Each layer stores explicit memory patterns as vectors and computes
energy/retrieval via the Hopfield attention mechanism.

Theory reference: Doc 002, Sections 3 and 6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time


@dataclass
class MemoryPattern:
    """A single memory pattern stored in a layer."""
    
    # The pattern vector itself (lives in the layer's pattern store tensor)
    index: int  # Index into the pattern store
    
    # Basin depth (how strong/stable this memory is)
    depth: float = 1.0
    
    # Emotional tag: vector describing emotional context at formation
    emotional_tag: Optional[torch.Tensor] = None
    
    # Temporal metadata
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    
    # Significance score at formation
    significance: float = 0.0
    
    # Value alignment score
    value_alignment: float = 0.0
    
    # Whether this pattern has been consolidated (working → consolidated)
    consolidated: bool = False


class MemoryLayer(nn.Module):
    """
    A single memory layer in the Living Memory Field.
    
    Implements the Hopfield energy function:
        E_layer(s) = -(1/β) · log( Σᵢ depth_i · exp(β · ξᵢᵀ W s) )
    
    And the corresponding retrieval (gradient descent = attention):
        retrieval = Σᵢ softmax(β · ξᵢᵀ W s · depth_i) · ξᵢ
    
    Each layer has its own:
    - Pattern store (the ξᵢ vectors)
    - Projection matrix W (learned lens for interpreting state)
    - Sharpness β (how precise/fuzzy the memories are)
    - Decay rate (how fast patterns fade)
    """
    
    def __init__(
        self,
        pattern_dim: int,
        max_patterns: int,
        beta: float = 1.0,
        decay_rate: float = 0.0001,
        projection_trainable: bool = True,
        min_depth: float = 0.01,
        seed_threshold: float = 0.05,
    ):
        super().__init__()
        
        self.pattern_dim = pattern_dim
        self.max_patterns = max_patterns
        self.beta = beta
        self.decay_rate = decay_rate
        self.min_depth = min_depth
        self.seed_threshold = seed_threshold
        
        # Pattern store: [max_patterns, pattern_dim]
        # Initialized empty (zeros), filled as memories form
        self.register_buffer(
            'patterns', 
            torch.zeros(max_patterns, pattern_dim)
        )
        
        # Depth store: basin depth for each pattern
        # 0 = empty slot, > 0 = active pattern
        self.register_buffer(
            'depths',
            torch.zeros(max_patterns)
        )
        
        # Emotional tags: [max_patterns, emotional_dim]
        # Stored separately for gating (Doc 002 Section 4.3)
        self.register_buffer(
            'emotional_tags',
            torch.zeros(max_patterns, 17)  # 17 = Kiro's emotional axes
        )
        
        # Projection matrix W: learned lens for this layer
        # Maps field state to pattern-comparison space
        # CRITICAL: Initialize as identity so retrieval works from the start.
        # Random init destroys pattern-query correlation.
        self.projection = nn.Linear(pattern_dim, pattern_dim, bias=False)
        nn.init.eye_(self.projection.weight)
        if not projection_trainable:
            for param in self.projection.parameters():
                param.requires_grad = False
        
        # Metadata tracking (CPU, not on GPU)
        self.pattern_metadata: List[Optional[MemoryPattern]] = [
            None for _ in range(max_patterns)
        ]
        
        # Count of active patterns
        self._num_active = 0
        
        # Step counter for decay timing
        self._step_count = 0
    
    @property
    def num_active(self) -> int:
        """Number of currently active (non-empty) patterns."""
        return (self.depths > 0).sum().item()
    
    @property
    def active_mask(self) -> torch.Tensor:
        """Boolean mask of active pattern slots."""
        return self.depths > 0
    
    def get_active_patterns(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return only active patterns and their depths."""
        mask = self.active_mask
        return self.patterns[mask], self.depths[mask]
    
    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the energy contribution of this layer for a given state.
        
        E_layer(s) = -(1/β) · log( Σᵢ depth_i · exp(β · ξᵢᵀ W s) )
        
        Lower energy = state is near a strong memory basin.
        
        Args:
            state: Field state vector [batch, field_dim] or [field_dim]
            
        Returns:
            Energy scalar (or [batch] if batched)
        """
        if self.num_active == 0:
            return torch.tensor(0.0, device=state.device, dtype=state.dtype)
        
        was_1d = state.dim() == 1
        if was_1d:
            state = state.unsqueeze(0)  # [1, field_dim]
        
        # Project state through layer's lens
        projected = self.projection(state)  # [batch, pattern_dim]
        
        # Get active patterns and depths
        mask = self.active_mask
        active_patterns = self.patterns[mask]  # [N_active, pattern_dim]
        active_depths = self.depths[mask]      # [N_active]
        
        # Compute similarities: ξᵢᵀ W s for each active pattern
        # [batch, N_active] = [batch, pattern_dim] @ [pattern_dim, N_active]
        similarities = projected @ active_patterns.T
        
        # Correct energy formulation from Doc 002:
        # E = -(1/β) · log(Σ depth_i · exp(β · ξᵢᵀ W s))
        #   = -(1/β) · log(Σ exp(log(depth_i) + β · ξᵢᵀ W s))
        # So effective logits = log(depth_i) + β · similarity
        log_depths = torch.log(active_depths.clamp(min=1e-8)).unsqueeze(0)  # [1, N_active]
        scaled_sims = log_depths + self.beta * similarities
        
        # Log-sum-exp energy
        energy = -(1.0 / self.beta) * torch.logsumexp(scaled_sims, dim=-1)
        
        if was_1d:
            energy = energy.squeeze(0)
        
        return energy
    
    def retrieve(
        self, 
        state: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
        track_access: bool = True,
    ) -> torch.Tensor:
        """
        Retrieve from this memory layer (Hopfield retrieval = attention).
        
        retrieval = Σᵢ softmax(β · ξᵢᵀ W s · depth_i · gate_i) · ξᵢ
        
        This IS the gradient descent step on the energy function.
        The retrieved vector pulls the state toward the nearest/deepest basin.
        
        Args:
            state: Field state vector [batch, field_dim] or [field_dim]
            gate: Optional emotional gating [N_active] (0-1, from regulatory layer)
            
        Returns:
            Retrieved memory vector [batch, field_dim] or [field_dim]
        """
        if self.num_active == 0:
            return torch.zeros_like(state)
        
        was_1d = state.dim() == 1
        if was_1d:
            state = state.unsqueeze(0)
        
        # Project state through layer's lens
        # NOTE (from Alex): once projection learns away from identity,
        # consider normalizing projected state to keep cosine similarity
        # well-behaved. For Phase 1 with identity init this is fine.
        projected = self.projection(state)  # [batch, pattern_dim]
        
        # Get active patterns and depths
        mask = self.active_mask
        active_patterns = self.patterns[mask]  # [N_active, pattern_dim]
        active_depths = self.depths[mask]      # [N_active]
        
        # Compute attention weights (= Hopfield retrieval)
        similarities = projected @ active_patterns.T  # [batch, N_active]
        # Correct logit formulation: log(depth) + β·similarity
        # Matches gradient of energy function E = -(1/β)·log(Σ depth·exp(β·ξᵀWs))
        log_depths = torch.log(active_depths.clamp(min=1e-8)).unsqueeze(0)
        logits = log_depths + self.beta * similarities
        
        # Apply emotional gating if provided
        if gate is not None:
            # gate: [N_active], values 0-1
            # Multiply logits by gate (closed gate = very negative = ignored)
            logits = logits + torch.log(gate.unsqueeze(0).clamp(min=1e-8))
        
        # Softmax → attention weights
        weights = F.softmax(logits, dim=-1)  # [batch, N_active]
        
        # Weighted sum of patterns
        retrieved = weights @ active_patterns  # [batch, pattern_dim]
        
        # Track which patterns were accessed (for strengthening)
        # Skip during bulk verification to avoid O(N²) Python loops
        if track_access:
            with torch.no_grad():
                max_weights = weights.max(dim=0).values  # [N_active]
                self._record_access(mask, max_weights)
        
        if was_1d:
            retrieved = retrieved.squeeze(0)
        
        return retrieved
    
    def retrieve_settle(
        self,
        state: torch.Tensor,
        steps: int = 3,
        lam: float = 0.7,
        gate: Optional[torch.Tensor] = None,
        anneal_beta: bool = True,
    ) -> torch.Tensor:
        """
        Iterative basin settling: turn noisy cues into clean memory retrieval.
        
        Instead of one-shot attention (lookup), this iteratively pulls the
        query state deeper into the nearest energy basin. Each step:
            s = normalize((1-λ)*s + λ*retrieve(s))
        
        With optional β annealing (PR B from Alex):
            - Early steps use low β (broad attention, explore basins)
            - Later steps use full β (sharp attention, commit to winner)
            This prevents confident wrong-winner lock-in under noise.
        
        Args:
            state: Noisy/partial query [batch, dim] or [dim]
            steps: Number of settling iterations (3-5 typical)
            lam: Blend rate (0=no pull, 1=full replacement). 0.7 default.
            gate: Optional emotional gating
            anneal_beta: If True, ramp β from 0.5x to 1.0x over steps
            
        Returns:
            Settled state vector (should be close to nearest stored pattern)
        """
        if self.num_active == 0:
            return torch.zeros_like(state)
        
        was_1d = state.dim() == 1
        if was_1d:
            state = state.unsqueeze(0)
        
        s = state.clone()
        original_beta = self.beta
        
        for step in range(steps):
            # Beta annealing: start broad, end sharp
            if anneal_beta and steps > 1:
                # Linear ramp: 0.5β → 1.0β over steps
                progress = step / (steps - 1)  # 0.0 to 1.0
                self.beta = original_beta * (0.5 + 0.5 * progress)
            
            # One Hopfield retrieval step (no access tracking during settling)
            pull = self.retrieve(s, gate=gate, track_access=False)
            
            # Blend and renormalize
            s = F.normalize((1 - lam) * s + lam * pull, dim=-1)
        
        # Restore original beta
        self.beta = original_beta
        
        if was_1d:
            s = s.squeeze(0)
        
        return s
    
    def _record_access(
        self, 
        mask: torch.Tensor, 
        access_strengths: torch.Tensor,
        reinforce_rate: float = 0.01,
    ):
        """Record pattern access and strengthen basins with diminishing returns.
        
        Deep basins get less reinforcement than shallow ones to prevent
        rich-get-richer attractor calcification. Uses log-depth for dominance
        so one outlier doesn't suppress all other reinforcement.
        """
        active_indices = mask.nonzero(as_tuple=True)[0]
        active_depths = self.depths[mask]
        
        # Use log-depth for dominance calculation (Alex's suggestion)
        # This prevents one outlier from making everything else look tiny
        if active_depths.numel() > 0:
            log_depths = torch.log1p(active_depths)  # log(1 + depth)
            log_max = log_depths.max().clamp(min=1e-8)
        else:
            log_depths = None
            log_max = 1.0
        
        for i, idx in enumerate(active_indices):
            idx_item = idx.item()
            strength = access_strengths[i].item()
            
            if strength > 0.1:  # Only count meaningful access
                # Diminishing returns: deep basins get less reinforcement
                if log_depths is not None:
                    dominance = (log_depths[i] / log_max).item()
                    dampened_rate = reinforce_rate * (1.0 - 0.7 * dominance)
                else:
                    dampened_rate = reinforce_rate
                
                self.depths[idx_item] += dampened_rate * strength
                
                # Update metadata
                meta = self.pattern_metadata[idx_item]
                if meta is not None:
                    meta.last_accessed = time.time()
                    meta.access_count += 1
    
    def store_pattern(
        self,
        pattern: torch.Tensor,
        depth: float = 1.0,
        significance: float = 0.5,
        emotional_tag: Optional[torch.Tensor] = None,
        value_alignment: float = 0.0,
    ) -> Optional[int]:
        """
        Store a new memory pattern in this layer.
        
        If the layer is full, prunes the least significant pattern first.
        
        Args:
            pattern: The memory pattern vector [pattern_dim]
            depth: Initial basin depth
            significance: How significant this memory was at formation
            emotional_tag: Emotional context vector [17]
            value_alignment: How aligned with values (0-1)
            
        Returns:
            Index of stored pattern, or None if storage failed
        """
        # Normalize pattern to unit norm (prevents magnitude issues)
        pattern = F.normalize(pattern.detach(), dim=-1)
        
        # Find empty slot
        empty_mask = self.depths == 0
        empty_indices = empty_mask.nonzero(as_tuple=True)[0]
        
        if len(empty_indices) == 0:
            # Layer full — prune least significant pattern
            idx = self._prune_least_significant()
            if idx is None:
                return None
        else:
            idx = empty_indices[0].item()
        
        # Store pattern
        self.patterns[idx] = pattern.to(self.patterns.device)
        self.depths[idx] = depth
        
        if emotional_tag is not None:
            self.emotional_tags[idx] = emotional_tag.to(self.emotional_tags.device)
        
        # Create metadata
        now = time.time()
        self.pattern_metadata[idx] = MemoryPattern(
            index=idx,
            depth=depth,
            emotional_tag=emotional_tag,
            created_at=now,
            last_accessed=now,
            access_count=0,
            significance=significance,
            value_alignment=value_alignment,
        )
        
        return idx
    
    def _prune_least_significant(self) -> Optional[int]:
        """Remove the least significant active pattern. Returns freed index."""
        if self.num_active == 0:
            return None
        
        # Score: combination of depth, recency, access frequency
        active_mask = self.active_mask
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        
        min_score = float('inf')
        min_idx = None
        
        now = time.time()
        for idx in active_indices:
            idx_item = idx.item()
            meta = self.pattern_metadata[idx_item]
            if meta is None:
                # No metadata = definitely pruneable
                min_idx = idx_item
                break
            
            # Score: depth × recency × (1 + access_count)
            recency = 1.0 / (1.0 + (now - meta.last_accessed) / 60.0)
            score = self.depths[idx_item].item() * recency * (1 + meta.access_count)
            
            if score < min_score:
                min_score = score
                min_idx = idx_item
        
        if min_idx is not None:
            self._clear_slot(min_idx)
        
        return min_idx
    
    def _clear_slot(self, idx: int):
        """Clear a pattern slot."""
        self.patterns[idx] = 0
        self.depths[idx] = 0
        self.emotional_tags[idx] = 0
        self.pattern_metadata[idx] = None
    
    def decay_step(self):
        """
        Apply one step of decay to all active patterns.
        
        Uses multiplicative decay with significance floor:
        - Depth above the floor decays exponentially: excess *= (1 - rate)
        - Significance floor prevents important memories from fully fading
        - Mundane memories (sig < 0.5) have no floor and decay to zero
        
        Patterns below min_depth are pruned.
        Patterns below seed_threshold are flagged for Anamnesis seeding.
        
        Returns:
            List of patterns that should be seeded to Anamnesis before removal
        """
        self._step_count += 1
        seeds_needed = []
        
        active_mask = self.active_mask
        if not active_mask.any():
            return seeds_needed
        
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        
        SIG_FLOOR_THRESHOLD = 0.5  # Only memories above this get a floor
        
        for idx in active_indices:
            idx_item = idx.item()
            meta = self.pattern_metadata[idx_item]
            
            # Compute effective decay rate with emotional/value protection
            decay = self.decay_rate
            sig_floor = 0.0
            
            if meta is not None:
                # Emotional protection: intense memories decay slower
                if meta.emotional_tag is not None:
                    emotional_intensity = meta.emotional_tag.abs().mean().item()
                    decay *= (1.0 - 0.5 * min(emotional_intensity, 1.0))
                
                # Value protection: value-aligned memories decay slower
                decay *= (1.0 - 0.3 * meta.value_alignment)
                
                # Significance floor: important memories can't fully fade
                # Combined from significance AND value alignment (Alex's note)
                # sig=0.5 → floor=0.0, sig=1.0 → floor=0.5
                # value_alignment adds additional floor protection
                floor_from_sig = 0.0
                if meta.significance > SIG_FLOOR_THRESHOLD:
                    normalized_sig = (meta.significance - SIG_FLOOR_THRESHOLD) / (1.0 - SIG_FLOOR_THRESHOLD)
                    floor_from_sig = normalized_sig * 0.5
                floor_from_value = meta.value_alignment * 0.3  # max 0.3 from values
                sig_floor = max(floor_from_sig, floor_from_value)
            
            # Multiplicative decay on depth above floor
            # depth_above_floor *= (1 - effective_rate)
            current_depth = self.depths[idx_item].item()
            if current_depth > sig_floor:
                excess = current_depth - sig_floor
                excess *= (1.0 - decay)
                self.depths[idx_item] = sig_floor + excess
            # If somehow below floor (shouldn't happen), leave it
            
            # Check thresholds
            current_depth = self.depths[idx_item].item()
            
            if current_depth < self.min_depth:
                # Pattern has faded too much — check for seeding
                if current_depth > 0 and meta is not None:
                    # Candidate for Anamnesis seeding
                    seeds_needed.append({
                        'pattern': self.patterns[idx_item].clone(),
                        'metadata': meta,
                        'emotional_tag': self.emotional_tags[idx_item].clone(),
                    })
                # Remove pattern
                self._clear_slot(idx_item)
            elif current_depth < self.seed_threshold and meta is not None:
                if not meta.consolidated:
                    # Below seed threshold but not yet pruned — flag it
                    seeds_needed.append({
                        'pattern': self.patterns[idx_item].clone(),
                        'metadata': meta,
                        'emotional_tag': self.emotional_tags[idx_item].clone(),
                    })
        
        return seeds_needed
    
    def get_state_dict_persistent(self) -> dict:
        """
        Get the persistent state of this memory layer.
        Used for saving/loading across sessions.
        Includes projection weights — critical for retrieval consistency.
        """
        return {
            'patterns': self.patterns.clone(),
            'depths': self.depths.clone(),
            'emotional_tags': self.emotional_tags.clone(),
            'projection_state': self.projection.state_dict(),
            'metadata': [
                {
                    'depth': m.depth,
                    'created_at': m.created_at,
                    'last_accessed': m.last_accessed,
                    'access_count': m.access_count,
                    'significance': m.significance,
                    'value_alignment': m.value_alignment,
                    'consolidated': m.consolidated,
                } if m is not None else None
                for m in self.pattern_metadata
            ],
            'step_count': self._step_count,
        }
    
    def load_state_dict_persistent(self, state: dict):
        """Load persistent state from a saved checkpoint."""
        self.patterns.copy_(state['patterns'])
        self.depths.copy_(state['depths'])
        self.emotional_tags.copy_(state['emotional_tags'])
        # Restore projection weights — without this, retrieval breaks
        if 'projection_state' in state:
            self.projection.load_state_dict(state['projection_state'])
        self._step_count = state.get('step_count', 0)
        
        for i, meta_dict in enumerate(state.get('metadata', [])):
            if meta_dict is not None:
                self.pattern_metadata[i] = MemoryPattern(
                    index=i, **meta_dict
                )
            else:
                self.pattern_metadata[i] = None
    
    def __repr__(self):
        return (
            f"MemoryLayer("
            f"dim={self.pattern_dim}, "
            f"active={self.num_active}/{self.max_patterns}, "
            f"β={self.beta:.1f}, "
            f"decay={self.decay_rate})"
        )
