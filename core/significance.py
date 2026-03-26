"""
ANIMA Living Memory Field - Significance Detector
Determines whether an input is worth forming a memory for.

Not every experience becomes a memory. The significance function evaluates
novelty, emotional intensity, goal relevance, surprise, and value alignment
to decide what crosses the threshold.

Theory reference: Doc 002, Section 7.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class SignificanceDetector(nn.Module):
    """
    Evaluates how significant an input is — whether it deserves to become
    a memory pattern in the working layer.
    
    significance = w_novelty · novelty 
                 + w_emotion · emotional_intensity
                 + w_surprise · prediction_error
                 + w_value · value_alignment
                 + w_goal · goal_relevance
    """
    
    def __init__(self, field_dim: int, config):
        super().__init__()
        self.config = config
        self.field_dim = field_dim
        
        # Weights for each component (learnable)
        self.component_weights = nn.Parameter(torch.tensor([
            config.w_novelty,
            config.w_emotion,
            config.w_surprise,
            config.w_value,
            config.w_goal,
        ]))
        
        # Prediction network: given current state, what do we expect next?
        # Used for computing prediction error (surprise)
        self.predictor = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.SiLU(),
            nn.Linear(field_dim, field_dim),
        )
        
        # Value alignment scorer
        # Projects input into value space for comparison
        self.value_projector = nn.Linear(field_dim, field_dim, bias=False)
    
    def evaluate(
        self,
        input_embedding: torch.Tensor,
        field_state: torch.Tensor,
        regulatory_state: Optional[torch.Tensor] = None,
        memory_layers: Optional[List] = None,
        goal_embedding: Optional[torch.Tensor] = None,
        value_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Evaluate the significance of an input.
        
        Args:
            input_embedding: The input to evaluate [field_dim]
            field_state: Current field state [field_dim]
            regulatory_state: Current emotional state [reg_dim]
            memory_layers: List of MemoryLayer objects for novelty computation
            goal_embedding: Current active goal representation [field_dim]
            value_embeddings: Value attractor centers [num_values, field_dim]
            
        Returns:
            (significance_score, component_breakdown)
            significance_score: scalar tensor 0-1
            component_breakdown: dict of individual component values
        """
        components = {}
        
        # === Novelty ===
        # How different is this from what we already know?
        novelty = self._compute_novelty(input_embedding, memory_layers)
        components['novelty'] = novelty
        
        # === Emotional intensity ===
        # How emotionally activated are we right now?
        if regulatory_state is not None:
            emotional_intensity = regulatory_state.abs().mean()
        else:
            emotional_intensity = torch.tensor(0.0, device=input_embedding.device)
        components['emotion'] = emotional_intensity
        
        # === Prediction error (surprise) ===
        surprise = self._compute_surprise(input_embedding, field_state)
        components['surprise'] = surprise
        
        # === Value alignment ===
        if value_embeddings is not None:
            value_align = self._compute_value_alignment(input_embedding, value_embeddings)
        else:
            value_align = torch.tensor(0.0, device=input_embedding.device)
        components['value'] = value_align
        
        # === Goal relevance ===
        if goal_embedding is not None:
            goal_rel = F.cosine_similarity(
                input_embedding.unsqueeze(0), 
                goal_embedding.unsqueeze(0),
            ).squeeze()
        else:
            goal_rel = torch.tensor(0.0, device=input_embedding.device)
        components['goal'] = goal_rel
        
        # === Weighted combination ===
        # Renormalize weights over ACTIVE components only.
        # When emotion/value/goal aren't provided (=0.0), their softmax weight
        # would be multiplied by zero - wasting weight budget.
        # Instead, redistribute their weight to the active components.
        
        component_values = torch.stack([
            novelty, emotional_intensity, surprise, value_align, goal_rel
        ])
        
        # Clamp components to [0, 1] range
        component_values = component_values.clamp(0.0, 1.0)
        
        # Active mask: components that can meaningfully contribute.
        # A component is "active" only if it has a real nonzero value,
        # not just if its argument was passed. This prevents zero-valued
        # components (e.g., a zero regulatory state tensor) from stealing
        # weight budget via softmax while contributing nothing.
        active_mask = torch.tensor([
            True,                                                    # novelty (always computed)
            emotional_intensity.item() > 1e-6,                      # emotion (need real signal)
            True,                                                    # surprise (always computed)
            value_embeddings is not None and value_align.item() > 0, # value (need embeddings + signal)
            goal_embedding is not None and goal_rel.item() > 0,     # goal (need embedding + signal)
        ], device=input_embedding.device)
        
        # Softmax only over active components
        raw_weights = self.component_weights.clone()
        raw_weights[~active_mask] = float('-inf')  # Zero out inactive in softmax
        weights = F.softmax(raw_weights, dim=0)
        
        significance = (weights * component_values).sum()
        
        return significance, components
    
    def _compute_novelty(
        self, 
        input_embedding: torch.Tensor, 
        memory_layers: Optional[List] = None,
    ) -> torch.Tensor:
        """
        Novelty = how different the input is from all known patterns.
        
        1 - max_similarity = high novelty if nothing matches well.
        """
        if memory_layers is None or len(memory_layers) == 0:
            return torch.tensor(1.0, device=input_embedding.device)  # Everything is novel
        
        max_similarity = torch.tensor(0.0, device=input_embedding.device)
        
        for layer in memory_layers:
            if layer.num_active == 0:
                continue
            
            active_patterns, active_depths = layer.get_active_patterns()
            
            # Cosine similarity between input and each pattern
            input_norm = F.normalize(input_embedding.unsqueeze(0), dim=-1)
            patterns_norm = F.normalize(active_patterns, dim=-1)
            
            similarities = (input_norm @ patterns_norm.T).squeeze(0)  # [N_active]
            
            # Weight by normalized depth so novelty stays in [0,1]
            # Divide by max depth so one outlier doesn't dominate
            depth_max = active_depths.max().clamp(min=1e-8)
            normalized_depths = active_depths / depth_max  # [0, 1] range
            weighted_sims = similarities * normalized_depths
            
            layer_max = weighted_sims.max()
            max_similarity = torch.max(max_similarity, layer_max)
        
        # Novelty: inverse of max similarity
        novelty = 1.0 - max_similarity.clamp(0.0, 1.0)
        
        return novelty
    
    def _compute_surprise(
        self, 
        input_embedding: torch.Tensor, 
        field_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Surprise = prediction error.
        
        The predictor tries to guess the next input based on current state.
        High error = surprising input.
        
        Uses cosine distance on normalized vectors so surprise measures
        directional prediction quality, not magnitude. This prevents
        surprise from saturating at ~0.38 when inputs are small.
        """
        predicted = self.predictor(field_state)
        
        # Normalize both vectors so we measure direction, not magnitude
        pred_norm = F.normalize(predicted, dim=-1)
        input_norm = F.normalize(input_embedding.detach(), dim=-1)
        
        # Cosine distance: 0 = perfect prediction, 1 = orthogonal, 2 = opposite
        cosine_sim = F.cosine_similarity(pred_norm.unsqueeze(0), input_norm.unsqueeze(0)).squeeze()
        cosine_distance = 1.0 - cosine_sim  # [0, 2] range
        
        # Map to [0, 1]: distance of 0 = no surprise, distance >= 1 = max surprise
        surprise = cosine_distance.clamp(0.0, 1.0)
        
        return surprise
    
    def _compute_value_alignment(
        self,
        input_embedding: torch.Tensor,
        value_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        How aligned is this input with core values?
        
        Projects input into value space and measures proximity
        to value attractor centers.
        """
        projected = self.value_projector(input_embedding)
        
        # Cosine similarity with each value center
        proj_norm = F.normalize(projected.unsqueeze(0), dim=-1)
        values_norm = F.normalize(value_embeddings, dim=-1)
        
        similarities = (proj_norm @ values_norm.T).squeeze(0)
        
        # Max alignment across values
        max_alignment = similarities.max().clamp(0.0, 1.0)
        
        return max_alignment
