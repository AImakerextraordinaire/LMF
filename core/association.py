"""
ANIMA Living Memory Field - Association Matrix
Tracks associations between memory patterns (ridge-lowering).

When two patterns are activated together or in sequence, the energy ridge
between them lowers, making transitions easier. This creates associative
chains — one memory triggering the next.

Theory reference: Doc 002, Section 8
"""

import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time


class AssociationMatrix:
    """
    Sparse association tracking between memory patterns.
    
    Instead of a full O(N²) matrix, we track only the top-K associations
    per pattern. Associations strengthen with co-activation and decay
    over time.
    
    Associations are tracked as:
        (layer_a, idx_a) → (layer_b, idx_b) : strength
    """
    
    def __init__(self, config):
        self.config = config
        self.max_assoc_per_pattern = config.max_associations_per_pattern
        self.min_strength = config.min_association_strength
        self.decay_rate = config.association_decay
        self.coactivation_window = config.coactivation_window
        
        # Sparse association storage
        # Key: (layer, idx) → Dict of (layer, idx) → strength
        self._associations: Dict[Tuple[str, int], Dict[Tuple[str, int], float]] = defaultdict(dict)
        
        # Recent activation history for co-activation detection
        self._recent_activations: List[Tuple[str, int, float]] = []  # (layer, idx, time)
    
    def record_activation(
        self,
        layer: str,
        pattern_idx: int,
        field_state: Optional[torch.Tensor] = None,
    ):
        """
        Record that a pattern was activated. Check for co-activations
        with recently active patterns and form/strengthen associations.
        """
        now = time.time()
        current = (layer, pattern_idx)
        
        # Check for co-activations within the window
        window_start = now - self.coactivation_window
        
        active_in_window = [
            (l, i, t) for l, i, t in self._recent_activations
            if t > window_start and (l, i) != current
        ]
        
        for other_layer, other_idx, activation_time in active_in_window:
            other = (other_layer, other_idx)
            
            # Temporal proximity: closer in time = stronger association
            temporal_weight = 1.0 - (now - activation_time) / self.coactivation_window
            
            # Strengthen bidirectional association
            strength_delta = 0.1 * temporal_weight
            
            self._strengthen(current, other, strength_delta)
            self._strengthen(other, current, strength_delta)
        
        # Add to recent activations
        self._recent_activations.append((layer, pattern_idx, now))
        
        # Prune old activations
        self._recent_activations = [
            (l, i, t) for l, i, t in self._recent_activations
            if t > window_start
        ]
    
    def _strengthen(
        self,
        source: Tuple[str, int],
        target: Tuple[str, int],
        delta: float,
    ):
        """Strengthen association from source to target."""
        assoc = self._associations[source]
        
        if target in assoc:
            assoc[target] = min(1.0, assoc[target] + delta)
        elif len(assoc) < self.max_assoc_per_pattern:
            assoc[target] = delta
        else:
            # At capacity — replace weakest if new association would be stronger
            weakest_key = min(assoc, key=assoc.get)
            if delta > assoc[weakest_key]:
                del assoc[weakest_key]
                assoc[target] = delta
    
    def get_associations(
        self, 
        layer: str, 
        pattern_idx: int,
    ) -> List[Tuple[str, int, float]]:
        """Get all associations for a pattern, sorted by strength."""
        key = (layer, pattern_idx)
        assoc = self._associations.get(key, {})
        
        results = [
            (target[0], target[1], strength)
            for target, strength in assoc.items()
        ]
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def decay_step(self):
        """Apply decay to all associations. Prune weak ones."""
        to_remove = []
        
        for source, targets in self._associations.items():
            weak_targets = []
            for target, strength in targets.items():
                new_strength = strength - self.decay_rate
                if new_strength < self.min_strength:
                    weak_targets.append(target)
                else:
                    targets[target] = new_strength
            
            for t in weak_targets:
                del targets[t]
            
            if not targets:
                to_remove.append(source)
        
        for source in to_remove:
            del self._associations[source]
    
    def get_persistent_state(self) -> dict:
        """Serialize for saving."""
        serializable = {}
        for source, targets in self._associations.items():
            key_str = f"{source[0]}:{source[1]}"
            serializable[key_str] = {
                f"{t[0]}:{t[1]}": s for t, s in targets.items()
            }
        return {'associations': serializable}
    
    def load_persistent_state(self, saved: dict):
        """Deserialize from saved state."""
        self._associations.clear()
        for key_str, targets in saved.get('associations', {}).items():
            parts = key_str.split(':')
            source = (parts[0], int(parts[1]))
            for t_str, strength in targets.items():
                t_parts = t_str.split(':')
                target = (t_parts[0], int(t_parts[1]))
                self._associations[source][target] = strength
    
    @property
    def total_associations(self) -> int:
        """Total number of association edges."""
        return sum(len(targets) for targets in self._associations.values())
    
    def __repr__(self):
        return (
            f"AssociationMatrix("
            f"patterns={len(self._associations)}, "
            f"edges={self.total_associations})"
        )
