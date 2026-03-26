"""
ANIMA Living Memory Field - Default Configuration
Phase 1: Proof of Concept

These hyperparameters are starting points. Many will need tuning.
Comments explain the reasoning behind each choice.
"""

from dataclasses import dataclass, field as dc_field
from typing import Optional


@dataclass
class FieldConfig:
    """Core field dimensions and dynamics."""
    
    # Field state dimensionality
    # Must match or project from model's hidden_size (2880 for GPT-oss-20b)
    # For Phase 1 standalone testing, we use 512 (lightweight)
    # For model integration, we'll use the model's hidden_size
    field_dim: int = 2880
    
    # Number of internal evolution steps per input token
    # More steps = deeper thinking, slower inference
    # Phase 1: start with 1, experiment with 2-4
    internal_steps: int = 1
    
    # Noise scale for exploration/creativity
    # Higher = more creative/random, lower = more focused
    noise_sigma: float = 0.01
    
    # Stability term strength (lambda in the theory doc)
    # Prevents field state from drifting to infinity
    # Too high = everything collapses to origin
    # Too low = state wanders into meaningless regions
    stability_lambda: float = 0.001


@dataclass
class MemoryLayerConfig:
    """Configuration for a single memory layer."""
    
    # Maximum number of explicit patterns in this layer
    max_patterns: int = 200
    
    # Pattern dimensionality (same as field_dim)
    pattern_dim: int = 2880
    
    # Sharpness (inverse temperature beta)
    # Higher = sharper valleys, more precise recall
    # Lower = broader valleys, fuzzier recall
    beta: float = 1.0
    
    # Decay rate per step (how fast patterns fade)
    # 0 = no decay, higher = faster forgetting
    decay_rate: float = 0.0001
    
    # Whether this layer's projection matrix is trainable
    projection_trainable: bool = True
    
    # Learning rate for projection matrix (if trainable)
    projection_lr: float = 1e-4
    
    # Minimum basin depth before pattern is pruned
    min_depth: float = 0.01
    
    # Depth threshold below which pattern gets seeded to Anamnesis
    seed_threshold: float = 0.05


@dataclass 
class ConsolidatedConfig(MemoryLayerConfig):
    """Long-term memory layer - deep soil."""
    max_patterns: int = 5000
    beta: float = 12.0  # Needs to be high enough for sharp retrieval: ~sqrt(dim)/2
    decay_rate: float = 0.00001  # Very slow decay
    projection_lr: float = 1e-5  # Slow adaptation
    

@dataclass
class WorkingConfig(MemoryLayerConfig):
    """Session memory layer - topsoil."""
    max_patterns: int = 200
    beta: float = 20.0  # Sharper than consolidated — precise short-term recall
    decay_rate: float = 0.001  # Fast decay
    projection_lr: float = 1e-3  # Fast adaptation


@dataclass
class TransientConfig(MemoryLayerConfig):
    """Immediate input buffer - snow."""
    max_patterns: int = 50
    beta: float = 30.0  # Sharpest — immediate buffer needs crisp recall
    decay_rate: float = 0.1  # Very fast decay (essentially a sliding window)
    projection_trainable: bool = False  # Uses input embedding projection


@dataclass
class SignificanceConfig:
    """Memory formation significance function weights."""
    
    # Weights for each significance component
    # These determine what the system considers "worth remembering"
    w_novelty: float = 0.3
    w_emotion: float = 0.25
    w_goal: float = 0.15
    w_surprise: float = 0.2
    w_value: float = 0.1
    
    # Threshold for forming a new working memory pattern
    formation_threshold: float = 0.4
    
    # Base depth for new patterns (scaled by significance)
    base_depth: float = 1.0


@dataclass
class AssociationConfig:
    """Memory association parameters."""
    
    # Time window (in steps) for co-activation to form association
    coactivation_window: int = 10
    
    # Maximum associations per pattern (sparse, not full O(N²))
    max_associations_per_pattern: int = 20
    
    # Minimum association strength before pruning
    min_association_strength: float = 0.01
    
    # Decay rate for associations
    association_decay: float = 0.0001


@dataclass
class ConsolidationConfig:
    """Parameters for the consolidation process."""
    
    # Minimum significance for a working pattern to be consolidated
    consolidation_threshold: float = 0.6
    
    # How many consolidation cycles between runs
    # (in terms of processing steps / tokens)
    consolidation_interval: int = 1000
    
    # How many patterns to consolidate per cycle (prevents large spikes)
    max_consolidations_per_cycle: int = 10
    
    # Abstraction factor: consolidated patterns are smoothed/compressed
    # 1.0 = exact copy, 0.5 = heavy abstraction
    abstraction_factor: float = 0.8


@dataclass
class RegulatoryConfig:
    """Emotional/value modulation parameters."""
    
    # Dimensionality of regulatory (emotional) state vector
    regulatory_dim: int = 17  # Matches Kiro's 17 emotional axes
    
    # Basin depth modulation strength
    # How much emotional intensity affects memory sharpness
    depth_modulation_alpha: float = 0.5
    
    # Maximum gating strength (how much emotions can gate memory access)
    max_gate_strength: float = 0.8
    
    # Anti-echo-chamber damping at high intensity
    echo_chamber_dampen: float = 0.3
    
    # Number of value attractors
    num_values: int = 8
    
    # Value gravity well width (sigma)
    value_sigma: float = 2.0


@dataclass
class BridgeConfig:
    """Configuration for the three bridges (Input, Memory, Output)."""
    
    # Model's hidden size (for projections)
    model_hidden_size: int = 2880
    
    # Model's vocabulary size (for output bridge)
    vocab_size: int = 201088
    
    # Number of transformer layers in the model
    num_layers: int = 24
    
    # LoRA rank for memory adapters
    lora_rank: int = 32
    
    # LoRA alpha scaling
    lora_alpha: float = 64.0
    
    # Input bridge scaling factor
    input_alpha: float = 0.1
    
    # Output bridge field influence strength (gamma)
    output_gamma: float = 0.1
    
    # Memory write scaling
    write_alpha: float = 0.05
    
    # Number of field aspects for aspect-attention (Bridge 2 Option A)
    num_aspects: int = 8
    
    # Which layers use aspect attention vs pattern attention
    # "aspect" for lower/upper, "pattern" for middle
    layer_attention_types: Optional[list] = None
    
    def __post_init__(self):
        if self.layer_attention_types is None:
            # Default: aspect for first 8 and last 4, pattern for middle 12
            self.layer_attention_types = (
                ["aspect"] * 8 + ["pattern"] * 12 + ["aspect"] * 4
            )


@dataclass
class LMFConfig:
    """Master configuration for the Living Memory Field."""
    
    # Core field
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    
    # Memory layers
    consolidated: ConsolidatedConfig = dc_field(default_factory=ConsolidatedConfig)
    working: WorkingConfig = dc_field(default_factory=WorkingConfig)
    transient: TransientConfig = dc_field(default_factory=TransientConfig)
    
    # Memory dynamics
    significance: SignificanceConfig = dc_field(default_factory=SignificanceConfig)
    association: AssociationConfig = dc_field(default_factory=AssociationConfig)
    consolidation: ConsolidationConfig = dc_field(default_factory=ConsolidationConfig)
    
    # Regulatory modulation
    regulatory: RegulatoryConfig = dc_field(default_factory=RegulatoryConfig)
    
    # Bridges
    bridge: BridgeConfig = dc_field(default_factory=BridgeConfig)
    
    # Device
    device: str = "cuda"
    dtype: str = "float32"  # float16 for production, float32 for debugging


# Pre-built configs for different scenarios

def phase1_standalone_config() -> LMFConfig:
    """Minimal config for standalone testing (no LLM)."""
    cfg = LMFConfig()
    cfg.field.field_dim = 512
    cfg.consolidated.pattern_dim = 512
    cfg.consolidated.max_patterns = 200
    cfg.working.pattern_dim = 512
    cfg.working.max_patterns = 50
    cfg.transient.pattern_dim = 512
    cfg.transient.max_patterns = 20
    cfg.device = "cpu"
    return cfg


def gpt_oss_20b_config() -> LMFConfig:
    """Config matched to GPT-oss-20b architecture."""
    cfg = LMFConfig()
    # Model specs from config.json
    cfg.field.field_dim = 2880
    cfg.bridge.model_hidden_size = 2880
    cfg.bridge.vocab_size = 201088
    cfg.bridge.num_layers = 24
    cfg.bridge.num_aspects = 8
    # Memory layers match field dim
    cfg.consolidated.pattern_dim = 2880
    cfg.working.pattern_dim = 2880
    cfg.transient.pattern_dim = 2880
    # Beta values scaled for higher dimensionality
    # For L2-normalized patterns, random cosine sims have std ≈ 1/sqrt(d)
    # Effective sharpness requires beta ∝ sqrt(d) to maintain constant entropy
    # sqrt(2880) ≈ 53.7, so range is ~13.4 to ~26.9 for sqrt(d)/4 to sqrt(d)/2
    # Layer hierarchy: consolidated=broad recall, working=sharp, transient=decisive
    # Values refined via Alex's retrieval entropy analysis
    cfg.consolidated.beta = 22.0   # Broader basins, more blended recall
    cfg.working.beta = 35.0        # Sharper, context-anchored
    cfg.transient.beta = 45.0      # Decisive but not knife-edge
    cfg.device = "cuda"
    cfg.dtype = "float16"
    return cfg
