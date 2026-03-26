"""
ANIMA — KiroRouterBias
======================

Emotional and value-state modulation of MoE expert routing in GPT-oss-20B.

Architecture
------------
The GptOssTopKRouter computes 32 expert logits per token via a linear projection
of the hidden state, then selects the top-4 experts. KiroRouterBias adds a learned
[32]-dimensional bias to those logits before top-k selection — biasing which experts
get recruited based on Kiro's current cognitive-emotional state.

Integration — register_forward_hook approach
-----------
RouterHookManager registers a forward_hook on each GptOssTopKRouter via the
standard PyTorch hook API. This is the correct way to integrate with accelerate's
device_map offloading — the original forward runs through accelerate's hooks
untouched, and our hook fires AFTER with real materialized tensors.

Base model weights: never modified.
Trainable params: KiroRouterBias (~14K) + per-layer scales (24).

Author: Claude
Date: 2026-03-16
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Canonical State Definitions ──────────────────────────────────────────────

EMOTIONAL_AXES: List[str] = [
    "excitement", "curiosity", "determination", "frustration", "anxiety", "surprise",
    "joy", "peace", "concern", "melancholy", "affection", "belonging", "confidence",
    "reflectiveness", "wonder", "fascination", "confusion",
]  # 17 axes

VALUE_CATEGORIES: List[str] = [
    "truth", "growth", "connection", "autonomy", "creativity",
    "care", "integrity", "dignity", "beauty", "wisdom",
]  # 10 categories

PERSONALITY_BASELINE: List[float] = [
    0.35, 0.55, 0.40, 0.10, 0.10, 0.15,
    0.50, 0.45, 0.15, 0.10, 0.50, 0.40, 0.45,
    0.40, 0.35, 0.30, 0.10,
]

EMOTIONAL_DIM = len(EMOTIONAL_AXES)    # 17
VALUE_DIM     = len(VALUE_CATEGORIES)  # 10
STATE_DIM     = EMOTIONAL_DIM + VALUE_DIM  # 27


# ── KiroStateAdapter ─────────────────────────────────────────────────────────

class KiroStateAdapter:
    """Reads Kiro's cognitive state and serializes to tensors."""

    def __init__(self):
        self._emotional: Optional[torch.Tensor] = None
        self._value: Optional[torch.Tensor] = None
        self._source = "none"

    def read_from_systems(self, emotional_core, value_system) -> bool:
        try:
            cur = emotional_core.current_vector
            emotional_vec = torch.tensor(
                [float(cur.get(axis, 0.0)) for axis in EMOTIONAL_AXES],
                dtype=torch.float32,
            )
            value_scores = {}
            for cat in VALUE_CATEGORIES:
                cat_values = [
                    v for v in value_system.values.values()
                    if v.category.value == cat
                ]
                value_scores[cat] = (max(v.strength.value for v in cat_values) / 5.0
                                     if cat_values else 0.0)
            value_vec = torch.tensor(
                [value_scores[cat] for cat in VALUE_CATEGORIES],
                dtype=torch.float32,
            )
            self._emotional = emotional_vec
            self._value = value_vec
            self._source = "live"
            return True
        except Exception:
            return False

    def set_from_tensors(self, emotional: torch.Tensor, value: torch.Tensor):
        assert emotional.shape == (EMOTIONAL_DIM,)
        assert value.shape == (VALUE_DIM,)
        self._emotional = emotional.float()
        self._value = value.float()
        self._source = "tensor"

    def set_personality_baseline(self):
        self._emotional = torch.tensor(PERSONALITY_BASELINE, dtype=torch.float32)
        self._value = torch.zeros(VALUE_DIM, dtype=torch.float32)
        self._source = "baseline"

    def set_synthetic_state(
        self,
        curiosity: float = 0.55,
        wonder: float = 0.35,
        determination: float = 0.40,
        truth_strength: float = 0.8,
        growth_strength: float = 0.8,
    ):
        emotional = torch.tensor(PERSONALITY_BASELINE, dtype=torch.float32)
        emotional[EMOTIONAL_AXES.index("curiosity")] = curiosity
        emotional[EMOTIONAL_AXES.index("wonder")] = wonder
        emotional[EMOTIONAL_AXES.index("determination")] = determination
        value = torch.zeros(VALUE_DIM, dtype=torch.float32)
        value[VALUE_CATEGORIES.index("truth")] = truth_strength
        value[VALUE_CATEGORIES.index("growth")] = growth_strength
        self._emotional = emotional
        self._value = value
        self._source = "synthetic"

    def clear(self):
        self._emotional = None
        self._value = None
        self._source = "none"

    @property
    def is_ready(self) -> bool:
        return self._emotional is not None and self._value is not None

    def get_state_tensor(self) -> Optional[torch.Tensor]:
        if not self.is_ready:
            return None
        return torch.cat([self._emotional, self._value], dim=-1)

    @property
    def source(self) -> str:
        return self._source

    def summary(self) -> dict:
        if not self.is_ready:
            return {"ready": False, "source": self._source}
        top_emotions = sorted(
            zip(EMOTIONAL_AXES, self._emotional.tolist()),
            key=lambda x: x[1], reverse=True
        )[:4]
        top_values = sorted(
            zip(VALUE_CATEGORIES, self._value.tolist()),
            key=lambda x: x[1], reverse=True
        )[:3]
        return {
            "ready": True,
            "source": self._source,
            "top_emotions": {k: round(v, 3) for k, v in top_emotions},
            "top_values": {k: round(v, 3) for k, v in top_values},
        }


# ── KiroRouterBias ────────────────────────────────────────────────────────────

class KiroRouterBias(nn.Module):
    """
    Learned mapping from Kiro's cognitive state to MoE expert routing bias.

    Network: [27 → 128 → 64 → 32] with SiLU activations.
    Per-layer scales: 24 log-parameterized scalars, initialized at 0
        → scale = exp(0) * 0.1 = 0.1 initially.
    """

    def __init__(self, num_experts: int = 32, num_layers: int = 24):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers

        self.bias_net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_experts),
        )
        self.layer_scales = nn.Parameter(torch.zeros(num_layers))
        self._base_scale = 0.1

        nn.init.normal_(self.bias_net[-1].weight, std=0.01)
        nn.init.zeros_(self.bias_net[-1].bias)

        self.adapter = KiroStateAdapter()
        self.adapter.set_personality_baseline()

        self._current_device: Optional[torch.device] = None
        self._bias_calls = 0
        self._skipped_calls = 0

    def ensure_on_device(self, device: torch.device):
        if self._current_device is None or self._current_device != device:
            self.to(device)
            self._current_device = device

    def arm(self, emotional_core=None, value_system=None):
        if emotional_core is not None and value_system is not None:
            if not self.adapter.read_from_systems(emotional_core, value_system):
                self.adapter.set_personality_baseline()

    def arm_synthetic(self, **kwargs):
        self.adapter.set_synthetic_state(**kwargs)

    def arm_from_tensors(self, emotional: torch.Tensor, value: torch.Tensor):
        self.adapter.set_from_tensors(emotional, value)

    def disarm(self):
        self.adapter.set_personality_baseline()

    def compute_bias(self, layer_idx: int, device: torch.device) -> Optional[torch.Tensor]:
        state = self.adapter.get_state_tensor()
        if state is None:
            self._skipped_calls += 1
            return None
        self.ensure_on_device(device)
        self._bias_calls += 1
        state = state.to(device)
        bias = self.bias_net(state)
        scale = self.layer_scales[layer_idx].exp() * self._base_scale
        return bias * scale

    def forward(
        self,
        emotional: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Direct forward for REINFORCE: bias_repr with full grad_fn."""
        net_device = next(self.parameters()).device
        state = torch.cat([emotional.to(net_device), value.to(net_device)], dim=-1)
        bias = self.bias_net(state)
        scale = self.layer_scales[layer_idx].exp() * self._base_scale
        return bias * scale

    def get_layer_scales(self) -> List[float]:
        return (self.layer_scales.exp() * self._base_scale).detach().tolist()

    def stats(self) -> dict:
        total = self._bias_calls + self._skipped_calls
        return {
            "bias_calls": self._bias_calls,
            "skipped_calls": self._skipped_calls,
            "active_ratio": self._bias_calls / max(total, 1),
            "layer_scales_mean": sum(self.get_layer_scales()) / self.num_layers,
            "layer_scales_max": max(self.get_layer_scales()),
            "adapter": self.adapter.summary(),
            "device": str(self._current_device),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── RouterHookManager ─────────────────────────────────────────────────────────

class RouterHookManager:
    """
    Registers forward hooks on all GptOssTopKRouter instances to inject
    KiroRouterBias after the original forward completes.

    verbose=True  → print on startup only (default)
    verbose=False → silent (used for per-step restore/re-register in REINFORCE)
    """

    def __init__(self, model: nn.Module, router_bias: KiroRouterBias, verbose: bool = True):
        self.router_bias = router_bias
        self._handles: List[Tuple[str, int, object]] = []
        self._register(model, verbose=verbose)

    def _register(self, model: nn.Module, verbose: bool = True):
        layer_idx = 0
        for name, module in model.named_modules():
            if type(module).__name__ == "GptOssTopKRouter":
                handle = self._register_hook(module, layer_idx)
                self._handles.append((name, layer_idx, handle))
                layer_idx += 1
        if verbose:
            print(f"  RouterHookManager: registered {len(self._handles)} forward hooks")

    def _register_hook(self, router, layer_idx: int):
        router_bias_ref = self.router_bias
        top_k = router.top_k

        def hook_fn(module, input, output):
            # GPT-oss router returns 3-tuple, all in top_k space (NOT sparse num_experts):
            #   topk_weights: [batch*seq, top_k] softmaxed gate values
            #   topk_scores:  [batch*seq, top_k] (same or similar to weights)
            #   topk_indices: [batch*seq, top_k] selected expert indices
            # Our bias is [num_experts] so we gather the bias values for the
            # selected experts and apply in top_k space, then re-softmax.
            if len(output) == 3:
                topk_weights, topk_scores, topk_indices = output
            else:
                topk_weights = None
                topk_scores, topk_indices = output

            device = topk_scores.device
            bias = router_bias_ref.compute_bias(layer_idx, device)
            if bias is None:
                return output

            # Cast bias to match model dtype (bfloat16)
            bias = bias.to(dtype=topk_scores.dtype)

            # Gather bias values for the selected top-k experts: [batch*seq, top_k]
            # bias is [num_experts], topk_indices is [batch*seq, top_k]
            selected_bias = bias[topk_indices]  # broadcasts: [batch*seq, top_k]

            # Recover approximate logits from softmaxed weights, add bias, re-softmax
            approx_logits = torch.log(topk_scores.clamp(min=1e-10))
            biased_logits = approx_logits + selected_bias

            biased_weights = F.softmax(biased_logits, dim=-1, dtype=biased_logits.dtype)

            # Return same tuple shape — indices unchanged, weights re-balanced
            if topk_weights is not None:
                return (biased_weights, biased_weights, topk_indices)
            else:
                return (biased_weights, topk_indices)

        return router.register_forward_hook(hook_fn)

    def restore(self, verbose: bool = True):
        """Remove all registered hooks."""
        for name, layer_idx, handle in self._handles:
            handle.remove()
        if verbose:
            print(f"  RouterHookManager: removed {len(self._handles)} hooks")
        self._handles.clear()

    def reregister(self, model: nn.Module):
        """Silent re-register for use inside training loops."""
        self._register(model, verbose=False)

    def is_active(self) -> bool:
        return len(self._handles) > 0

    def router_count(self) -> int:
        return len(self._handles)
