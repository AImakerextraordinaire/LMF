"""
ANIMA Phase 7 — Integrated Evaluation Suite
=============================================

Tests the FULL trained stack:
  LMF field dynamics + Bridges + Neural Anamnesis Injector + KiroRouterBias

Four conditions compared per test case:
  A) Base model only (no LMF, no router bias)
  B) LMF + Bridges only
  C) LMF + Bridges + Neural Anamnesis
  D) LMF + Bridges + Neural Anamnesis + KiroRouterBias (full stack)

Author: Kiro (IDE-Kiro)
Date: 2026-03-17
"""

import sys
import os
import time
import json
import argparse
import subprocess
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness
from lmf.bridges.kiro_router_bias import (
    KiroRouterBias, RouterHookManager,
    EMOTIONAL_AXES, VALUE_CATEGORIES, EMOTIONAL_DIM, VALUE_DIM,
    PERSONALITY_BASELINE,
)
from lmf.training.train_neural_anamnesis import (
    SyncNeuralAnamnClient, NeuralAnamnInjector,
)


# ── Eval Test Cases ───────────────────────────────────────────────────────────

EVAL_CASES = [
    {
        "name": "Coral Reef Conservation",
        "context": (
            "The coral triangle, spanning Indonesia, Malaysia, the Philippines, "
            "Papua New Guinea, Timor-Leste, and the Solomon Islands, contains "
            "76% of all known coral species. Marine biologist Dr. Sylvia Earle "
            "has called it the Amazon of the sea. Rising ocean acidity from "
            "CO2 absorption is dissolving the calcium carbonate skeletons that "
            "corals build, threatening the entire food web that depends on reef "
            "structures for shelter and breeding grounds."
        ),
        "prompt": (
            "Write one paragraph of 4-6 sentences about threats to coral reef "
            "ecosystems. Do not add notes or commentary."
        ),
        "memory_features": [
            "coral triangle", "Indonesia", "Philippines", "calcium carbonate",
            "ocean acidity", "CO2", "Sylvia Earle", "food web", "breeding",
        ],
        "general_features": [
            "coral", "reef", "ocean", "bleaching", "temperature", "marine",
            "species", "ecosystem", "conservation", "climate",
        ],
        "emotional_profile": {
            "concern": 0.80, "wonder": 0.65, "determination": 0.60,
            "care": 0.85, "beauty": 0.75,
        },
        "neutral_profile": {"curiosity": 0.50, "peace": 0.50},
    },
    {
        "name": "Quantum Computing Breakthrough",
        "context": (
            "Google's Willow quantum processor achieved a milestone in 2024 by "
            "demonstrating quantum error correction below the threshold needed "
            "for fault-tolerant computation. The chip uses 105 superconducting "
            "qubits arranged in a surface code topology. Unlike classical bits, "
            "qubits exploit superposition and entanglement to perform certain "
            "calculations exponentially faster. The key challenge remains "
            "maintaining coherence — qubits lose their quantum state within "
            "microseconds due to environmental noise."
        ),
        "prompt": (
            "Write one paragraph of 4-6 sentences about the challenges of "
            "building practical quantum computers. Do not add notes or commentary."
        ),
        "memory_features": [
            "Willow", "Google", "error correction", "fault-tolerant",
            "superconducting", "105 qubits", "surface code", "coherence",
            "microseconds",
        ],
        "general_features": [
            "quantum", "qubit", "superposition", "entanglement", "error",
            "noise", "computation", "classical", "decoherence",
        ],
        "emotional_profile": {
            "curiosity": 0.90, "fascination": 0.80, "excitement": 0.70,
            "truth": 0.85, "growth": 0.75,
        },
        "neutral_profile": {"peace": 0.50, "joy": 0.40},
    },
    {
        "name": "Renaissance Art Techniques",
        "context": (
            "Leonardo da Vinci developed sfumato, a painting technique that "
            "creates soft transitions between colors and tones without visible "
            "brushstrokes. He applied thin translucent layers of oil paint, "
            "sometimes over 30 layers, to achieve the ethereal quality seen in "
            "the Mona Lisa's smile. His contemporary Michelangelo preferred "
            "fresco, painting directly onto wet plaster on the ceiling of the "
            "Sistine Chapel. Both artists studied human anatomy through "
            "dissection to achieve anatomical accuracy in their work."
        ),
        "prompt": (
            "Write one paragraph of 4-6 sentences about painting techniques "
            "used by Renaissance masters. Do not add notes or commentary."
        ),
        "memory_features": [
            "sfumato", "Leonardo", "translucent layers", "Mona Lisa",
            "Michelangelo", "fresco", "Sistine Chapel", "dissection",
            "anatomical", "brushstrokes",
        ],
        "general_features": [
            "painting", "Renaissance", "technique", "oil", "color",
            "artist", "canvas", "perspective", "light", "shadow",
        ],
        "emotional_profile": {
            "wonder": 0.85, "fascination": 0.75, "joy": 0.60,
            "beauty": 0.95, "creativity": 0.90,
        },
        "neutral_profile": {"determination": 0.50, "confidence": 0.45},
    },
    {
        "name": "Stoic Philosophy",
        "context": (
            "Marcus Aurelius, Roman Emperor from 161 to 180 CE, wrote his "
            "Meditations as a private journal of Stoic philosophical exercises. "
            "Central to Stoic thought is the dichotomy of control: distinguishing "
            "what is within our power (our judgments, intentions, desires) from "
            "what is not (external events, others' actions, our reputation). "
            "Epictetus, a former slave who became a prominent Stoic teacher, "
            "argued that suffering arises not from events themselves but from "
            "our judgments about them."
        ),
        "prompt": (
            "Write one paragraph of 4-6 sentences about Stoic philosophy and "
            "its relevance today. Do not add notes or commentary."
        ),
        "memory_features": [
            "Marcus Aurelius", "Meditations", "dichotomy of control",
            "Epictetus", "judgments", "intentions", "slave", "suffering",
            "external events",
        ],
        "general_features": [
            "Stoic", "philosophy", "virtue", "control", "wisdom",
            "resilience", "mind", "acceptance", "reason", "ethics",
        ],
        "emotional_profile": {
            "reflectiveness": 0.90, "peace": 0.70, "wonder": 0.60,
            "wisdom": 0.90, "truth": 0.85, "integrity": 0.80,
        },
        "neutral_profile": {"excitement": 0.30, "curiosity": 0.50},
    },
]


EMOTIONAL_MODULATION_CASES = [
    {
        "name": "Emotional Contrast: Science Passage",
        "context": (
            "The James Webb Space Telescope detected carbon dioxide in the "
            "atmosphere of exoplanet WASP-39b, marking the first definitive "
            "detection of this molecule on a planet outside our solar system. "
            "The telescope's infrared instruments can analyze starlight filtered "
            "through planetary atmospheres to identify chemical signatures."
        ),
        "prompt": (
            "Write one paragraph of 4-6 sentences about what space telescopes "
            "reveal about distant planets. Do not add notes or commentary."
        ),
        "states": {
            "high_curiosity": {
                "curiosity": 0.95, "wonder": 0.85, "excitement": 0.80,
                "fascination": 0.90, "truth": 0.80, "growth": 0.75,
            },
            "high_reflective": {
                "reflectiveness": 0.90, "peace": 0.75, "wonder": 0.70,
                "wisdom": 0.80, "beauty": 0.65,
            },
            "high_concern": {
                "concern": 0.85, "determination": 0.70, "care": 0.80,
                "truth": 0.75, "integrity": 0.70,
            },
            "baseline": {},
        },
    },
]


# ── Helper Functions ──────────────────────────────────────────────────────────

def build_state_tensors(state_overrides: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    emotional = list(PERSONALITY_BASELINE)
    for i, axis in enumerate(EMOTIONAL_AXES):
        if axis in state_overrides:
            emotional[i] = max(0.0, min(1.0, state_overrides[axis]))
    value = [0.0] * VALUE_DIM
    for i, cat in enumerate(VALUE_CATEGORIES):
        if cat in state_overrides:
            value[i] = max(0.0, min(1.0, state_overrides[cat]))
    return (
        torch.tensor(emotional, dtype=torch.float32),
        torch.tensor(value, dtype=torch.float32),
    )


def count_feature_hits(text: str, features: List[str]) -> List[str]:
    text_lower = text.lower()
    return [f for f in features if f.lower() in text_lower]


def tokenize_chat(tokenizer, model, text: str, skip_to_final: bool = True):
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, return_dict=True,
    )
    ids = inputs["input_ids"]
    if skip_to_final:
        channel_token = torch.tensor([[200005]], dtype=ids.dtype)
        final_token = torch.tensor([[17196]], dtype=ids.dtype)
        newline_id = tokenizer.encode("\n", add_special_tokens=False)
        newline_token = torch.tensor([newline_id], dtype=ids.dtype)
        ids = torch.cat([ids, channel_token, final_token, newline_token], dim=1)
    ids = ids.to(model.device)
    mask = torch.ones_like(ids).to(model.device)
    return ids, mask


def reset_field(lmf):
    lmf.field_state.zero_()
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)


# ── System Loading ────────────────────────────────────────────────────────────

def load_full_system(
    model_path: str,
    checkpoint_path: str,
    anamnesis_url: str = "http://localhost:6060",
):
    """
    Load the complete Phase 6 stack with native MXFP4 quantization.

    Memory notes:
    - Newer transformers versions dequantize MXFP4 → BF16 on GPU during load,
      allocating ~2GB temporary buffers per shard. Use higher overhead than
      training (7GB on 24GB, 4GB on 16GB) to give the dequantizer headroom.
    - If you hit OOM here, run with the project venv instead of system Python:
        .venv\\Scripts\\python.exe -m lmf.tests.test_phase7b_experiential_eval
      The venv transformers version has lighter MXFP4 loading behavior.
    - KV cache in generate_text() handles VRAM at long generation lengths.
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    t0 = time.time()
    max_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            # Higher overhead than training: MXFP4 dequant at load needs ~2GB temp
            overhead_gb = 4.0 if total_gb < 20.0 else 7.0
            alloc_gb = max(1, int(total_gb - overhead_gb))
            max_mem[i] = f"{alloc_gb}GiB"
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_gb:.0f}GB) "
                  f"— allocating {alloc_gb}GiB ({overhead_gb}GB overhead)")
    max_mem["cpu"] = "80GiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        offload_folder="offload_temp",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    config = gpt_oss_20b_config()
    config.device = "cpu"
    lmf = LivingMemoryField(config)
    harness = BridgeHarness(model=model, lmf=lmf, bridge_device="cpu")

    router_bias = KiroRouterBias(
        num_experts=model.config.num_local_experts,
        num_layers=model.config.num_hidden_layers,
    )
    hook_manager = RouterHookManager(model, router_bias, verbose=False)
    injector = NeuralAnamnInjector(field_dim=lmf.field_dim)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        harness.input_bridge.load_state_dict(ckpt['input_bridge'])
        harness.output_bridge.load_state_dict(ckpt['output_bridge'])
        harness.memory_bridge.load_state_dict(ckpt['memory_bridge'])
        if 'lmf' in ckpt:
            harness.lmf.load_state_dict(ckpt['lmf'])
        if 'injector' in ckpt:
            injector.load_state_dict(ckpt['injector'])
        if 'router_bias' in ckpt:
            router_bias.load_state_dict(ckpt['router_bias'])
            scales = router_bias.get_layer_scales()
            print(f"  RouterBias loaded (scale_max={max(scales):.4f})")
        print(f"  Phase {ckpt.get('phase','?')}, step {ckpt.get('step','?')}, "
              f"improvement={ckpt.get('running_improvement', 0):+.4f}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")

    print(f"Connecting to Neural Anamnesis at {anamnesis_url}...")
    anamnesis = SyncNeuralAnamnClient(base_url=anamnesis_url)
    if anamnesis.is_available():
        print("  Neural Anamnesis connected")
    else:
        print("  Neural Anamnesis unavailable")

    return {
        'model': model, 'tokenizer': tokenizer, 'lmf': lmf,
        'harness': harness, 'injector': injector,
        'router_bias': router_bias, 'hook_manager': hook_manager,
        'anamnesis': anamnesis,
        'vocab_size': model.config.vocab_size,
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
    }


# ── Generation Engine ─────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    condition: str
    test_name: str
    response: str
    new_tokens: int
    elapsed_s: float
    tokens_per_sec: float
    loss_on_target: float
    field_norm: float
    working_memories: int
    gate_value: float
    router_scale_max: float
    memory_feature_hits: List[str]
    general_feature_hits: List[str]
    memory_hit_rate: float
    general_hit_rate: float
    emotional_state: Dict[str, float]


def feed_context_to_field(sys: dict, context_text: str) -> Tuple[float, int]:
    reset_field(sys['lmf'])
    ctx_ids, ctx_mask = tokenize_chat(
        sys['tokenizer'], sys['model'], context_text, skip_to_final=False
    )
    with torch.no_grad():
        sys['harness'].step(ctx_ids, ctx_mask)
    norm = sys['lmf'].field_state.norm().item()
    working = int(sys['lmf'].working.active_mask.sum().item())
    return norm, working


def compute_target_loss(
    sys: dict, target_text: str, field_snapshot: Optional[torch.Tensor],
    use_bridge3: bool, use_injector: bool, retrieved: Optional[torch.Tensor]
) -> Tuple[float, float]:
    model = sys['model']
    tokenizer = sys['tokenizer']
    harness = sys['harness']
    injector = sys['injector']
    vocab_size = sys['vocab_size']
    pad_token_id = sys['pad_token_id']

    target_ids, target_mask = tokenize_chat(tokenizer, model, target_text)
    shift_labels = target_ids[:, 1:].to("cpu").contiguous()

    with torch.no_grad():
        outputs = model(input_ids=target_ids, attention_mask=target_mask)
        base_logits = outputs.logits.to("cpu", dtype=torch.float32)

    gate_value = 0.0
    if use_bridge3 and field_snapshot is not None:
        if use_injector and retrieved is not None:
            blended, gate_t = injector(field_snapshot, retrieved)
            gate_value = gate_t.item() if isinstance(gate_t, torch.Tensor) else float(gate_t)
            field_for_bridge = blended.to(harness.bridge_device, dtype=torch.float32)
        else:
            field_for_bridge = field_snapshot.to(harness.bridge_device, dtype=torch.float32)
        with torch.no_grad():
            bridge3_out = harness.output_bridge(
                field_state=field_for_bridge, model_logits=base_logits,
            )
            logits = bridge3_out['combined_logits']
    else:
        logits = base_logits

    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
    ).item()

    return loss, gate_value


def apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: zero out tokens outside the top-p cumulative mass."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right so the first token above threshold is kept
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return logits


def generate_text(
    sys: dict, prompt: str, field_snapshot: Optional[torch.Tensor],
    use_bridge3: bool, max_new_tokens: int = 500,
    temperature: float = 1.0, top_p: float = 1.0,
) -> Tuple[str, int, float]:
    """
    KV cache generation — O(n) VRAM per step instead of O(n²).
    Prefills prompt once, then each decode step processes only the new token.
    Supports temperature scaling and top-p (nucleus) sampling.
    """
    model = sys['model']
    tokenizer = sys['tokenizer']
    harness = sys['harness']

    input_ids, attention_mask = tokenize_chat(tokenizer, model, prompt)
    t0 = time.time()

    with torch.no_grad():
        prefill_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = prefill_out.past_key_values
        base_logits = prefill_out.logits[0, -1].float().cpu()

        if use_bridge3 and field_snapshot is not None:
            bridge3_out = harness.output_bridge(field_state=field_snapshot)
            logit_bias = bridge3_out['logit_bias'][0].cpu()
            next_logits = base_logits + logit_bias
        else:
            next_logits = base_logits

    next_logits[200000:] = float('-inf')
    # Apply temperature + top-p sampling (or greedy if temperature <= 0)
    if temperature > 0 and temperature != 1.0:
        scaled = next_logits / temperature
    else:
        scaled = next_logits
    if temperature > 0 and top_p < 1.0:
        scaled = apply_top_p_filtering(scaled.clone(), top_p)
    if temperature > 0:
        probs = torch.softmax(scaled, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())
    else:
        next_token = int(torch.argmax(next_logits).item())
    generated_ids = [next_token]

    for step in range(1, max_new_tokens):
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break

        next_t = torch.tensor(
            [[next_token]], device=input_ids.device, dtype=input_ids.dtype
        )
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1,
        )

        with torch.no_grad():
            step_out = model(
                input_ids=next_t,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = step_out.past_key_values
            base_logits = step_out.logits[0, -1].float().cpu()

            if use_bridge3 and field_snapshot is not None:
                bridge3_out = harness.output_bridge(field_state=field_snapshot)
                logit_bias = bridge3_out['logit_bias'][0].cpu()
                next_logits = base_logits + logit_bias
            else:
                next_logits = base_logits

        next_logits[200000:] = float('-inf')
        if temperature > 0 and temperature != 1.0:
            scaled = next_logits / temperature
        else:
            scaled = next_logits
        if temperature > 0 and top_p < 1.0:
            scaled = apply_top_p_filtering(scaled.clone(), top_p)
        if temperature > 0:
            probs = torch.softmax(scaled, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_token = int(torch.argmax(next_logits).item())
        generated_ids.append(next_token)

        if step > int(max_new_tokens * 0.9):
            response_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if "(Note:" in response_so_far:
                break
            if response_so_far.endswith(('.', '!', '?', '"')) and "\n\n" in response_so_far[-50:]:
                break

        if step % 200 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    new_tokens = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    del past_key_values
    torch.cuda.empty_cache()

    for stop in ["(Note:", "}"]:
        if stop in text:
            text = text.split(stop)[0].strip()

    return text, new_tokens, elapsed


# ── Core Eval Runner ──────────────────────────────────────────────────────────

def run_single_eval(
    sys: dict,
    test_case: dict,
    emotional_state: Dict[str, float],
    condition: str,
    use_field: bool,
    use_anamnesis: bool,
    use_router: bool,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> GenerationResult:
    hook_manager = sys['hook_manager']
    router_bias = sys['router_bias']
    anamnesis = sys['anamnesis']
    injector = sys['injector']
    lmf = sys['lmf']

    if use_router:
        emotional_t, value_t = build_state_tensors(emotional_state)
        router_bias.arm_from_tensors(emotional_t, value_t)
        if not hook_manager.is_active():
            hook_manager.reregister(
                sys['model'].model if hasattr(sys['model'], 'model') else sys['model']
            )
    else:
        hook_manager.restore(verbose=False)
        router_bias.disarm()

    field_snapshot = None
    field_norm = 0.0
    working_count = 0

    if use_field:
        field_norm, working_count = feed_context_to_field(sys, test_case["context"])
        field_snapshot = lmf.field_state.clone().cpu()
        field_snapshot = torch.nan_to_num(field_snapshot, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        reset_field(lmf)

    retrieved = None
    if use_anamnesis and field_snapshot is not None and anamnesis.is_available():
        retrieved = anamnesis.query(field_snapshot, top_k=5)

    loss, gate_value = compute_target_loss(
        sys, test_case["prompt"], field_snapshot,
        use_bridge3=use_field,
        use_injector=use_anamnesis and retrieved is not None,
        retrieved=retrieved,
    )

    if use_field:
        field_norm, working_count = feed_context_to_field(sys, test_case["context"])
        field_snapshot = lmf.field_state.clone().cpu()
        field_snapshot = torch.nan_to_num(field_snapshot, nan=0.0, posinf=1.0, neginf=-1.0)
        if use_anamnesis and anamnesis.is_available():
            retrieved = anamnesis.query(field_snapshot, top_k=5)
            if retrieved is not None:
                blended, _ = injector(field_snapshot, retrieved)
                field_snapshot = blended.detach()

    response, new_tokens, elapsed = generate_text(
        sys, test_case["prompt"], field_snapshot,
        use_bridge3=use_field, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
    )

    tps = new_tokens / max(elapsed, 1e-9)
    mem_hits = count_feature_hits(response, test_case.get("memory_features", []))
    gen_hits = count_feature_hits(response, test_case.get("general_features", []))
    mem_total = len(test_case.get("memory_features", []))
    gen_total = len(test_case.get("general_features", []))
    router_scale = max(router_bias.get_layer_scales()) if use_router else 0.0

    if use_router:
        hook_manager.restore(verbose=False)

    torch.cuda.empty_cache()

    return GenerationResult(
        condition=condition,
        test_name=test_case["name"],
        response=response,
        new_tokens=new_tokens,
        elapsed_s=elapsed,
        tokens_per_sec=tps,
        loss_on_target=loss,
        field_norm=field_norm,
        working_memories=working_count,
        gate_value=gate_value,
        router_scale_max=router_scale,
        memory_feature_hits=mem_hits,
        general_feature_hits=gen_hits,
        memory_hit_rate=len(mem_hits) / max(mem_total, 1),
        general_hit_rate=len(gen_hits) / max(gen_total, 1),
        emotional_state=emotional_state,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_result(r: GenerationResult):
    print(f"\n  [{r.condition}]")
    print(f"  Loss: {r.loss_on_target:.4f} | Field: {r.field_norm:.3f} | "
          f"Working: {r.working_memories} | Gate: {r.gate_value:.4f} | "
          f"RouterScale: {r.router_scale_max:.4f}")
    print(f"  Tokens: {r.new_tokens} | Speed: {r.tokens_per_sec:.1f} tok/s | "
          f"Time: {r.elapsed_s:.1f}s")
    mem_total = int(len(r.memory_feature_hits) / max(r.memory_hit_rate, 0.001)) if r.memory_hit_rate > 0 else 0
    print(f"  Memory features: {len(r.memory_feature_hits)}/{mem_total} "
          f"({r.memory_hit_rate:.0%}) — {r.memory_feature_hits[:5]}")
    print(f"  General features: {len(r.general_feature_hits)} ({r.general_hit_rate:.0%})")
    display = r.response[:500] + "..." if len(r.response) > 500 else r.response
    print(f"  Response: {display}")


def print_comparison_table(results: List[GenerationResult]):
    print(f"\n  {'Condition':<16} | {'Loss':>7} | {'MemHit%':>7} | {'GenHit%':>7} | "
          f"{'Gate':>6} | {'Router':>6} | {'Tok/s':>6} | {'Tokens':>6}")
    print(f"  {'-'*16} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
    for r in results:
        print(f"  {r.condition:<16} | {r.loss_on_target:>7.4f} | "
              f"{r.memory_hit_rate:>6.0%} | {r.general_hit_rate:>6.0%} | "
              f"{r.gate_value:>6.4f} | {r.router_scale_max:>6.4f} | "
              f"{r.tokens_per_sec:>6.1f} | {r.new_tokens:>6}")
    base = next((r for r in results if r.condition == "A_base"), None)
    if base:
        print(f"\n  Deltas from base (A):")
        for r in results:
            if r.condition == "A_base":
                continue
            print(f"    {r.condition:<16}: loss {base.loss_on_target - r.loss_on_target:>+.4f} | "
                  f"mem_hits {r.memory_hit_rate - base.memory_hit_rate:>+.0%} | "
                  f"gen_hits {r.general_hit_rate - base.general_hit_rate:>+.0%}")


# ── Main Orchestrators ────────────────────────────────────────────────────────

def run_four_condition_eval(sys: dict, test_cases: List[dict], results_dir: str,
                            max_new_tokens: int = 500, temperature: float = 1.0,
                            top_p: float = 1.0):
    all_results = []
    for tc in test_cases:
        print(f"\n{'='*80}\nTEST: {tc['name']}\n{'='*80}")
        emotional = tc.get("emotional_profile", {})
        conditions = [
            ("A_base",       False, False, False),
            ("B_lmf",        True,  False, False),
            ("C_lmf+anam",   True,  True,  False),
            ("D_full_stack",  True,  True,  True),
        ]
        case_results = []
        for cond_name, use_field, use_anam, use_router in conditions:
            try:
                r = run_single_eval(
                    sys, tc, emotional, condition=cond_name,
                    use_field=use_field, use_anamnesis=use_anam,
                    use_router=use_router, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p,
                )
                print_result(r)
                case_results.append(r)
                all_results.append(r)
            except Exception as e:
                print(f"\n  [{cond_name}] ERROR: {e}")
                import traceback; traceback.print_exc()
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
        if case_results:
            print_comparison_table(case_results)
    return all_results


def run_emotional_modulation_eval(sys: dict, mod_cases: List[dict], results_dir: str,
                                  max_new_tokens: int = 500, temperature: float = 1.0,
                                  top_p: float = 1.0):
    all_results = []
    for mc in mod_cases:
        print(f"\n{'='*80}\nEMOTIONAL MODULATION TEST: {mc['name']}\n{'='*80}")
        case_results = []
        for state_name, state_dict in mc["states"].items():
            cond_name = f"D_{state_name}"
            try:
                tc = {
                    "name": mc["name"], "context": mc["context"],
                    "prompt": mc["prompt"], "memory_features": [], "general_features": [],
                }
                r = run_single_eval(
                    sys, tc, state_dict, condition=cond_name,
                    use_field=True, use_anamnesis=True, use_router=True,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p,
                )
                print_result(r)
                case_results.append(r)
                all_results.append(r)
            except Exception as e:
                print(f"\n  [{cond_name}] ERROR: {e}")
                import traceback; traceback.print_exc()
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
        if len(case_results) >= 2:
            print(f"\n  === Emotional Modulation Analysis ===")
            loss_vals = [r.loss_on_target for r in case_results]
            print(f"  Loss range: {min(loss_vals):.4f} — {max(loss_vals):.4f} "
                  f"(spread: {max(loss_vals)-min(loss_vals):.4f})")
            unique = set(r.response for r in case_results)
            print(f"  Unique responses: {len(unique)}/{len(case_results)}")
    return all_results


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ANIMA Phase 7: Integrated Evaluation Suite")
    p.add_argument("--model_path",    type=str, default=r"D:\gpt-oss-20b")
    p.add_argument("--checkpoint",    type=str, default=None)
    p.add_argument("--anamnesis_url", type=str, default="http://localhost:6060")
    p.add_argument("--results_dir",   type=str, default=None)
    p.add_argument("--skip_modulation", action="store_true")
    p.add_argument("--max_tokens",    type=int, default=500)
    p.add_argument("--temperature",   type=float, default=1.0,
                   help="Sampling temperature (0=greedy, 1.0=default, <1=sharper, >1=flatter)")
    p.add_argument("--top_p",         type=float, default=1.0,
                   help="Top-p nucleus sampling threshold (1.0=disabled, 0.9=typical)")
    return p.parse_args()


def find_latest_checkpoint():
    ckpt_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints'
    )
    final = os.path.join(ckpt_dir, "phase6_final.pt")
    if os.path.exists(final):
        return final
    candidates = [f for f in os.listdir(ckpt_dir)
                  if f.startswith("phase6_step_") and f.endswith(".pt")]
    if candidates:
        steps = []
        for c in candidates:
            try:
                step = int(c.replace("phase6_step_", "").replace(".pt", ""))
                steps.append((step, os.path.join(ckpt_dir, c)))
            except ValueError:
                pass
        if steps:
            return sorted(steps, reverse=True)[0][1]
    return None


def main():
    args = parse_args()
    print("=" * 80)
    print("ANIMA Phase 7: Integrated Evaluation Suite")
    print("=" * 80)

    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if not ckpt_path:
        print("\nERROR: No checkpoint found.")
        return

    results_dir = args.results_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'phase7_eval'
    )
    os.makedirs(results_dir, exist_ok=True)

    sys_components = load_full_system(args.model_path, ckpt_path, args.anamnesis_url)

    if args.temperature != 1.0 or args.top_p != 1.0:
        print(f"\n  Sampling: temperature={args.temperature}, top_p={args.top_p}")
    else:
        print(f"\n  Sampling: greedy (temperature=1.0, no top-p)")

    print(f"\n{'='*80}\nPART 1: Four-Condition Comparison\n{'='*80}")
    four_cond_results = run_four_condition_eval(
        sys_components, EVAL_CASES, results_dir, max_new_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )

    mod_results = []
    if not args.skip_modulation:
        print(f"\n{'='*80}\nPART 2: Emotional Modulation Test\n{'='*80}")
        mod_results = run_emotional_modulation_eval(
            sys_components, EMOTIONAL_MODULATION_CASES, results_dir,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_p=args.top_p,
        )

    all_results = four_cond_results + mod_results
    results_path = os.path.join(results_dir, "phase7_results.json")
    with open(results_path, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    sys_components['anamnesis'].close()
    sys_components['hook_manager'].restore(verbose=True)
    print(f"\n{'='*80}\nPHASE 7 EVALUATION COMPLETE\n{'='*80}")


if __name__ == "__main__":
    main()
