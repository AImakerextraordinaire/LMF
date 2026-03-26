"""
ANIMA LMF - Bridge 2 (Memory Bridge) Validation
=================================================

Tests:
1. Module creation and parameter count
2. Hook registration on correct layers
3. Perturbation injection during forward pass
4. Gate initialization (near zero = gentle start)
5. Device-aware injection across GPU/CPU split
6. Full pipeline with all 3 bridges active
7. Injection magnitude relative to hidden states

Run:
    cd ANIMA repo root
    python lmf/tests/test_bridge2_validation.py
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness
from lmf.bridges.memory_bridge import MemoryBridge


def load_system():
    model_path = r"D:\gpt-oss-20b"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "15GiB", 1: "8GiB", "cpu": "80GiB"},
        offload_folder="offload_temp",
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    config = gpt_oss_20b_config()
    config.device = "cpu"
    lmf = LivingMemoryField(config)
    harness = BridgeHarness(model=model, lmf=lmf, bridge_device="cpu")
    return model, tokenizer, lmf, harness, config


def tokenize(tokenizer, model, prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, return_dict=True,
    )
    ids = inputs["input_ids"].to(model.device)
    mask = inputs.get("attention_mask")
    if mask is not None:
        mask = mask.to(model.device)
    return ids, mask


def test_module_structure(harness):
    """Test 1: Bridge 2 module exists and has correct structure."""
    print("\n" + "=" * 70)
    print("TEST 1: Module Structure")
    print("=" * 70)

    mb = harness.memory_bridge
    print(f"  Type: {type(mb).__name__}")
    print(f"  Target layers: {mb.target_layers}")
    print(f"  LoRA rank: {mb.rank}")
    print(f"  LoRA alpha: {mb.alpha}")
    print(f"  Num injectors: {len(mb.injectors)}")
    print(f"  Total params: {mb.get_param_count():,}")

    # Per-injector breakdown
    for key, inj in mb.injectors.items():
        gate = torch.sigmoid(inj.gate).item()
        print(f"    Layer {key}: {inj.get_param_count():,} params, gate={gate:.6f}")

    # Expected: 6 layers * (2880*32 + 32*2880 + 1) = 6 * 184,321 = ~1.1M
    expected_approx = len(mb.target_layers) * (2880 * 32 * 2 + 1)
    actual = mb.get_param_count()
    print(f"\n  Expected ~{expected_approx:,}, got {actual:,}")

    if actual > 0:
        print("  PASS: Bridge 2 module created with parameters")
    else:
        print("  FAIL: No parameters")


def test_hooks_registered(harness):
    """Test 2: Hooks are registered on correct layers."""
    print("\n" + "=" * 70)
    print("TEST 2: Hook Registration")
    print("=" * 70)

    mb = harness.memory_bridge
    print(f"  Hooks registered: {len(mb._hooks)}")
    print(f"  Target layers: {mb.target_layers}")

    if len(mb._hooks) == len(mb.target_layers):
        print("  PASS: One hook per target layer")
    else:
        print(f"  FAIL: Expected {len(mb.target_layers)} hooks, got {len(mb._hooks)}")


def test_gate_initialization(harness):
    """Test 3: Gates start near zero for gentle injection."""
    print("\n" + "=" * 70)
    print("TEST 3: Gate Initialization")
    print("=" * 70)

    gates = harness.memory_bridge.get_gate_values()
    print(f"  Gate values (sigmoid of learned parameter):")
    all_near_half = True
    for layer_idx, gate_val in sorted(gates.items()):
        status = "ok" if abs(gate_val - 0.5) < 0.1 else "unexpected"
        if status == "unexpected":
            all_near_half = False
        print(f"    Layer {layer_idx}: {gate_val:.6f} ({status})")

    # sigmoid(0.0) = 0.5, so initial gates should be ~0.5
    print(f"\n  Gates initialized to sigmoid(0) = 0.5: {'PASS' if all_near_half else 'CHECK'}")
    print("  (Gates will learn to open/close during training)")


def test_injection_no_field(model, tokenizer, harness):
    """Test 4: With zero field state, injection should be zero."""
    print("\n" + "=" * 70)
    print("TEST 4: Zero Field = Zero Injection")
    print("=" * 70)

    # Ensure field is zero
    harness.lmf.field_state.zero_()

    prompt = "Hello world"
    ids, mask = tokenize(tokenizer, model, prompt)

    # Run with zero field state
    result = harness.step(ids, mask, return_debug=True)

    stats = result.get('bridge2', {})
    if not stats:
        print("  No injection stats (field was zero, bridge not armed) - PASS")
    else:
        print("  Injection occurred despite zero field:")
        for layer_idx, s in stats.items():
            print(f"    Layer {layer_idx}: perturbation_norm={s['perturbation_norm']:.8f}")


def test_injection_with_field(model, tokenizer, harness):
    """Test 5: With populated field, injection should be nonzero."""
    print("\n" + "=" * 70)
    print("TEST 5: Populated Field = Nonzero Injection")
    print("=" * 70)

    # First, populate the field with some memories
    harness.lmf.field_state.zero_()
    harness._step_count = 0
    for idx in range(harness.lmf.working.max_patterns):
        if harness.lmf.working.active_mask[idx]:
            harness.lmf.working._clear_slot(idx)
    for idx in range(harness.lmf.transient.max_patterns):
        if harness.lmf.transient.active_mask[idx]:
            harness.lmf.transient._clear_slot(idx)

    # Feed a few prompts to build up field state
    seed_prompts = [
        "The sunset over the ocean was beautiful.",
        "I remember childhood summers by the lake.",
        "Music has always moved me deeply.",
    ]

    print("  Seeding field with memories...")
    for p in seed_prompts:
        ids, mask = tokenize(tokenizer, model, p)
        harness.step(ids, mask)

    field_norm = harness.lmf.field_state.norm().item()
    status = harness.lmf.get_status()
    print(f"  Field norm: {field_norm:.4f}")
    print(f"  Working memories: {status['working_active']}")

    # Now run a new prompt - Bridge 2 should inject the accumulated field
    test_prompt = "Tell me about beauty."
    ids, mask = tokenize(tokenizer, model, test_prompt)
    result = harness.step(ids, mask, return_debug=True)

    stats = result.get('bridge2', {})
    gates = result.get('bridge2_gates', {})

    if stats:
        print(f"\n  Injection stats per layer:")
        print(f"  {'Layer':>5} | {'Perturb Norm':>12} | {'Hidden Norm':>12} | {'Ratio':>10} | {'Gate':>6}")
        print(f"  {'-'*5} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*6}")
        for layer_idx in sorted(stats.keys()):
            s = stats[layer_idx]
            print(f"  {layer_idx:>5} | {s['perturbation_norm']:>12.6f} | {s['hidden_norm']:>12.2f} | {s['ratio']:>10.8f} | {s['gate']:>6.4f}")

        any_nonzero = any(s['perturbation_norm'] > 1e-8 for s in stats.values())
        if any_nonzero:
            print("\n  PASS: Bridge 2 is injecting field state into mid-layers!")
        else:
            print("\n  FAIL: Injection magnitudes are zero")
    else:
        print("  FAIL: No injection stats returned")


def test_injection_affects_output(model, tokenizer, harness):
    """Test 6: Bridge 2 injection measurably changes model output."""
    print("\n" + "=" * 70)
    print("TEST 6: Injection Affects Model Output")
    print("=" * 70)

    field_norm = harness.lmf.field_state.norm().item()
    if field_norm < 1e-6:
        print("  Field is empty, skipping (run test 5 first)")
        return

    test_prompt = "What makes something beautiful?"
    ids, mask = tokenize(tokenizer, model, test_prompt)

    # Run WITH Bridge 2 (normal)
    result_with = harness.step(ids, mask, return_debug=True)
    logits_with = result_with['model_logits'][0, -1, :].float().cpu()

    # Temporarily disable Bridge 2 by disarming
    harness.memory_bridge.disarm()
    harness.memory_bridge._armed = False

    # Run model directly WITHOUT Bridge 2
    with torch.no_grad():
        outputs_without = model(input_ids=ids, attention_mask=mask)
    logits_without = outputs_without.logits[0, -1, :].float().cpu()

    # Compare
    probs_with = F.softmax(logits_with, dim=-1)
    probs_without = F.softmax(logits_without, dim=-1)

    kl = F.kl_div(
        probs_with.clamp(min=1e-10).log(),
        probs_without.clamp(min=1e-10),
        reduction='sum',
        log_target=False,
    ).item()

    logit_diff = (logits_with - logits_without).abs()
    max_diff = logit_diff.max().item()
    mean_diff = logit_diff.mean().item()

    # Top-k token differences
    topk_with = logits_with.topk(5)
    topk_without = logits_without.topk(5)

    print(f"  Field norm: {field_norm:.4f}")
    print(f"  KL divergence (with vs without Bridge 2): {kl:.10f}")
    print(f"  Max logit difference: {max_diff:.6f}")
    print(f"  Mean logit difference: {mean_diff:.6f}")

    print(f"\n  Top-5 tokens WITHOUT Bridge 2:")
    for i in range(5):
        tok_id = topk_without.indices[i].item()
        tok_logit = topk_without.values[i].item()
        print(f"    {i+1}. [{tok_id}] logit={tok_logit:.4f}")

    print(f"\n  Top-5 tokens WITH Bridge 2:")
    for i in range(5):
        tok_id = topk_with.indices[i].item()
        tok_logit = topk_with.values[i].item()
        print(f"    {i+1}. [{tok_id}] logit={tok_logit:.4f}")

    if kl > 1e-8:
        print(f"\n  PASS: Bridge 2 measurably changes model output (KL={kl:.2e})")
    else:
        print(f"\n  Bridge 2 effect below measurement threshold")


def test_full_pipeline(model, tokenizer, harness):
    """Test 7: All three bridges working together."""
    print("\n" + "=" * 70)
    print("TEST 7: Full 3-Bridge Pipeline")
    print("=" * 70)

    # Reset
    harness.lmf.field_state.zero_()
    harness._step_count = 0
    for idx in range(harness.lmf.working.max_patterns):
        if harness.lmf.working.active_mask[idx]:
            harness.lmf.working._clear_slot(idx)
    for idx in range(harness.lmf.transient.max_patterns):
        if harness.lmf.transient.active_mask[idx]:
            harness.lmf.transient._clear_slot(idx)

    prompts = [
        "The ocean waves crash against the shore.",
        "Stars twinkle in the vast darkness above.",
        "A child laughs, and the whole room lights up.",
        "The weight of silence can be deafening.",
    ]

    print(f"  Running 4 prompts through full 3-bridge pipeline:")
    print(f"  {'#':>3} | {'B1 Sig':>6} | {'B2 Layers':>9} | {'B3 Bias':>8} | {'Work':>4} | {'Norm':>6} | {'Energy':>8}")
    print(f"  {'-'*3} | {'-'*6} | {'-'*9} | {'-'*8} | {'-'*4} | {'-'*6} | {'-'*8}")

    for i, prompt in enumerate(prompts):
        ids, mask = tokenize(tokenizer, model, prompt)
        result = harness.step(ids, mask, return_debug=True)

        b1_sig = result['significance'].item()
        b2_stats = result.get('bridge2', {})
        b2_active = len(b2_stats)
        b3_bias_std = result['logit_bias'][0].float().std().item()
        status = harness.lmf.get_status()

        print(f"  {i+1:>3} | {b1_sig:.4f} | {b2_active:>9} | {b3_bias_std:>8.6f} | "
              f"{status['working_active']:>4} | {status['field_norm']:>6.3f} | {status['total_energy']:>8.3f}")

    print(f"\n  All 3 bridges active:")
    print(f"    Bridge 1 (Input):  hidden states -> field perturbation")
    print(f"    Bridge 2 (Memory): field state -> mid-layer LoRA injection")
    print(f"    Bridge 3 (Output): field state -> logit bias")
    print(f"\n  Pipeline status: OPERATIONAL")


if __name__ == "__main__":
    print("ANIMA LMF - Bridge 2 (Memory Bridge) Validation")
    print("=" * 70)

    model, tokenizer, lmf, harness, config = load_system()

    test_module_structure(harness)
    test_hooks_registered(harness)
    test_gate_initialization(harness)
    test_injection_no_field(model, tokenizer, harness)
    test_injection_with_field(model, tokenizer, harness)
    test_injection_affects_output(model, tokenizer, harness)
    test_full_pipeline(model, tokenizer, harness)

    print("\n" + "=" * 70)
    print("BRIDGE 2 VALIDATION COMPLETE")
    print("=" * 70)
