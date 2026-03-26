"""
Significance Calibration — Diagnostic + Fix

Diagnoses both significance gates:
  1. Bridge 1's sig_head (too low for real activations)
  2. LMF's SignificanceDetector (unreachable threshold with inactive components)

Then applies fixes and validates memory formation.

Run:
    ..\.venv\Scripts\python.exe lmf\tests\test_significance_calibration.py
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
from lmf.core.significance import SignificanceDetector


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


def diagnose_gate1(model, tokenizer, harness):
    """Diagnose Bridge 1's sig_head behavior on real activations."""
    print("\n" + "=" * 70)
    print("GATE 1 DIAGNOSIS: Bridge 1 sig_head")
    print("=" * 70)

    prompts = [
        "Hello.",
        "The sunset over the ocean was beautiful.",
        "Quantum entanglement challenges our understanding of locality.",
        "I remember the smell of rain on hot pavement.",
        "What is the meaning of life?",
    ]

    ib = harness.input_bridge
    print(f"\n  sig_head architecture: {ib.sig_head}")
    print(f"  alpha: {ib.alpha}")

    for prompt in prompts:
        ids, mask = tokenize(tokenizer, model, prompt)

        # Get hidden states
        with torch.no_grad():
            outputs = model(ids, attention_mask=mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

        # Move to bridge device
        hidden_b = last_hidden.to(harness.bridge_device, dtype=torch.float32)
        mask_b = mask.to(harness.bridge_device) if mask is not None else None

        # Attention pool
        pooled = ib.pool(hidden_b, mask_b)  # [B, 2880]

        # Sig head input stats
        pooled_mag = pooled.abs().mean().item()
        pooled_std = pooled.std().item()
        pooled_max = pooled.abs().max().item()

        # Sig head intermediate values
        with torch.no_grad():
            # Manual forward through sig head
            x = ib.sig_head[0](pooled)  # Linear → [B, 64]
            pre_gelu = x.clone()
            x = ib.sig_head[1](x)       # GELU
            x = ib.sig_head[2](x)       # Linear → [B, 1]
            pre_sigmoid = x.clone()
            sig = torch.sigmoid(x)      # Sigmoid → [0, 1]

        print(f"\n  Prompt: '{prompt[:55]}...'")
        print(f"    Pooled: mean_abs={pooled_mag:.2f}, std={pooled_std:.2f}, max_abs={pooled_max:.2f}")
        print(f"    After Linear1: mean={pre_gelu.mean().item():.4f}, std={pre_gelu.std().item():.4f}")
        print(f"    Pre-sigmoid: {pre_sigmoid.item():.4f}")
        print(f"    Significance output: {sig.item():.6f}")
        print(f"    Scaled perturbation magnitude: {ib.alpha * sig.item():.6f}")


def diagnose_gate2(lmf, harness, config):
    """Diagnose LMF SignificanceDetector component values."""
    print("\n" + "=" * 70)
    print("GATE 2 DIAGNOSIS: LMF SignificanceDetector")
    print("=" * 70)

    sd = lmf.significance

    # Show component weights
    raw_weights = sd.component_weights.data
    softmax_weights = F.softmax(raw_weights, dim=0)
    labels = ['novelty', 'emotion', 'surprise', 'value', 'goal']

    print(f"\n  Component weights (raw → softmax):")
    for i, label in enumerate(labels):
        print(f"    {label:>10}: {raw_weights[i].item():.3f} → {softmax_weights[i].item():.4f}")

    print(f"\n  Formation threshold: {config.significance.formation_threshold}")

    # Compute max possible significance with different component scenarios
    scenarios = [
        ("All active (ideal)", [1.0, 1.0, 1.0, 1.0, 1.0]),
        ("Novelty + surprise only (current reality)", [1.0, 0.0, 1.0, 0.0, 0.0]),
        ("Novelty=1, surprise=0.38 (typical)", [1.0, 0.0, 0.38, 0.0, 0.0]),
        ("Novelty=0.5 (some memories exist)", [0.5, 0.0, 0.38, 0.0, 0.0]),
    ]

    print(f"\n  Max significance under different scenarios:")
    for name, values in scenarios:
        vals = torch.tensor(values).clamp(0.0, 1.0)
        sig = (softmax_weights * vals).sum().item()
        status = "✓" if sig > config.significance.formation_threshold else "✗"
        print(f"    {status} {name}: {sig:.4f} (threshold={config.significance.formation_threshold})")

    # Now test with actual perturbation-scale inputs
    print(f"\n  Testing with realistic perturbation magnitudes:")
    for mag_label, magnitude in [("Bridge1 actual (~0.0003)", 0.0003), 
                                  ("10x boosted", 0.003), 
                                  ("Normalized (~1.0)", 1.0)]:
        fake_input = F.normalize(torch.randn(lmf.field_dim), dim=0) * magnitude
        sig_score, components = sd.evaluate(
            input_embedding=fake_input,
            field_state=lmf.field_state,
            regulatory_state=lmf.regulatory.state,
            memory_layers=[lmf.consolidated, lmf.working, lmf.transient],
        )
        print(f"\n    {mag_label}:")
        for k, v in components.items():
            print(f"      {k}: {v.item():.6f}")
        print(f"      TOTAL: {sig_score.item():.6f} {'✓ FORMS MEMORY' if sig_score.item() > config.significance.formation_threshold else '✗ below threshold'}")


def diagnose_full_pipeline(model, tokenizer, lmf, harness, config):
    """Trace significance through the full pipeline for one prompt."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE TRACE")
    print("=" * 70)

    prompt = "I remember the smell of rain on hot pavement."
    ids, mask = tokenize(tokenizer, model, prompt)

    # Reset
    lmf.field_state.zero_()
    harness._step_count = 0

    # Run full step with debug
    result = harness.step(ids, mask, return_debug=True)

    b1 = result['bridge1']
    sig_bridge1 = b1['significance'].item()
    pert = b1['scaled_perturbation'][0]
    pert_norm = pert.norm().item()

    print(f"\n  Prompt: '{prompt}'")
    print(f"\n  Bridge 1:")
    print(f"    sig_head output: {sig_bridge1:.6f}")
    print(f"    alpha: {harness.input_bridge.alpha}")
    print(f"    scaled_perturbation norm: {pert_norm:.6f}")
    print(f"    effective scale: alpha * sig = {harness.input_bridge.alpha * sig_bridge1:.6f}")

    # Now manually trace what process_input does with this perturbation
    sd = lmf.significance
    sig_score, components = sd.evaluate(
        input_embedding=pert.to(lmf.field_state.device),
        field_state=lmf.field_state,
        regulatory_state=lmf.regulatory.state,
        memory_layers=[lmf.consolidated, lmf.working, lmf.transient],
    )

    print(f"\n  LMF SignificanceDetector:")
    for k, v in components.items():
        print(f"    {k}: {v.item():.6f}")
    print(f"    TOTAL sig: {sig_score.item():.6f}")
    print(f"    threshold: {config.significance.formation_threshold}")
    print(f"    Memory would form: {'YES' if sig_score.item() > config.significance.formation_threshold else 'NO'}")

    # What would happen with a normalized perturbation?
    pert_normalized = F.normalize(pert, dim=0)
    sig_score_n, components_n = sd.evaluate(
        input_embedding=pert_normalized.to(lmf.field_state.device),
        field_state=lmf.field_state,
        regulatory_state=lmf.regulatory.state,
        memory_layers=[lmf.consolidated, lmf.working, lmf.transient],
    )

    print(f"\n  With NORMALIZED perturbation (magnitude=1.0):")
    for k, v in components_n.items():
        print(f"    {k}: {v.item():.6f}")
    print(f"    TOTAL sig: {sig_score_n.item():.6f}")
    print(f"    Memory would form: {'YES' if sig_score_n.item() > config.significance.formation_threshold else 'NO'}")


if __name__ == "__main__":
    print("ANIMA LMF — Significance Calibration Diagnostic")
    print("=" * 70)

    model, tokenizer, lmf, harness, config = load_system()

    diagnose_gate1(model, tokenizer, harness)
    diagnose_gate2(lmf, harness, config)
    diagnose_full_pipeline(model, tokenizer, lmf, harness, config)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE — Review above to plan calibration fixes")
    print("=" * 70)
