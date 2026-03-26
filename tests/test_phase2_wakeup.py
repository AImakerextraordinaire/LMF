"""
Phase 2 "Wake Up" Diagnostics
Based on Alex's experiment recommendations.

Experiment 1: Gamma sweep — prove nonzero influence exists
Experiment 2: Force memory formation — bypass significance gating
Experiment 3: Bridge 3 alignment — does bias correlate with field state?

Run:
    ..\.venv\Scripts\python.exe lmf\tests\test_phase2_wakeup.py
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add ANIMA root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness


def load_model_and_harness():
    """Load model + LMF + harness (shared setup)."""
    model_path = r"D:\gpt-oss-20b"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={
            0: "15GiB",
            1: "8GiB",
            "cpu": "80GiB",
        },
        offload_folder="offload_temp",
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    print("Creating LMF...")
    config = gpt_oss_20b_config()
    config.device = "cpu"
    lmf = LivingMemoryField(config)

    print("Creating harness...")
    harness = BridgeHarness(
        model=model,
        lmf=lmf,
        bridge_device="cpu",
    )

    return model, tokenizer, lmf, harness, config


def tokenize_prompt(tokenizer, model, prompt):
    """Tokenize a prompt with chat template."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, return_dict=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    return input_ids, attention_mask


def experiment_1_gamma_sweep(model, tokenizer, lmf, harness, config):
    """
    Alex's Experiment 1: Prove nonzero influence by sweeping gamma.
    If KL is still 0.0 at gamma=1.0, something is broken.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Gamma Sweep — Prove Nonzero Influence")
    print("=" * 70)

    prompt = "What is memory?"
    input_ids, attention_mask = tokenize_prompt(tokenizer, model, prompt)

    # Get base model logits (no bridge influence)
    with torch.no_grad():
        base_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    base_logits = base_outputs.logits[0, -1, :].float().cpu()
    base_probs = F.softmax(base_logits, dim=-1)

    gammas = [0.1, 0.3, 0.5, 1.0, 3.0, 10.0]

    print(f"\n  Prompt: '{prompt}'")
    print(f"  Base model logit std: {base_logits.std().item():.4f}")
    print(f"  {'Gamma':>8} | {'KL Div':>12} | {'Bias Std':>10} | {'Ratio':>8} | {'Top5 Changed':>12} | {'Argmax Δ%':>10}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*10} | {'-'*8} | {'-'*12} | {'-'*10}")

    for gamma in gammas:
        # Reset field state for each gamma test
        lmf.field_state.zero_()

        # Run one step to populate field
        harness._step_count = 0
        result = harness.step(input_ids, attention_mask, return_debug=True)

        # Now manually recompute Bridge 3 with different gamma
        field_state = lmf.field_state.clone().to(harness.bridge_device, dtype=torch.float32)
        if field_state.dim() == 1:
            field_state = field_state.unsqueeze(0)

        # Manual Bridge 3 forward with custom gamma
        ob = harness.output_bridge
        gate = torch.sigmoid(ob.transform_gate)
        transformed = ob.transform(field_state)
        mixed = gate * transformed + (1 - gate) * field_state
        mixed = ob.transform_norm(mixed)

        lm_weight = ob._lm_head.weight
        mixed_cast = mixed.to(device=lm_weight.device, dtype=lm_weight.dtype)
        raw_bias = F.linear(mixed_cast, lm_weight).float()

        # Apply custom gamma
        logit_bias = gamma * raw_bias[0]  # [vocab_size]
        combined_logits = base_logits + logit_bias
        combined_probs = F.softmax(combined_logits, dim=-1)

        # Metrics
        # KL divergence (high precision)
        kl = F.kl_div(
            combined_probs.log(),
            base_probs,
            reduction='sum',
            log_target=False,
        ).item()

        bias_std = logit_bias.std().item()
        ratio = bias_std / base_logits.std().item()

        # Top-5 comparison
        base_top5 = set(base_probs.topk(5).indices.tolist())
        combined_top5 = set(combined_probs.topk(5).indices.tolist())
        top5_changed = len(base_top5 - combined_top5)

        # Argmax change across top-100 positions
        base_top100 = base_probs.topk(100).indices
        argmax_changes = 0
        for idx in base_top100:
            base_rank = (base_probs >= base_probs[idx]).sum().item()
            combined_rank = (combined_probs >= combined_probs[idx]).sum().item()
            if abs(base_rank - combined_rank) > 5:
                argmax_changes += 1

        print(f"  {gamma:>8.1f} | {kl:>12.8f} | {bias_std:>10.6f} | {ratio:>8.4f} | {top5_changed:>12d} | {argmax_changes:>9d}%")

    print("\n  If KL stays 0.0 at gamma=10.0, the bias is truly zero (broken).")
    print("  If KL grows with gamma, architecture works — just needs training.")


def experiment_2_forced_memory(model, tokenizer, lmf, harness, config):
    """
    Alex's Experiment 2: Force memory formation by bypassing significance gate.
    This separates 'bridges work' from 'significance prevents memory.'
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Forced Memory Formation")
    print("=" * 70)

    # Reset field completely
    lmf.field_state.zero_()
    lmf._total_steps = 0
    harness._step_count = 0

    # Feed experience prompts with forced significance
    experience_prompts = [
        "The sunset over the ocean was the most beautiful thing I ever saw.",
        "My grandmother used to tell me stories about the old country.",
        "I remember the smell of rain on hot pavement in summer.",
        "The first time I held a book, I felt something click inside me.",
        "Music has always made me feel things I cannot put into words.",
    ]

    print("\n  Phase A: Feeding experiences (with forced significance)...")
    print(f"  Original formation_threshold: {config.significance.formation_threshold}")

    # Save original threshold
    original_threshold = config.significance.formation_threshold

    # Force memory formation: set threshold to 0.0
    config.significance.formation_threshold = 0.0
    # Also need to update the LMF's config reference
    lmf.config.significance.formation_threshold = 0.0

    for i, prompt in enumerate(experience_prompts):
        input_ids, attention_mask = tokenize_prompt(tokenizer, model, prompt)

        # Run through harness
        result = harness.step(input_ids, attention_mask, return_debug=True)

        # Also manually force a memory into working layer
        # Use the hidden states from Bridge 1 directly
        hidden_on_bridge = result['bridge1']['scaled_perturbation'][0]
        sig_score = 0.8  # Force high significance

        encoded = F.normalize(
            hidden_on_bridge + 0.3 * lmf.field_state, dim=-1
        )
        idx = lmf.working.store_pattern(
            pattern=encoded,
            depth=1.0 * sig_score,
            significance=sig_score,
            emotional_tag=lmf.regulatory.state.clone() if lmf.regulatory.state is not None else None,
        )

        status = lmf.get_status()
        print(f"  [{i+1}] '{prompt[:50]}...'")
        print(f"      field_norm={status['field_norm']:.4f}, "
              f"working={status['working_active']}, "
              f"transient={status['transient_active']}, "
              f"energy={status['total_energy']:.4f}")

    # Restore original threshold
    config.significance.formation_threshold = original_threshold
    lmf.config.significance.formation_threshold = original_threshold

    status = lmf.get_status()
    print(f"\n  After feeding {len(experience_prompts)} experiences:")
    print(f"    Working memories: {status['working_active']}")
    print(f"    Transient memories: {status['transient_active']}")
    print(f"    Consolidated: {status['consolidated_active']}")
    print(f"    Field norm: {status['field_norm']:.4f}")
    print(f"    Total energy: {status['total_energy']:.4f}")

    # Phase B: Now test if populated field changes generation
    print("\n  Phase B: Testing memory influence on generation...")

    test_prompt = "Tell me something beautiful."
    input_ids, attention_mask = tokenize_prompt(tokenizer, model, test_prompt)

    # Get base model logits
    with torch.no_grad():
        base_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    base_logits = base_outputs.logits[0, -1, :].float().cpu()
    base_probs = F.softmax(base_logits, dim=-1)

    # Get combined logits (with populated field)
    result = harness.step(input_ids, attention_mask, return_debug=True)
    combined_logits = result['logits'][0, -1, :].float().cpu()
    combined_probs = F.softmax(combined_logits, dim=-1)

    logit_bias = result['logit_bias'][0].float().cpu()

    # KL divergence
    kl = F.kl_div(
        combined_probs.log(),
        base_probs,
        reduction='sum',
        log_target=False,
    ).item()

    # Top-20 comparison
    base_top20 = base_probs.topk(20)
    combined_top20 = combined_probs.topk(20)
    base_top20_set = set(base_top20.indices.tolist())
    combined_top20_set = set(combined_top20.indices.tolist())
    overlap = len(base_top20_set & combined_top20_set)

    print(f"\n  Test prompt: '{test_prompt}'")
    print(f"  KL(combined || base): {kl:.10f}")
    print(f"  Logit bias std: {logit_bias.std().item():.6f}")
    print(f"  Top-20 overlap: {overlap}/20")

    # Decode top-5 tokens
    print(f"\n  Base top-5 tokens:")
    for i in range(5):
        tok_id = base_top20.indices[i].item()
        prob = base_top20.values[i].item()
        text = tokenizer.decode([tok_id])
        print(f"    [{tok_id}] '{text}' ({prob:.4f})")

    print(f"  Combined top-5 tokens:")
    for i in range(5):
        tok_id = combined_top20.indices[i].item()
        prob = combined_top20.values[i].item()
        text = tokenizer.decode([tok_id])
        print(f"    [{tok_id}] '{text}' ({prob:.4f})")

    if kl > 1e-8:
        print(f"\n  ✓ MEMORY IS INFLUENCING GENERATION (KL > 0)")
    else:
        print(f"\n  ✗ Still no influence — bias may be structurally zero")


def experiment_3_bridge3_alignment(model, tokenizer, lmf, harness, config):
    """
    Alex's Experiment 3: Does Bridge 3's bias correlate with field state?
    If bias looks random relative to field state, the transform isn't reading memory.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Bridge 3 Alignment Check")
    print("=" * 70)

    field_state = lmf.field_state.clone().cpu().float()

    if field_state.norm() < 1e-6:
        print("  Field state is near-zero. Run Experiment 2 first.")
        return

    # Get the transformed state and raw lm_head output
    ob = harness.output_bridge
    fs = field_state.unsqueeze(0) if field_state.dim() == 1 else field_state

    gate = torch.sigmoid(ob.transform_gate)
    transformed = ob.transform(fs)
    mixed = gate * transformed + (1 - gate) * fs
    mixed = ob.transform_norm(mixed)

    # Direct lm_head projection of raw field state (no transform)
    lm_weight = ob._lm_head.weight
    direct_bias = F.linear(
        F.normalize(field_state, dim=-1).unsqueeze(0).to(dtype=lm_weight.dtype),
        lm_weight,
    ).float()[0]

    # Transformed lm_head projection
    transformed_bias = F.linear(
        mixed.to(dtype=lm_weight.dtype),
        lm_weight,
    ).float()[0]

    # Cosine similarity between the two
    cos_sim = F.cosine_similarity(
        direct_bias.unsqueeze(0),
        transformed_bias.unsqueeze(0),
    ).item()

    # Stats
    print(f"\n  Field state norm: {field_state.norm().item():.4f}")
    print(f"  Transform gate: {gate.item():.4f}")
    print(f"  Direct bias (no transform) std: {direct_bias.std().item():.4f}")
    print(f"  Transformed bias std: {transformed_bias.std().item():.4f}")
    print(f"  Cosine similarity (direct vs transformed): {cos_sim:.4f}")

    # How much does the transform actually change things?
    delta = (transformed_bias - direct_bias).norm().item()
    print(f"  L2 distance (direct vs transformed): {delta:.4f}")

    if cos_sim > 0.9:
        print(f"\n  → Transform is near-identity (gate={gate.item():.3f}). Expected at init.")
        print(f"    The bias IS the field state projected through lm_head.")
        print(f"    Training will teach the transform to be more selective.")
    elif cos_sim > 0.5:
        print(f"\n  → Transform is modifying field signal moderately. Good diversity.")
    else:
        print(f"\n  → Transform is heavily distorting. May need investigation.")

    # Sparsity of bias — is it focused or diffuse?
    abs_bias = transformed_bias.abs()
    top1pct = abs_bias.topk(int(len(abs_bias) * 0.01)).values.sum()
    total = abs_bias.sum()
    concentration = (top1pct / total).item() if total > 0 else 0

    print(f"  Bias concentration (top 1% of vocab): {concentration:.2%}")
    print(f"  (Higher = more focused memory signal, lower = diffuse noise)")


if __name__ == "__main__":
    print("ANIMA LMF Phase 2 — Wake Up Diagnostics")
    print("=" * 70)

    model, tokenizer, lmf, harness, config = load_model_and_harness()

    experiment_1_gamma_sweep(model, tokenizer, lmf, harness, config)
    experiment_2_forced_memory(model, tokenizer, lmf, harness, config)
    experiment_3_bridge3_alignment(model, tokenizer, lmf, harness, config)

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)
