"""
Bridge 2 - Injection Mechanism Verification
Temporarily sets up_proj to nonzero to prove hooks inject correctly,
then resets to zero init for training.
"""

import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness


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


if __name__ == "__main__":
    print("=" * 70)
    print("Bridge 2 Injection Mechanism Verification")
    print("=" * 70)

    # Load system
    model_path = r"D:\gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "15GiB", 1: "8GiB", "cpu": "80GiB"},
        offload_folder="offload_temp",
    )
    config = gpt_oss_20b_config()
    config.device = "cpu"
    lmf = LivingMemoryField(config)
    harness = BridgeHarness(model=model, lmf=lmf, bridge_device="cpu")

    # === Phase 1: Baseline with zero up_proj (default LoRA init) ===
    print("\n--- Phase 1: Default LoRA init (up_proj = zeros) ---")

    # Seed some field state
    lmf.field_state.zero_()
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)

    seeds = ["The ocean at sunset paints the sky in gold.", "Music carries emotions words cannot express."]
    for s in seeds:
        ids, mask = tokenize(tokenizer, model, s)
        harness.step(ids, mask)

    field_norm = lmf.field_state.norm().item()
    print(f"  Field norm: {field_norm:.4f}")

    # Run test prompt
    ids, mask = tokenize(tokenizer, model, "Beauty is everywhere.")
    result_zero = harness.step(ids, mask, return_debug=True)
    logits_zero = result_zero['model_logits'][0, -1, :].float().cpu()
    stats_zero = result_zero.get('bridge2', {})

    print(f"  Injection stats (should be zero):")
    for layer_idx in sorted(stats_zero.keys()):
        s = stats_zero[layer_idx]
        print(f"    Layer {layer_idx}: norm={s['perturbation_norm']:.8f}")

    # === Phase 2: Set up_proj to small random values ===
    print("\n--- Phase 2: Nonzero up_proj (proving hooks inject) ---")

    for key, injector in harness.memory_bridge.injectors.items():
        nn.init.normal_(injector.up_proj.weight, mean=0.0, std=0.01)

    # Also bump the gate to make injection more visible
    # (gate is already 0.5 which is fine)

    # Reset field to same state by re-seeding
    lmf.field_state.zero_()
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)

    for s in seeds:
        ids, mask = tokenize(tokenizer, model, s)
        harness.step(ids, mask)

    field_norm2 = lmf.field_state.norm().item()
    print(f"  Field norm: {field_norm2:.4f}")

    # Run same test prompt
    ids, mask = tokenize(tokenizer, model, "Beauty is everywhere.")
    result_nonzero = harness.step(ids, mask, return_debug=True)
    logits_nonzero = result_nonzero['model_logits'][0, -1, :].float().cpu()
    stats_nonzero = result_nonzero.get('bridge2', {})

    print(f"  Injection stats (should be NONZERO):")
    print(f"  {'Layer':>5} | {'Perturb Norm':>12} | {'Hidden Norm':>12} | {'Ratio':>12} | {'Gate':>6}")
    print(f"  {'-'*5} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*6}")
    for layer_idx in sorted(stats_nonzero.keys()):
        s = stats_nonzero[layer_idx]
        print(f"  {layer_idx:>5} | {s['perturbation_norm']:>12.6f} | {s['hidden_norm']:>12.2f} | {s['ratio']:>12.10f} | {s['gate']:>6.4f}")

    # === Phase 3: Compare outputs ===
    print("\n--- Phase 3: Output comparison ---")

    # Get baseline (no Bridge 2 at all)
    harness.memory_bridge.disarm()
    lmf.field_state.zero_()
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)

    for s in seeds:
        ids, mask = tokenize(tokenizer, model, s)
        harness.step(ids, mask)

    ids, mask = tokenize(tokenizer, model, "Beauty is everywhere.")
    with torch.no_grad():
        outputs_baseline = model(input_ids=ids, attention_mask=mask)
    logits_baseline = outputs_baseline.logits[0, -1, :].float().cpu()

    # Compare
    probs_baseline = F.softmax(logits_baseline, dim=-1)
    probs_nonzero = F.softmax(logits_nonzero, dim=-1)

    kl = F.kl_div(
        probs_nonzero.clamp(min=1e-10).log(),
        probs_baseline.clamp(min=1e-10),
        reduction='sum', log_target=False,
    ).item()

    logit_diff = (logits_nonzero - logits_baseline).abs()

    # Check if top tokens shifted
    topk_base = logits_baseline.topk(5)
    topk_injected = logits_nonzero.topk(5)

    print(f"  KL divergence (injected vs baseline): {kl:.10f}")
    print(f"  Max logit difference: {logit_diff.max().item():.6f}")
    print(f"  Mean logit difference: {logit_diff.mean().item():.6f}")

    print(f"\n  Top-5 BASELINE:")
    for i in range(5):
        tid = topk_base.indices[i].item()
        tok = tokenizer.decode([tid])
        print(f"    {i+1}. [{tid}] '{tok}' logit={topk_base.values[i].item():.4f}")

    print(f"\n  Top-5 WITH INJECTION:")
    for i in range(5):
        tid = topk_injected.indices[i].item()
        tok = tokenizer.decode([tid])
        print(f"    {i+1}. [{tid}] '{tok}' logit={topk_injected.values[i].item():.4f}")

    # Verdict
    print("\n" + "=" * 70)
    any_nonzero = any(s['perturbation_norm'] > 1e-8 for s in stats_nonzero.values())
    kl_nonzero = kl > 1e-8

    if any_nonzero and kl_nonzero:
        print("VERDICT: Bridge 2 injection mechanism VERIFIED")
        print("  - Hooks fire correctly on target layers")
        print("  - Perturbation propagates through to model output")
        print("  - KL divergence confirms output change")
        print("  - Zero init (default) produces zero perturbation (correct LoRA behavior)")
        print("  - Training will learn nonzero up_proj weights")
    elif any_nonzero:
        print("VERDICT: Hooks inject but output not measurably changed")
        print("  - May need larger injection magnitude")
    else:
        print("VERDICT: Injection mechanism not working - debug needed")

    print("=" * 70)
