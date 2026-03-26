"""
Significance Calibration - Validation
Runs AFTER applying fixes to significance.py

Checks:
1. Component weight renormalization with only novelty+surprise active
2. Surprise now uses cosine distance (should be ~0.5 untrained, not ~0.38)
3. Total significance exceeds formation threshold
4. Memories actually form naturally (no forced threshold)

Run:
    ..\.venv\Scripts\python.exe lmf\tests\test_significance_validation.py
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


def test_significance_components(lmf, config):
    """Test that weight renormalization works correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: Weight Renormalization")
    print("=" * 70)

    sd = lmf.significance
    raw_weights = sd.component_weights.data
    labels = ['novelty', 'emotion', 'surprise', 'value', 'goal']

    active_mask = torch.tensor([True, False, True, False, False])
    test_weights = raw_weights.clone()
    test_weights[~active_mask] = float('-inf')
    renormed = F.softmax(test_weights, dim=0)

    print(f"\n  With novelty + surprise active only:")
    for i, label in enumerate(labels):
        print(f"    {label:>10}: {renormed[i].item():.4f}")

    novelty_weight = renormed[0].item()
    surprise_weight = renormed[2].item()
    print(f"\n  Active weight sum: {novelty_weight + surprise_weight:.4f}")

    for nov, sur, label in [(1.0, 0.5, "Nov=1.0, Sur=0.5"),
                             (1.0, 0.8, "Nov=1.0, Sur=0.8"),
                             (0.5, 0.5, "Nov=0.5, Sur=0.5"),
                             (0.3, 0.7, "Nov=0.3, Sur=0.7")]:
        sig = novelty_weight * nov + surprise_weight * sur
        status = "V" if sig > config.significance.formation_threshold else "X"
        print(f"    {status} {label}: sig={sig:.4f} (threshold={config.significance.formation_threshold})")


def test_surprise_range(lmf):
    """Test that surprise now uses cosine distance."""
    print("\n" + "=" * 70)
    print("TEST 2: Surprise Range (Cosine Distance)")
    print("=" * 70)

    sd = lmf.significance
    field_state = torch.zeros(lmf.field_dim)

    for label, inp in [("Random normalized", F.normalize(torch.randn(lmf.field_dim), dim=0)),
                        ("Small magnitude (0.088)", F.normalize(torch.randn(lmf.field_dim), dim=0) * 0.088),
                        ("Tiny magnitude (0.001)", F.normalize(torch.randn(lmf.field_dim), dim=0) * 0.001)]:
        with torch.no_grad():
            surprise = sd._compute_surprise(inp, field_state)
        print(f"  {label}: surprise = {surprise.item():.6f}")

    field_state = torch.randn(lmf.field_dim) * 0.5
    print(f"\n  With random field state (norm={field_state.norm():.3f}):")
    for label, inp in [("Random normalized", F.normalize(torch.randn(lmf.field_dim), dim=0)),
                        ("Small magnitude (0.088)", F.normalize(torch.randn(lmf.field_dim), dim=0) * 0.088)]:
        with torch.no_grad():
            surprise = sd._compute_surprise(inp, field_state)
        print(f"  {label}: surprise = {surprise.item():.6f}")


def test_natural_memory_formation(model, tokenizer, lmf, harness, config):
    """THE REAL TEST: Do memories form naturally?"""
    print("\n" + "=" * 70)
    print("TEST 3: Natural Memory Formation (No Forced Threshold)")
    print("=" * 70)

    lmf.field_state.zero_()
    lmf._total_steps = 0
    harness._step_count = 0
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)

    print(f"\n  Formation threshold: {config.significance.formation_threshold}")

    prompts = [
        "The sunset over the ocean was the most beautiful thing I ever saw.",
        "My grandmother used to tell me stories about the old country.",
        "I remember the smell of rain on hot pavement in summer.",
        "The first time I held a book, I felt something click inside me.",
        "Music has always made me feel things I cannot put into words.",
        "The stars looked different that night, like they were closer.",
        "I learned more from failure than I ever did from success.",
        "The sound of waves has always calmed me.",
    ]

    print(f"\n  {'#':>3} | {'Prompt':50} | {'Sig':>6} | {'Work':>4} | {'Trans':>5} | {'Norm':>6} | {'Energy':>8} | Formed?")
    print(f"  {'-'*3} | {'-'*50} | {'-'*6} | {'-'*4} | {'-'*5} | {'-'*6} | {'-'*8} | -------")

    total_formed = 0
    for i, prompt in enumerate(prompts):
        ids, mask = tokenize(tokenizer, model, prompt)
        before_working = lmf.working.num_active

        result = harness.step(ids, mask, return_debug=True)

        after_working = lmf.working.num_active
        formed = after_working > before_working
        if formed:
            total_formed += 1

        sig_bridge1 = result['significance'].item()
        status = lmf.get_status()

        print(f"  {i+1:>3} | {prompt[:50]:50} | {sig_bridge1:.4f} | {status['working_active']:>4} | "
              f"{status['transient_active']:>5} | {status['field_norm']:>6.3f} | {status['total_energy']:>8.3f} | "
              f"{'YES' if formed else '  no'}")

    print(f"\n  RESULT: {total_formed}/{len(prompts)} prompts formed working memories naturally")
    if total_formed > 0:
        print(f"  SIGNIFICANCE CALIBRATION SUCCESSFUL!")
    else:
        print(f"  Still no natural memory formation. Need further calibration.")
        print(f"  Consider lowering formation_threshold from {config.significance.formation_threshold} to 0.3")


def test_memory_influence(model, tokenizer, lmf, harness, config):
    """If memories formed, test whether they influence generation."""
    print("\n" + "=" * 70)
    print("TEST 4: Memory Influence After Calibration")
    print("=" * 70)

    status = lmf.get_status()
    if status['working_active'] == 0:
        print("  No working memories - skipping influence test.")
        return

    print(f"  Working memories: {status['working_active']}")
    print(f"  Field norm: {status['field_norm']:.4f}")

    test_prompt = "Tell me something beautiful."
    ids, mask = tokenize(tokenizer, model, test_prompt)

    # Generate past template
    generated = ids.clone()
    gen_mask = mask.clone() if mask is not None else None
    with torch.no_grad():
        for _ in range(15):
            out = model(input_ids=generated, attention_mask=gen_mask)
            next_tok = out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_tok], dim=-1)
            if gen_mask is not None:
                gen_mask = torch.cat([gen_mask, torch.ones(1,1,device=gen_mask.device,dtype=gen_mask.dtype)], dim=-1)

    # Find high-entropy content position
    content_pos = None
    with torch.no_grad():
        out = model(input_ids=generated, attention_mask=gen_mask)
    for pos in range(ids.shape[1], generated.shape[1]):
        logits_at = out.logits[0, pos, :].float().cpu()
        probs_at = F.softmax(logits_at, dim=-1)
        entropy = -(probs_at * (probs_at + 1e-10).log()).sum().item()
        if entropy > 1.0 and generated[0, pos].item() < 200000:
            content_pos = pos
            break

    if content_pos is None:
        content_pos = generated.shape[1] - 1

    trunc_ids = generated[:, :content_pos]
    trunc_mask = gen_mask[:, :content_pos] if gen_mask is not None else None

    with torch.no_grad():
        base_out = model(input_ids=trunc_ids, attention_mask=trunc_mask)
    base_logits = base_out.logits[0, -1, :].float().cpu()
    base_probs = F.softmax(base_logits, dim=-1)

    result = harness.step(trunc_ids, trunc_mask, return_debug=True)
    combined_logits = result['logits'][0, -1, :].float().cpu()
    combined_probs = F.softmax(combined_logits, dim=-1)

    kl = F.kl_div(
        combined_probs.clamp(min=1e-10).log(),
        base_probs.clamp(min=1e-10),
        reduction='sum',
        log_target=False,
    ).item()

    tok_text = tokenizer.decode([generated[0, content_pos].item()])
    print(f"\n  Measuring at position {content_pos} ('{tok_text}')")
    print(f"  KL divergence: {kl:.10f}")
    print(f"  Logit bias std: {result['logit_bias'][0].float().std().item():.6f}")

    if kl > 1e-6:
        print(f"\n  Naturally-formed memories are influencing generation!")


if __name__ == "__main__":
    print("ANIMA LMF - Significance Calibration Validation")
    print("=" * 70)

    model, tokenizer, lmf, harness, config = load_system()

    test_significance_components(lmf, config)
    test_surprise_range(lmf)
    test_natural_memory_formation(model, tokenizer, lmf, harness, config)
    test_memory_influence(model, tokenizer, lmf, harness, config)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
