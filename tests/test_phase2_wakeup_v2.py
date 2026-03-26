"""
Phase 2 "Wake Up" Diagnostics v2
Fixed: measure at content generation position, not chat template control tokens.

The v1 results showed KL=0 because the model predicts <|channel|> with prob≈1.0
at the last prompt position. We need to actually GENERATE a few tokens to get past
the chat template scaffolding, then measure influence on real content tokens.

Run:
    ..\.venv\Scripts\python.exe lmf\tests\test_phase2_wakeup_v2.py
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
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
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
    input_ids = inputs["input_ids"].to(model.device)
    mask = inputs.get("attention_mask")
    if mask is not None:
        mask = mask.to(model.device)
    return input_ids, mask


def find_content_position(model, tokenizer, input_ids, attention_mask, n_prefix=8):
    """
    Generate a few tokens to get past chat template scaffolding,
    then return the input_ids and position where real content generation starts.
    
    Returns extended input_ids, mask, and the index of the first content token.
    """
    print(f"  Generating {n_prefix} prefix tokens to find content position...")
    generated = input_ids.clone()
    gen_mask = attention_mask.clone() if attention_mask is not None else None
    
    with torch.no_grad():
        for i in range(n_prefix):
            outputs = model(
                input_ids=generated,
                attention_mask=gen_mask,
            )
            next_logits = outputs.logits[0, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=-1)
            if gen_mask is not None:
                gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=gen_mask.device, dtype=gen_mask.dtype)], dim=-1)
            
            tok_text = tokenizer.decode(next_token[0].tolist())
            is_special = next_token.item() >= 200000
            print(f"    [{i}] token {next_token.item()}: '{tok_text}' {'(special)' if is_special else ''}")
    
    content_start = input_ids.shape[1]  # first generated position
    return generated, gen_mask, content_start


def experiment_1_fixed(model, tokenizer, lmf, harness, config):
    """
    Gamma sweep with LOGIT-LEVEL comparison (not probability level).
    Also measures at content position, not chat template position.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Gamma Sweep (Fixed — Logit-Level + Content Position)")
    print("=" * 70)

    prompt = "What is memory?"
    input_ids, mask = tokenize(tokenizer, model, prompt)
    
    # Generate prefix to get past special tokens
    extended_ids, extended_mask, content_start = find_content_position(
        model, tokenizer, input_ids, mask, n_prefix=8
    )
    
    # Find first non-special content position
    content_pos = None
    for i in range(content_start, extended_ids.shape[1]):
        if extended_ids[0, i].item() < 200000:
            content_pos = i
            break
    
    if content_pos is None:
        print("  No content tokens found in prefix. Using last position.")
        content_pos = extended_ids.shape[1] - 1
    
    measure_pos = content_pos
    tok_at_pos = tokenizer.decode([extended_ids[0, measure_pos].item()])
    print(f"\n  Measuring at position {measure_pos} (token: '{tok_at_pos}')")
    
    # Use input up to (but not including) the measurement position
    # so we measure the model's prediction FOR that position
    trunc_ids = extended_ids[:, :measure_pos]
    trunc_mask = extended_mask[:, :measure_pos] if extended_mask is not None else None
    
    # Base model logits at this position
    with torch.no_grad():
        base_out = model(input_ids=trunc_ids, attention_mask=trunc_mask)
    base_logits = base_out.logits[0, -1, :].float().cpu()
    base_probs = F.softmax(base_logits, dim=-1)
    
    # Check distribution entropy
    entropy = -(base_probs * (base_probs + 1e-10).log()).sum().item()
    print(f"  Base distribution entropy: {entropy:.4f} bits")
    print(f"  Base logit std: {base_logits.std().item():.4f}")
    print(f"  Base top-1 prob: {base_probs.max().item():.6f}")
    
    # Decode base top-5
    base_top5 = base_probs.topk(5)
    print(f"  Base top-5:")
    for i in range(5):
        tid = base_top5.indices[i].item()
        print(f"    [{tid}] '{tokenizer.decode([tid])}' (p={base_top5.values[i].item():.4f})")
    
    # Gamma sweep
    gammas = [0.1, 0.5, 1.0, 3.0, 10.0]
    print(f"\n  {'Gamma':>6} | {'KL Div':>14} | {'Bias Std':>10} | {'MaxΔLogit':>10} | {'Top1 Same':>10} | {'Top5 Overlap':>12}")
    print(f"  {'-'*6} | {'-'*14} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")
    
    for gamma in gammas:
        # Reset and run bridge cycle
        lmf.field_state.zero_()
        harness._step_count = 0
        result = harness.step(trunc_ids, trunc_mask, return_debug=True)
        
        # Manual Bridge 3 with custom gamma
        field_state = lmf.field_state.clone().cpu().float()
        if field_state.dim() == 1:
            field_state = field_state.unsqueeze(0)
        
        ob = harness.output_bridge
        gate = torch.sigmoid(ob.transform_gate)
        transformed = ob.transform(field_state)
        mixed = gate * transformed + (1 - gate) * field_state
        mixed = ob.transform_norm(mixed)
        
        lm_weight = ob._lm_head.weight
        raw_bias = F.linear(mixed.to(dtype=lm_weight.dtype), lm_weight).float()[0]
        logit_bias = gamma * raw_bias
        
        combined_logits = base_logits + logit_bias
        combined_probs = F.softmax(combined_logits, dim=-1)
        
        # Metrics
        kl = F.kl_div(
            combined_probs.clamp(min=1e-10).log(),
            base_probs.clamp(min=1e-10),
            reduction='sum',
            log_target=False,
        ).item()
        
        bias_std = logit_bias.std().item()
        max_delta = (combined_logits - base_logits).abs().max().item()
        
        base_top1 = base_probs.argmax().item()
        combined_top1 = combined_probs.argmax().item()
        top1_same = base_top1 == combined_top1
        
        base_top5_set = set(base_probs.topk(5).indices.tolist())
        combined_top5_set = set(combined_probs.topk(5).indices.tolist())
        top5_overlap = len(base_top5_set & combined_top5_set)
        
        print(f"  {gamma:>6.1f} | {kl:>14.10f} | {bias_std:>10.4f} | {max_delta:>10.4f} | {'Yes' if top1_same else 'NO':>10} | {top5_overlap:>12d}/5")
    
    print()


def experiment_2_fixed(model, tokenizer, lmf, harness, config):
    """
    Force memory formation, then measure influence at content position.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Forced Memory → Content Generation Influence")
    print("=" * 70)
    
    # Reset everything
    lmf.field_state.zero_()
    lmf._total_steps = 0
    harness._step_count = 0
    
    # Clear memory layers
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)
    
    # Feed experiences with forced significance
    experiences = [
        "The sunset over the ocean was the most beautiful thing I ever saw.",
        "My grandmother used to tell me stories about the old country.",
        "I remember the smell of rain on hot pavement in summer.",
        "The first time I held a book, I felt something click inside me.",
        "Music has always made me feel things I cannot put into words.",
    ]
    
    original_threshold = lmf.config.significance.formation_threshold
    lmf.config.significance.formation_threshold = 0.0
    
    print("\n  Phase A: Building memories...")
    for i, prompt in enumerate(experiences):
        input_ids, mask = tokenize(tokenizer, model, prompt)
        result = harness.step(input_ids, mask, return_debug=True)
        
        # Manually store a strong memory from the bridge perturbation
        pert = result['bridge1']['scaled_perturbation'][0]
        
        encoded = F.normalize(pert + 0.3 * lmf.field_state, dim=-1)
        lmf.working.store_pattern(
            pattern=encoded, depth=0.8, significance=0.8,
            emotional_tag=lmf.regulatory.state.clone(),
        )
        
        status = lmf.get_status()
        print(f"  [{i+1}] working={status['working_active']}, "
              f"field_norm={status['field_norm']:.3f}, "
              f"energy={status['total_energy']:.3f}")
    
    lmf.config.significance.formation_threshold = original_threshold
    
    status = lmf.get_status()
    print(f"\n  Memories: {status['working_active']} working, "
          f"{status['transient_active']} transient")
    print(f"  Field norm: {status['field_norm']:.4f}")
    print(f"  Total energy: {status['total_energy']:.4f}")
    
    # Phase B: Test influence at content generation position
    print("\n  Phase B: Measuring influence at content position...")
    
    test_prompt = "Tell me something beautiful."
    input_ids, mask = tokenize(tokenizer, model, test_prompt)
    
    # Generate prefix to get past special tokens
    extended_ids, extended_mask, content_start = find_content_position(
        model, tokenizer, input_ids, mask, n_prefix=8
    )
    
    # Find content position
    content_pos = None
    for i in range(content_start, extended_ids.shape[1]):
        if extended_ids[0, i].item() < 200000:
            content_pos = i
            break
    if content_pos is None:
        content_pos = extended_ids.shape[1] - 1
    
    trunc_ids = extended_ids[:, :content_pos]
    trunc_mask = extended_mask[:, :content_pos] if extended_mask is not None else None
    
    # Base logits
    with torch.no_grad():
        base_out = model(input_ids=trunc_ids, attention_mask=trunc_mask)
    base_logits = base_out.logits[0, -1, :].float().cpu()
    base_probs = F.softmax(base_logits, dim=-1)
    
    entropy = -(base_probs * (base_probs + 1e-10).log()).sum().item()
    print(f"\n  Content position entropy: {entropy:.4f} bits")
    
    # Combined logits with memory-populated field
    result = harness.step(trunc_ids, trunc_mask, return_debug=True)
    combined_logits = result['logits'][0, -1, :].float().cpu()
    combined_probs = F.softmax(combined_logits, dim=-1)
    logit_bias = result['logit_bias'][0].float().cpu()
    
    # Also test with boosted gamma
    for gamma_label, gamma_val in [("default (0.1)", 0.1), ("boosted (1.0)", 1.0), ("strong (5.0)", 5.0)]:
        if gamma_val == 0.1:
            test_combined = combined_logits
        else:
            test_combined = base_logits + gamma_val * (logit_bias / 0.1)  # rescale from default gamma
        
        test_probs = F.softmax(test_combined, dim=-1)
        
        kl = F.kl_div(
            test_probs.clamp(min=1e-10).log(),
            base_probs.clamp(min=1e-10),
            reduction='sum',
            log_target=False,
        ).item()
        
        base_top5 = base_probs.topk(5)
        test_top5 = test_probs.topk(5)
        
        print(f"\n  === Gamma {gamma_label} ===")
        print(f"  KL divergence: {kl:.10f}")
        print(f"  Base top-5:")
        for i in range(5):
            tid = base_top5.indices[i].item()
            p = base_top5.values[i].item()
            # Also show this token's combined prob
            cp = test_probs[tid].item()
            delta = cp - p
            print(f"    '{tokenizer.decode([tid])}' base={p:.4f} combined={cp:.4f} (Δ={delta:+.4f})")
        
        print(f"  Combined top-5:")
        for i in range(5):
            tid = test_top5.indices[i].item()
            p = test_probs[tid].item()
            bp = base_probs[tid].item()
            delta = p - bp
            print(f"    '{tokenizer.decode([tid])}' combined={p:.4f} base={bp:.4f} (Δ={delta:+.4f})")


if __name__ == "__main__":
    print("ANIMA LMF Phase 2 — Wake Up Diagnostics v2")
    print("=" * 70)
    
    model, tokenizer, lmf, harness, config = load_system()
    experiment_1_fixed(model, tokenizer, lmf, harness, config)
    experiment_2_fixed(model, tokenizer, lmf, harness, config)
    
    print("\n" + "=" * 70)
    print("DIAGNOSTICS v2 COMPLETE")
    print("=" * 70)
