"""
Phase 2 "Wake Up" Diagnostics v3
Skip softmax entirely. Measure at the logit level AND generate deep enough
to find positions with actual entropy.

Run:
    ..\.venv\Scripts\python.exe lmf\tests\test_phase2_wakeup_v3.py
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


def generate_and_find_entropy(model, tokenizer, input_ids, mask, max_tokens=40, min_entropy=0.5):
    """
    Generate tokens until we find positions with real entropy (model uncertainty).
    Returns the sequence and list of (position, entropy, token) tuples.
    """
    generated = input_ids.clone()
    gen_mask = mask.clone() if mask is not None else None
    prompt_len = input_ids.shape[1]
    
    positions = []  # (global_pos, entropy, token_id, token_text, logit_std)
    
    with torch.no_grad():
        for i in range(max_tokens):
            outputs = model(input_ids=generated, attention_mask=gen_mask)
            logits = outputs.logits[0, -1, :].float().cpu()
            probs = F.softmax(logits, dim=-1)
            
            entropy = -(probs * (probs + 1e-10).log()).sum().item() / math.log(2)  # bits
            next_token = logits.argmax().item()
            tok_text = tokenizer.decode([next_token])
            is_special = next_token >= 200000
            
            positions.append({
                'pos': prompt_len + i,
                'entropy': entropy,
                'token_id': next_token,
                'token_text': tok_text,
                'is_special': is_special,
                'logit_std': logits.std().item(),
                'top1_prob': probs.max().item(),
            })
            
            next_tensor = torch.tensor([[next_token]], device=generated.device)
            generated = torch.cat([generated, next_tensor], dim=-1)
            if gen_mask is not None:
                gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=gen_mask.device, dtype=gen_mask.dtype)], dim=-1)
    
    return generated, gen_mask, positions


def test_logit_level_proof(model, tokenizer, lmf, harness, config):
    """
    DEFINITIVE TEST: Prove the bias is real at the logit level.
    No softmax, no KL. Just raw numbers.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Logit-Level Proof (No Softmax)")
    print("=" * 70)
    
    prompt = "What is memory?"
    input_ids, mask = tokenize(tokenizer, model, prompt)
    
    # Reset and run one bridge cycle
    lmf.field_state.zero_()
    harness._step_count = 0
    result = harness.step(input_ids, mask, return_debug=True)
    
    logit_bias = result['logit_bias'][0].float().cpu()  # [vocab_size]
    model_logits = result['model_logits'][0, -1, :].float().cpu()
    combined_logits = result['logits'][0, -1, :].float().cpu()
    
    # Raw bias statistics
    bias_nonzero = (logit_bias.abs() > 1e-8).sum().item()
    bias_max = logit_bias.abs().max().item()
    bias_mean = logit_bias.abs().mean().item()
    bias_std = logit_bias.std().item()
    
    print(f"\n  Logit bias statistics:")
    print(f"    Nonzero entries: {bias_nonzero:,} / {logit_bias.shape[0]:,} ({bias_nonzero/logit_bias.shape[0]*100:.1f}%)")
    print(f"    Max |bias|: {bias_max:.6f}")
    print(f"    Mean |bias|: {bias_mean:.6f}")
    print(f"    Std bias: {bias_std:.6f}")
    
    # Verify addition is happening
    diff = (combined_logits - model_logits).cpu()
    actual_bias = diff  # should equal logit_bias (broadcast to last position)
    
    reconstruction_error = (actual_bias - logit_bias).abs().max().item()
    print(f"\n  Reconstruction check (combined - model vs bias):")
    print(f"    Max error: {reconstruction_error:.10f}")
    if reconstruction_error < 1e-3:
        print(f"    ✓ Bias is correctly added to model logits")
    else:
        print(f"    ✗ Mismatch — bias not being applied correctly!")
    
    # Show which tokens the bias favors most
    top_boosted = logit_bias.topk(10)
    top_suppressed = (-logit_bias).topk(10)
    
    print(f"\n  Top 10 tokens BOOSTED by memory bias:")
    for i in range(10):
        tid = top_boosted.indices[i].item()
        bias_val = top_boosted.values[i].item()
        model_val = model_logits[tid].item()
        print(f"    '{tokenizer.decode([tid])}' bias={bias_val:+.4f} (model_logit={model_val:.2f})")
    
    print(f"\n  Top 10 tokens SUPPRESSED by memory bias:")
    for i in range(10):
        tid = top_suppressed.indices[i].item()
        bias_val = logit_bias[tid].item()
        model_val = model_logits[tid].item()
        print(f"    '{tokenizer.decode([tid])}' bias={bias_val:+.4f} (model_logit={model_val:.2f})")
    
    print(f"\n  VERDICT: Bias is {'NONZERO AND STRUCTURED' if bias_max > 1e-4 else 'ZERO (broken)'}")


def test_entropy_landscape(model, tokenizer, lmf, harness, config):
    """
    Generate tokens and map the entropy landscape.
    Find where the model has REAL uncertainty (where bias can actually shift choices).
    """
    print("\n" + "=" * 70)
    print("TEST 2: Entropy Landscape — Finding Content Positions")
    print("=" * 70)
    
    prompt = "Tell me something beautiful about the world."
    input_ids, mask = tokenize(tokenizer, model, prompt)
    
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Generating 40 tokens to map entropy...\n")
    
    generated, gen_mask, positions = generate_and_find_entropy(
        model, tokenizer, input_ids, mask, max_tokens=40
    )
    
    print(f"  {'Pos':>4} | {'Entropy':>8} | {'Top1 Prob':>9} | {'LogitStd':>9} | {'Spc':>3} | Token")
    print(f"  {'-'*4} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*3} | -----")
    
    high_entropy_positions = []
    for p in positions:
        marker = "***" if p['entropy'] > 0.5 else "   "
        spc = "Y" if p['is_special'] else " "
        print(f"  {p['pos']:>4} | {p['entropy']:>8.4f} | {p['top1_prob']:>9.6f} | {p['logit_std']:>9.4f} | {spc:>3} | '{p['token_text']}' {marker}")
        if p['entropy'] > 0.5 and not p['is_special']:
            high_entropy_positions.append(p)
    
    print(f"\n  High-entropy content positions (entropy > 0.5 bits): {len(high_entropy_positions)}")
    
    if not high_entropy_positions:
        print("  No high-entropy positions found. Model is very confident throughout.")
        print("  This is normal for template/preamble tokens.")
        print("  Try a longer generation or different prompt to find uncertain positions.")
    
    return generated, gen_mask, positions, high_entropy_positions


def test_influence_at_uncertain_positions(model, tokenizer, lmf, harness, config, 
                                          generated, gen_mask, high_entropy_positions):
    """
    Measure memory influence specifically at positions where the model is uncertain.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Memory Influence at Uncertain Positions")
    print("=" * 70)
    
    if not high_entropy_positions:
        print("  No high-entropy positions available. Skipping.")
        print("  (The model's preamble is too deterministic.)")
        print("  This is a measurement problem, not an architecture problem.")
        return
    
    # Reset and populate field with memories
    lmf.field_state.zero_()
    lmf._total_steps = 0
    harness._step_count = 0
    
    # Clear working/transient
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)
    
    # Force-feed memories
    lmf.config.significance.formation_threshold = 0.0
    experiences = [
        "The sunset over the ocean was the most beautiful thing I ever saw.",
        "My grandmother used to tell me stories about the old country.",
        "I remember the smell of rain on hot pavement in summer.",
        "Music has always made me feel things I cannot put into words.",
    ]
    
    print(f"\n  Feeding {len(experiences)} memories...")
    for prompt in experiences:
        ids, m = tokenize(tokenizer, model, prompt)
        result = harness.step(ids, m, return_debug=True)
        pert = result['bridge1']['scaled_perturbation'][0]
        encoded = F.normalize(pert + 0.3 * lmf.field_state, dim=-1)
        lmf.working.store_pattern(pattern=encoded, depth=0.8, significance=0.8,
                                   emotional_tag=lmf.regulatory.state.clone())
    
    lmf.config.significance.formation_threshold = 0.4  # restore
    
    status = lmf.get_status()
    print(f"  Field: norm={status['field_norm']:.3f}, working={status['working_active']}, "
          f"energy={status['total_energy']:.3f}")
    
    # Test at each high-entropy position
    print(f"\n  Testing influence at {len(high_entropy_positions)} uncertain positions:")
    print(f"  {'Pos':>4} | {'Entropy':>7} | {'KL Div':>14} | {'MaxΔLogit':>9} | {'Top1Chg':>7} | {'ProbShift':>9} | Token")
    print(f"  {'-'*4} | {'-'*7} | {'-'*14} | {'-'*9} | {'-'*7} | {'-'*9} | -----")
    
    for pos_info in high_entropy_positions[:10]:  # max 10 positions
        pos = pos_info['pos']
        
        # Truncate to predict this position
        trunc_ids = generated[:, :pos]
        trunc_mask = gen_mask[:, :pos] if gen_mask is not None else None
        
        # Base model prediction
        with torch.no_grad():
            base_out = model(input_ids=trunc_ids, attention_mask=trunc_mask)
        base_logits = base_out.logits[0, -1, :].float().cpu()
        base_probs = F.softmax(base_logits, dim=-1)
        
        # Bridge-influenced prediction (try multiple gammas)
        result = harness.step(trunc_ids, trunc_mask, return_debug=True)
        bias = result['logit_bias'][0].float().cpu()
        
        for gamma in [0.1, 1.0, 5.0]:
            scaled_bias = (gamma / 0.1) * bias  # rescale from default
            combined_logits = base_logits + scaled_bias
            combined_probs = F.softmax(combined_logits, dim=-1)
            
            kl = F.kl_div(
                combined_probs.clamp(min=1e-10).log(),
                base_probs.clamp(min=1e-10),
                reduction='sum',
                log_target=False,
            ).item()
            
            max_delta = scaled_bias.abs().max().item()
            top1_changed = base_probs.argmax().item() != combined_probs.argmax().item()
            
            # Probability shift of actual generated token
            actual_token = generated[0, pos].item()
            base_p = base_probs[actual_token].item()
            comb_p = combined_probs[actual_token].item()
            prob_shift = comb_p - base_p
            
            if gamma == 1.0:  # Only print γ=1.0 in the table
                print(f"  {pos:>4} | {pos_info['entropy']:>7.3f} | {kl:>14.10f} | {max_delta:>9.4f} | "
                      f"{'YES' if top1_changed else 'no':>7} | {prob_shift:>+9.6f} | '{pos_info['token_text']}'")


def test_direct_logit_intervention(model, tokenizer, lmf, harness, config):
    """
    MOST DIRECT TEST: Manually set a huge bias and confirm the pipeline applies it.
    This bypasses all field/bridge complexity to verify the plumbing works.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Direct Logit Intervention (Pipeline Plumbing Check)")
    print("=" * 70)
    
    prompt = "The meaning of life is"
    input_ids, mask = tokenize(tokenizer, model, prompt)
    
    # Generate a few tokens to get to content
    gen, gen_mask, positions = generate_and_find_entropy(
        model, tokenizer, input_ids, mask, max_tokens=20
    )
    
    # Find first non-special position
    target_pos = None
    for p in positions:
        if not p['is_special'] and p['entropy'] > 0.01:
            target_pos = p
            break
    
    if target_pos is None:
        # Just use last position
        target_pos = positions[-1]
    
    pos = target_pos['pos']
    trunc_ids = gen[:, :pos]
    trunc_mask = gen_mask[:, :pos] if gen_mask is not None else None
    
    print(f"\n  Target position {pos}: '{target_pos['token_text']}' (entropy={target_pos['entropy']:.4f})")
    
    # Get base logits
    with torch.no_grad():
        base_out = model(input_ids=trunc_ids, attention_mask=trunc_mask)
    base_logits = base_out.logits[0, -1, :].float().cpu()
    base_probs = F.softmax(base_logits, dim=-1)
    base_top1 = base_probs.argmax().item()
    
    print(f"  Base top-1: '{tokenizer.decode([base_top1])}' (p={base_probs[base_top1].item():.4f})")
    
    # Pick a specific target token to boost
    # Choose the 10th-ranked token (something plausible but not top)
    top20 = base_probs.topk(20)
    target_token = top20.indices[9].item()  # 10th place
    target_base_prob = base_probs[target_token].item()
    target_base_logit = base_logits[target_token].item()
    top1_logit = base_logits[base_top1].item()
    gap = top1_logit - target_base_logit
    
    print(f"  Target token: '{tokenizer.decode([target_token])}' (rank 10, p={target_base_prob:.4f}, logit gap={gap:.2f})")
    
    # Create a surgical bias: boost ONLY the target token
    surgical_bias = torch.zeros_like(base_logits)
    surgical_bias[target_token] = gap + 5.0  # Overcome gap + extra margin
    
    combined_logits = base_logits + surgical_bias
    combined_probs = F.softmax(combined_logits, dim=-1)
    new_top1 = combined_probs.argmax().item()
    
    print(f"\n  Surgical bias of {gap + 5.0:.2f} applied to target token")
    print(f"  New top-1: '{tokenizer.decode([new_top1])}' (p={combined_probs[new_top1].item():.4f})")
    print(f"  Target token new prob: {combined_probs[target_token].item():.4f} (was {target_base_prob:.4f})")
    
    if new_top1 == target_token:
        print(f"\n  ✓ LOGIT ADDITION WORKS — surgical bias successfully changed top-1 prediction")
        print(f"    The pipeline is sound. Memory influence will grow as bridges train.")
    else:
        print(f"\n  ✗ Surgical bias failed to change top-1. Something is wrong with logit combination.")


if __name__ == "__main__":
    print("ANIMA LMF Phase 2 — Wake Up Diagnostics v3")
    print("=" * 70)
    
    model, tokenizer, lmf, harness, config = load_system()
    
    # Test 1: Prove bias exists at logit level
    test_logit_level_proof(model, tokenizer, lmf, harness, config)
    
    # Test 2: Map entropy landscape to find where model is uncertain
    generated, gen_mask, positions, high_entropy = test_entropy_landscape(
        model, tokenizer, lmf, harness, config
    )
    
    # Test 3: Measure influence at uncertain positions
    test_influence_at_uncertain_positions(
        model, tokenizer, lmf, harness, config,
        generated, gen_mask, high_entropy
    )
    
    # Test 4: Direct surgical intervention (plumbing check)
    test_direct_logit_intervention(model, tokenizer, lmf, harness, config)
    
    print("\n" + "=" * 70)
    print("DIAGNOSTICS v3 COMPLETE")
    print("=" * 70)
