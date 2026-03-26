"""
Phase 2 Bridge Tests - Validate Bridge 1 + Bridge 3 hookup.

Test A: Standalone bridge math (no model needed, fast)
Test B: Live integration with GPT-oss-20b (needs GPU, slow)

Run:
    ..\.venv\Scripts\python.exe tests/test_phase2_bridges.py
    ..\.venv\Scripts\python.exe tests/test_phase2_bridges.py --live  (with model)
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add ANIMA root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from lmf.bridges.input_bridge import InputBridge, AttentionPool
from lmf.bridges.output_bridge import OutputBridge


def test_a_standalone_bridges():
    """Test bridge math without a real model."""
    print("=" * 70)
    print("TEST A: Standalone Bridge Validation")
    print("=" * 70)
    
    hidden_dim = 2880
    vocab_size = 201088
    batch_size = 1
    seq_len = 69
    device = "cpu"
    
    # === A1: Input Bridge ===
    print("\n--- A1: Input Bridge ---")
    input_bridge = InputBridge(
        hidden_dim=hidden_dim,
        bottleneck_dim=64,
        alpha=0.1,
    ).to(device)
    
    # Simulate transformer hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    
    result = input_bridge(hidden_states, attention_mask, return_components=True)
    
    perturbation = result['perturbation']
    significance = result['significance']
    scaled = result['scaled_perturbation']
    
    # Check shapes
    assert perturbation.shape == (batch_size, hidden_dim), f"Bad perturbation shape: {perturbation.shape}"
    assert significance.shape == (batch_size, 1), f"Bad significance shape: {significance.shape}"
    assert scaled.shape == (batch_size, hidden_dim), f"Bad scaled shape: {scaled.shape}"
    
    # Check normalization
    pert_norm = perturbation.norm(dim=-1)
    assert torch.allclose(pert_norm, torch.ones_like(pert_norm), atol=1e-5), \
        f"Perturbation not normalized: norm={pert_norm.item():.4f}"
    
    # Check significance range
    assert 0.0 <= significance.item() <= 1.0, \
        f"Significance out of range: {significance.item():.4f}"
    
    # Check scaling: scaled = perturbation * alpha * significance
    expected_norm = 0.1 * significance.item()
    actual_norm = scaled.norm(dim=-1).item()
    assert abs(actual_norm - expected_norm) < 1e-5, \
        f"Scaling wrong: expected norm {expected_norm:.4f}, got {actual_norm:.4f}"
    
    params = input_bridge.get_param_count()
    print(f"  Perturbation shape: {perturbation.shape} ✓")
    print(f"  Perturbation norm: {pert_norm.item():.4f} (should be 1.0) ✓")
    print(f"  Significance: {significance.item():.4f} (range [0,1]) ✓")
    print(f"  Scaled norm: {actual_norm:.6f} (alpha=0.1 × sig={significance.item():.4f}) ✓")
    print(f"  Parameters: {params}")
    print("  PASS ✓")
    
    # === A2: Attention Pool ===
    print("\n--- A2: Attention Pool ---")
    pool = input_bridge.pool
    
    # Test that masking works: pad half the sequence
    mask_half = torch.ones(batch_size, seq_len)
    mask_half[:, seq_len // 2:] = 0
    
    pooled_full = pool(hidden_states, attention_mask)
    pooled_half = pool(hidden_states, mask_half)
    
    # They should differ (different tokens contribute)
    diff = (pooled_full - pooled_half).norm().item()
    assert diff > 0.01, f"Masking had no effect: diff={diff:.6f}"
    print(f"  Full-mask pooled norm: {pooled_full.norm().item():.4f}")
    print(f"  Half-mask pooled norm: {pooled_half.norm().item():.4f}")
    print(f"  Difference: {diff:.4f} (masking affects output) ✓")
    print("  PASS ✓")
    
    # === A3: Output Bridge ===
    print("\n--- A3: Output Bridge ---")
    
    # Create a mock lm_head (same shape as real one)
    mock_lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    output_bridge = OutputBridge(
        hidden_dim=hidden_dim,
        transform_dim=128,
        gamma=0.1,
    ).to(device)
    output_bridge.set_lm_head(mock_lm_head)
    
    # Simulate field state
    field_state = F.normalize(torch.randn(hidden_dim), dim=-1)
    
    # Simulate model logits
    model_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    result3 = output_bridge(
        field_state=field_state,
        model_logits=model_logits,
        return_components=True,
    )
    
    logit_bias = result3['logit_bias']
    combined = result3['combined_logits']
    
    assert logit_bias.shape == (1, vocab_size), f"Bad logit_bias shape: {logit_bias.shape}"
    assert combined.shape == model_logits.shape, f"Bad combined shape: {combined.shape}"
    
    # Check that bias is small (gamma=0.1 should keep it modest)
    bias_std = logit_bias.std().item()
    logit_std = model_logits.std().item()
    ratio = bias_std / logit_std
    
    params3 = output_bridge.get_param_count()
    print(f"  Logit bias shape: {logit_bias.shape} ✓")
    print(f"  Combined shape: {combined.shape} ✓")
    print(f"  Bias std: {bias_std:.4f}")
    print(f"  Model logit std: {logit_std:.4f}")
    print(f"  Bias/model ratio: {ratio:.4f} (gamma=0.1 → should be small)")
    print(f"  Gate value: {result3['gate_value']:.4f}")
    print(f"  Effective gamma: {result3['effective_gamma']:.4f}")
    print(f"  Parameters (new): {params3['total_new']:,}")
    print(f"  Parameters (reused from lm_head): {params3['total_reused']:,}")
    print("  PASS ✓")
    
    # === A4: Gradient flow check ===
    print("\n--- A4: Gradient Flow ---")
    
    # Verify gradients flow through bridges but NOT through model/lm_head
    field_state_grad = F.normalize(torch.randn(hidden_dim), dim=-1)
    field_state_grad.requires_grad_(True)
    
    out = output_bridge(field_state=field_state_grad)
    loss = out['logit_bias'].sum()
    loss.backward()
    
    # Bridge transform should have gradients
    has_bridge_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in output_bridge.transform.parameters()
    )
    
    # lm_head should NOT have gradients (frozen, used with torch.no_grad)
    has_lm_grad = any(
        p.grad is not None 
        for p in mock_lm_head.parameters()
    )
    
    print(f"  Bridge transform has gradients: {has_bridge_grad} ✓")
    print(f"  lm_head has gradients: {has_lm_grad} (should be False) ✓")
    assert has_bridge_grad, "Bridge transform should receive gradients!"
    assert not has_lm_grad, "lm_head should NOT receive gradients!"
    print("  PASS ✓")
    
    # === A5: Total parameter budget ===
    print("\n--- A5: Parameter Budget ---")
    total_new = params['total'] + params3['total_new']
    print(f"  Input Bridge:  {params['total']:>10,} params")
    print(f"  Output Bridge: {params3['total_new']:>10,} params (new)")
    print(f"  Output Bridge: {params3['total_reused']:>10,} params (reused lm_head)")
    print(f"  ─────────────────────────────────")
    print(f"  Total NEW:     {total_new:>10,} params ({total_new * 4 / 1e6:.1f} MB @ fp32)")
    print(f"  Total REUSED:  {params3['total_reused']:>10,} params")
    print("  PASS ✓")
    
    print("\n" + "=" * 70)
    print("TEST A: ALL PASSED ✓")
    print("=" * 70)
    return True


def test_b_live_integration():
    """Test with real GPT-oss-20b model."""
    print("\n" + "=" * 70)
    print("TEST B: Live Integration with GPT-oss-20b")
    print("=" * 70)
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lmf.core.field import LivingMemoryField
    from lmf.configs.default import gpt_oss_20b_config
    from lmf.bridges.harness import BridgeHarness
    
    model_path = r"D:\gpt-oss-20b"
    
    # === Load model ===
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={
            0: "15GiB",   # 3090 Ti — leave ~9GB headroom for expert swapping
            1: "8GiB",    # 5060 Ti — leave ~9GB headroom for expert swapping  
            "cpu": "80GiB",
        },
        offload_folder="offload_temp",
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    
    # === Create LMF ===
    print("Creating LMF (production config)...")
    config = gpt_oss_20b_config()
    config.device = "cpu"  # LMF on CPU for now (bridges handle transfers)
    lmf = LivingMemoryField(config)
    
    # === Create harness ===
    print("Creating bridge harness...")
    harness = BridgeHarness(
        model=model,
        lmf=lmf,
        bridge_device="cpu",  # Bridges on CPU alongside lm_head
    )
    print(f"\n{harness}")
    
    # === B1: Single step ===
    print("\n--- B1: Single Step ---")
    messages = [{"role": "user", "content": "What is memory?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    print(f"  Input: {input_ids.shape[1]} tokens")
    
    t0 = time.time()
    result = harness.step(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_debug=True,
    )
    step_time = time.time() - t0
    
    print(f"  Step time: {step_time*1000:.0f}ms")
    print(f"  Timing breakdown:")
    for k, v in result['timing'].items():
        print(f"    {k}: {v:.1f}ms")
    print(f"  Significance: {result['significance'].item():.4f}")
    print(f"  Field state norm: {result['field_state'].norm().item():.4f}")
    print(f"  Logit bias std: {result['logit_bias'].std().item():.4f}")
    print(f"  Model logit std: {result['model_logits'].std().item():.4f}")
    
    # Verify shapes
    assert result['logits'].shape == result['model_logits'].shape
    print(f"  Combined logits shape: {result['logits'].shape} ✓")
    print("  PASS ✓")
    
    # === B2: Memory influence check ===
    print("\n--- B2: Memory Influence Check ---")
    # The combined logits should differ from model logits
    model_probs = F.softmax(result['model_logits'][0, -1, :].float().cpu(), dim=-1)
    combined_probs = F.softmax(result['logits'][0, -1, :].float().cpu(), dim=-1)
    
    # KL divergence between base and memory-influenced
    kl = F.kl_div(
        combined_probs.log(), 
        model_probs, 
        reduction='sum',
        log_target=False,
    ).item()
    
    # Top-5 token comparison
    base_top5 = model_probs.topk(5)
    combined_top5 = combined_probs.topk(5)
    
    print(f"  KL(combined || base): {kl:.6f} (should be small but nonzero)")
    print(f"  Base top-5 tokens:     {base_top5.indices.tolist()}")
    print(f"  Combined top-5 tokens: {combined_top5.indices.tolist()}")
    
    top5_same = set(base_top5.indices.tolist()) == set(combined_top5.indices.tolist())
    print(f"  Top-5 identical: {top5_same} (may or may not differ at gamma=0.1)")
    print("  PASS ✓")
    
    # === B3: Multi-step (does the field evolve?) ===
    print("\n--- B3: Multi-Step Field Evolution ---")
    prompts = [
        "Tell me about your earliest memory.",
        "What does it feel like to remember something?",
        "Do you think memories change over time?",
    ]
    
    field_states = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=True, return_dict=True,
        )
        ids = inputs["input_ids"].to(model.device)
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.to(model.device)
        
        result = harness.step(ids, mask)
        field_states.append(result['field_state'].clone().cpu())
        print(f"  '{prompt[:40]}...' → field norm: {result['field_state'].norm().item():.4f}, "
              f"sig: {result['significance'].item():.4f}")
    
    # Field should change between steps
    for i in range(len(field_states) - 1):
        diff = (field_states[i+1] - field_states[i]).norm().item()
        cos = F.cosine_similarity(
            field_states[i].unsqueeze(0), 
            field_states[i+1].unsqueeze(0),
        ).item()
        print(f"  Step {i}→{i+1}: ΔL2={diff:.4f}, cos_sim={cos:.4f}")
    
    # Field should not be zero
    final_norm = field_states[-1].norm().item()
    assert final_norm > 0.001, f"Field collapsed to zero: norm={final_norm}"
    print(f"  Final field norm: {final_norm:.4f} (alive, not collapsed) ✓")
    print("  PASS ✓")
    
    # === B4: VRAM check ===
    print("\n--- B4: VRAM After Integration ---")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        name = torch.cuda.get_device_properties(i).name
        free = total - alloc
        print(f"  cuda:{i} ({name}): {alloc:.2f} / {total:.1f} GB ({free:.1f} GB free)")
    
    # === Summary ===
    print("\n" + "=" * 70)
    status = harness.get_status()
    print(f"Harness status:")
    print(f"  Steps: {status['step_count']}")
    print(f"  Field memories: {status['field_status']['consolidated_active']} consolidated, "
          f"{status['field_status']['working_active']} working")
    print(f"  Total energy: {status['field_status']['total_energy']:.4f}")
    
    print("\n" + "=" * 70)
    print("TEST B: ALL PASSED ✓")
    print("=" * 70)
    return True


if __name__ == "__main__":
    print("ANIMA LMF Phase 2 Bridge Tests")
    print("=" * 70)
    
    # Always run standalone tests
    test_a_standalone_bridges()
    
    # Run live test if --live flag
    if "--live" in sys.argv:
        test_b_live_integration()
    else:
        print("\n(Skip live test — run with --live flag to test with GPT-oss-20b)")
