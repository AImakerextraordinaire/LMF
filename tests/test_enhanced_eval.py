"""
ANIMA Phase 3 - Enhanced Evaluation Suite
Implements Alex's methodology improvements:
1. Distractor-memory ablation (the killer test)
2. Uncertain-position-only metrics
3. Target-set probability mass
4. Higher precision reporting (scientific notation)
"""

import sys, os, time, torch, json
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GPU selection: use largest GPU only
import subprocess
_gpu_result = subprocess.run(
    [sys.executable, "-c",
     "import torch\n"
     "best=-1; best_mem=0\n"
     "for i in range(torch.cuda.device_count()):\n"
     "    t=torch.cuda.get_device_properties(i).total_mem\n"
     "    if t>best_mem: best=i; best_mem=t\n"
     "print(best)"],
    capture_output=True, text=True
)
_best = _gpu_result.stdout.strip()
if _best.isdigit():
    os.environ["CUDA_VISIBLE_DEVICES"] = _best

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness


def load_system(checkpoint_path=None):
    model_path = r"D:\gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "10GiB", 1: "18GiB", "cpu": "80GiB"},
        offload_folder="offload_temp",
    )
    config = gpt_oss_20b_config()
    config.device = "cpu"
    lmf = LivingMemoryField(config)
    harness = BridgeHarness(model=model, lmf=lmf, bridge_device="cpu")
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        harness.input_bridge.load_state_dict(ckpt['input_bridge'])
        harness.output_bridge.load_state_dict(ckpt['output_bridge'])
        harness.memory_bridge.load_state_dict(ckpt['memory_bridge'])
        harness.lmf.load_state_dict(ckpt['lmf'])
        print(f"Checkpoint loaded (step {ckpt.get('step', '?')})")
        if 'bridge2_gates' in ckpt:
            print(f"Bridge 2 gates: {ckpt['bridge2_gates']}")
    return model, tokenizer, lmf, harness


def reset_field(lmf):
    lmf.field_state.zero_()
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)


def tokenize(tokenizer, model, text):
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, return_dict=True,
    )
    ids = inputs["input_ids"].to(model.device)
    mask = inputs.get("attention_mask")
    if mask is not None:
        mask = mask.to(model.device)
    return ids, mask


def feed_context(harness, tokenizer, model, lmf, context_text):
    """Feed context to build memory, return field state."""
    reset_field(lmf)
    ctx_ids, ctx_mask = tokenize(tokenizer, model, context_text)
    with torch.no_grad():
        harness.step(ctx_ids, ctx_mask)
    return lmf.field_state.norm().item()


def get_logits(model, harness, lmf, full_ids, full_mask, use_memory=True):
    """Get logits for a full sequence with or without memory."""
    with torch.no_grad():
        if use_memory:
            result = harness.step(full_ids, full_mask, return_debug=True)
            return result['logits'][0].float().cpu(), result.get('logit_bias', None)
        else:
            outputs = model(input_ids=full_ids, attention_mask=full_mask)
            return outputs.logits[0].float().cpu(), None


# ============================================================
# Test cases with target token sets
# ============================================================

TEST_CASES = [
    {
        "name": "Petra",
        "context": "The ancient city of Petra was carved into rose-red cliffs by the Nabataean people over two thousand years ago. Located in modern-day Jordan, Petra served as a crucial trading hub connecting Arabia, Egypt, and the Mediterranean. The Nabataeans were master hydraulic engineers who built an elaborate system of dams, cisterns, and water channels to sustain their desert city.",
        "target": "Today, Petra remains one of the most remarkable archaeological sites in the world. The Treasury, carved directly into the sandstone cliff face, is perhaps the most iconic structure from this ancient Nabataean civilization. Visitors can still see traces of the sophisticated water management systems.",
        "target_set": ["Petra", "Jordan", "Nab", "trading", "Arabia", "Egypt", "Mediterranean", "Treasury", "sandstone", "cliff", "carved", "water", "desert", "hydraulic", "dams", "ancient"],
    },
    {
        "name": "Marie Curie",
        "context": "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. She moved to Paris to study physics and mathematics at the Sorbonne, where she met Pierre Curie. Together, they discovered two new elements: polonium, named after Marie's homeland of Poland, and radium. Marie became the first woman to win a Nobel Prize.",
        "target": "Her groundbreaking research on radioactivity earned her not one but two Nobel Prizes, making her the first person to win Nobel Prizes in two different sciences. The element polonium, which she named in honor of her native Poland, was discovered through her painstaking work.",
        "target_set": ["Pierre", "Cur", "ie", "pol", "onium", "rad", "ium", "Nobel", "Poland", "Warsaw", "Sorbonne", "physics", "radioactivity", "elements", "discovered", "woman"],
    },
    {
        "name": "Octopus",
        "context": "The octopus is one of the most intelligent invertebrates on Earth. It has three hearts, blue blood, and eight arms lined with suckers. Each arm contains a cluster of neurons, giving octopuses a distributed nervous system. They can change color and texture in milliseconds to camouflage themselves, and have been observed using tools like coconut shells for shelter.",
        "target": "Scientists studying octopus cognition have been amazed by their problem-solving abilities. Their distributed nervous system, with neural clusters in each of their eight arms, allows for remarkably complex behavior.",
        "target_set": ["octopus", "hearts", "blood", "eight", "arms", "suckers", "neurons", "distributed", "nervous", "camouflage", "tools", "coconut", "color", "invertebrate"],
    },
]


def analyze_passage(model, harness, tokenizer, lmf, tc, context_to_feed):
    """
    Analyze a passage with a given memory context.
    Returns detailed metrics.
    """
    # Feed context
    field_norm = feed_context(harness, tokenizer, model, lmf, context_to_feed)

    # Tokenize full passage (context + target) for analysis
    full_text = tc['context'] + " " + tc['target']
    full_ids, full_mask = tokenize(tokenizer, model, full_text)
    seq_len = full_ids.shape[1]

    # Get logits with memory
    mem_logits, logit_bias = get_logits(model, harness, lmf, full_ids, full_mask, use_memory=True)

    # Get logits without memory
    reset_field(lmf)
    base_logits, _ = get_logits(model, harness, lmf, full_ids, full_mask, use_memory=False)

    # Analyze content positions (second half of sequence)
    start_pos = max(seq_len // 2, 10)
    end_pos = seq_len - 2
    positions = list(range(start_pos, min(end_pos, start_pos + 30)))

    # === Metrics ===
    total_positions = 0
    rank_improvements = 0
    uncertain_positions = 0
    uncertain_improvements = 0
    total_kl = 0.0
    top1_flips = 0
    position_details = []

    for pos in positions:
        if pos >= seq_len - 1:
            break

        actual_next = full_ids[0, pos + 1].item()
        if actual_next >= 200000:
            continue

        base_pos = base_logits[pos]
        mem_pos = mem_logits[pos]

        base_probs = F.softmax(base_pos, dim=-1)
        mem_probs = F.softmax(mem_pos, dim=-1)

        base_sorted = base_pos.argsort(descending=True)
        mem_sorted = mem_pos.argsort(descending=True)

        base_rank = (base_sorted == actual_next).nonzero(as_tuple=True)[0].item() + 1
        mem_rank = (mem_sorted == actual_next).nonzero(as_tuple=True)[0].item() + 1
        rank_change = base_rank - mem_rank

        base_p = base_probs[actual_next].item()
        mem_p = mem_probs[actual_next].item()
        base_top1_p = base_probs[base_sorted[0]].item()

        # Is this an uncertain position?
        is_uncertain = base_top1_p < 0.6

        kl = F.kl_div(
            mem_probs.clamp(min=1e-10).log(),
            base_probs.clamp(min=1e-10),
            reduction='sum',
        ).item()

        total_positions += 1
        total_kl += kl
        if rank_change > 0:
            rank_improvements += 1
        if base_sorted[0].item() != mem_sorted[0].item():
            top1_flips += 1

        if is_uncertain:
            uncertain_positions += 1
            if rank_change > 0:
                uncertain_improvements += 1

        position_details.append({
            'pos': pos,
            'token': tokenizer.decode([actual_next]).strip(),
            'base_rank': base_rank,
            'mem_rank': mem_rank,
            'rank_change': rank_change,
            'base_p': base_p,
            'mem_p': mem_p,
            'is_uncertain': is_uncertain,
            'kl': kl,
        })

    # === Target-set probability mass ===
    target_mass_base = 0.0
    target_mass_mem = 0.0
    target_tokens_found = 0

    for word in tc['target_set']:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            continue
        tid = token_ids[0]
        target_tokens_found += 1

        # Average across content positions
        for pos in positions:
            if pos >= seq_len - 1:
                break
            bp = F.softmax(base_logits[pos], dim=-1)[tid].item()
            mp = F.softmax(mem_logits[pos], dim=-1)[tid].item()
            target_mass_base += bp
            target_mass_mem += mp

    return {
        'field_norm': field_norm,
        'total_positions': total_positions,
        'rank_improvements': rank_improvements,
        'rank_improvement_pct': rank_improvements / max(total_positions, 1) * 100,
        'uncertain_positions': uncertain_positions,
        'uncertain_improvements': uncertain_improvements,
        'uncertain_improvement_pct': uncertain_improvements / max(uncertain_positions, 1) * 100,
        'avg_kl': total_kl / max(total_positions, 1),
        'top1_flips': top1_flips,
        'target_mass_base': target_mass_base,
        'target_mass_mem': target_mass_mem,
        'target_mass_delta': target_mass_mem - target_mass_base,
        'target_tokens_found': target_tokens_found,
        'details': position_details,
    }


if __name__ == "__main__":
    print("=" * 90)
    print("ANIMA Phase 3 - Enhanced Evaluation Suite")
    print("  Distractor ablation | Uncertain-position metrics | Target-set mass")
    print("=" * 90)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, "bridges_final.pt")
    model, tokenizer, lmf, harness = load_system(ckpt_path)

    for tc in TEST_CASES:
        print(f"\n{'='*90}")
        print(f"TEST: {tc['name']}")
        print(f"{'='*90}")

        # === 1. Matched memory (correct context) ===
        print(f"\n  --- MATCHED MEMORY ({tc['name']} context -> {tc['name']} target) ---")
        matched = analyze_passage(model, harness, tokenizer, lmf, tc, tc['context'])

        print(f"  Field norm: {matched['field_norm']:.3f}")
        print(f"  All positions:       {matched['rank_improvements']}/{matched['total_positions']} improved ({matched['rank_improvement_pct']:.0f}%)")
        print(f"  Uncertain positions: {matched['uncertain_improvements']}/{matched['uncertain_positions']} improved ({matched['uncertain_improvement_pct']:.0f}%)")
        print(f"  Top-1 flips: {matched['top1_flips']}")
        print(f"  Avg KL/position: {matched['avg_kl']:.6e}")
        print(f"  Target-set mass: base={matched['target_mass_base']:.6e}, mem={matched['target_mass_mem']:.6e}, delta={matched['target_mass_delta']:+.6e}")

        # Show top rank improvements with higher precision
        sorted_details = sorted(matched['details'], key=lambda d: d['rank_change'], reverse=True)
        print(f"\n  Top rank improvements:")
        for d in sorted_details[:8]:
            if d['rank_change'] > 0:
                unc = " [UNCERTAIN]" if d['is_uncertain'] else ""
                print(f"    '{d['token']:<15}' rank {d['base_rank']:>5} -> {d['mem_rank']:>5} ({d['rank_change']:>+4})  P: {d['base_p']:.4e} -> {d['mem_p']:.4e}{unc}")

        # === 2. Distractor memories ===
        other_cases = [t for t in TEST_CASES if t['name'] != tc['name']]

        for distractor in other_cases:
            print(f"\n  --- DISTRACTOR ({distractor['name']} context -> {tc['name']} target) ---")
            dist_result = analyze_passage(model, harness, tokenizer, lmf, tc, distractor['context'])

            print(f"  All positions:       {dist_result['rank_improvements']}/{dist_result['total_positions']} improved ({dist_result['rank_improvement_pct']:.0f}%)")
            print(f"  Uncertain positions: {dist_result['uncertain_improvements']}/{dist_result['uncertain_positions']} improved ({dist_result['uncertain_improvement_pct']:.0f}%)")
            print(f"  Target-set mass delta: {dist_result['target_mass_delta']:+.6e}")
            print(f"  Avg KL/position: {dist_result['avg_kl']:.6e}")

        # === 3. No memory baseline ===
        print(f"\n  --- NO MEMORY (baseline) ---")
        reset_field(lmf)
        no_mem = analyze_passage(model, harness, tokenizer, lmf, tc, "The weather is nice today.")

        print(f"  All positions:       {no_mem['rank_improvements']}/{no_mem['total_positions']} improved ({no_mem['rank_improvement_pct']:.0f}%)")
        print(f"  Target-set mass delta: {no_mem['target_mass_delta']:+.6e}")

        # === 4. Comparison summary ===
        print(f"\n  === SELECTIVITY SUMMARY for {tc['name']} ===")
        print(f"  {'Memory Source':<25} | {'Rank Impr%':>10} | {'Uncertain%':>10} | {'Mass Delta':>12} | {'Avg KL':>10}")
        print(f"  {'-'*25} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*10}")
        print(f"  {'MATCHED (correct)':25} | {matched['rank_improvement_pct']:>9.0f}% | {matched['uncertain_improvement_pct']:>9.0f}% | {matched['target_mass_delta']:>+11.4e} | {matched['avg_kl']:>9.4e}")
        for distractor in other_cases:
            dist_r = analyze_passage(model, harness, tokenizer, lmf, tc, distractor['context'])
            print(f"  {distractor['name'] + ' (distractor)':25} | {dist_r['rank_improvement_pct']:>9.0f}% | {dist_r['uncertain_improvement_pct']:>9.0f}% | {dist_r['target_mass_delta']:>+11.4e} | {dist_r['avg_kl']:>9.4e}")
        print(f"  {'No memory (baseline)':25} | {no_mem['rank_improvement_pct']:>9.0f}% | {no_mem['uncertain_improvement_pct']:>9.0f}% | {no_mem['target_mass_delta']:>+11.4e} | {no_mem['avg_kl']:>9.4e}")

    print(f"\n{'='*90}")
    print("ENHANCED EVALUATION COMPLETE")
    print(f"{'='*90}")
