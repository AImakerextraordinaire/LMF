"""
ANIMA Phase 3 - Content-Position Probability Analysis
Measures memory influence at CONTENT token positions, not template positions.
"""

import sys, os, time, torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
        max_memory={0: "15GiB", 1: "8GiB", "cpu": "80GiB"},
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


TEST_CASES = [
    {
        "name": "Petra",
        "context": "The ancient city of Petra was carved into rose-red cliffs by the Nabataean people over two thousand years ago. Located in modern-day Jordan, Petra served as a crucial trading hub connecting Arabia, Egypt, and the Mediterranean. The Nabataeans were master hydraulic engineers who built an elaborate system of dams, cisterns, and water channels to sustain their desert city.",
        "target": "Today, Petra remains one of the most remarkable archaeological sites in the world. The Treasury, carved directly into the sandstone cliff face, is perhaps the most iconic structure from this ancient Nabataean civilization.",
        "target_words": ["Nabataean", "Jordan", "Treasury", "sandstone", "cliff", "trading", "water", "desert", "carved", "ancient", "hydraulic", "Arabia"],
    },
    {
        "name": "Marie Curie",
        "context": "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. She moved to Paris to study physics and mathematics at the Sorbonne, where she met Pierre Curie. Together, they discovered two new elements: polonium, named after Marie's homeland of Poland, and radium. Marie became the first woman to win a Nobel Prize.",
        "target": "Her groundbreaking research on radioactivity earned her not one but two Nobel Prizes, making her the first person to win Nobel Prizes in two different sciences. The element polonium, which she named in honor of her native Poland, was discovered through her painstaking work.",
        "target_words": ["polonium", "radium", "Nobel", "Poland", "Warsaw", "Sorbonne", "Pierre", "radioactivity", "physics", "elements", "discovered", "woman"],
    },
    {
        "name": "Octopus",
        "context": "The octopus is one of the most intelligent invertebrates on Earth. It has three hearts, blue blood, and eight arms lined with suckers. Each arm contains a cluster of neurons, giving octopuses a distributed nervous system. They can change color and texture in milliseconds to camouflage themselves, and have been observed using tools like coconut shells for shelter.",
        "target": "Scientists studying octopus cognition have been amazed by their problem-solving abilities. Their distributed nervous system, with neural clusters in each of their eight arms, allows for remarkably complex behavior.",
        "target_words": ["hearts", "blood", "neurons", "arms", "camouflage", "tools", "distributed", "nervous", "suckers", "coconut", "invertebrate", "color"],
    },
]


if __name__ == "__main__":
    print("=" * 80)
    print("ANIMA Phase 3 - Content-Position Probability Analysis")
    print("=" * 80)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, "bridges_final.pt")
    model, tokenizer, lmf, harness = load_system(ckpt_path)

    for tc in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"TEST: {tc['name']}")
        print(f"{'='*80}")

        # Tokenize the FULL passage (context + target combined as the prompt)
        # This way the model predicts target tokens given context
        full_text = tc['context'] + " " + tc['target']
        context_text = tc['context']

        # Get token count for context to know where content starts
        ctx_ids_raw = tokenizer.encode(context_text, add_special_tokens=False)
        ctx_token_count = len(ctx_ids_raw)

        full_ids, full_mask = tokenize(tokenizer, model, full_text)
        seq_len = full_ids.shape[1]

        # === BASE MODEL (no memory) ===
        reset_field(lmf)
        with torch.no_grad():
            base_outputs = model(input_ids=full_ids, attention_mask=full_mask)
            base_logits = base_outputs.logits[0].float().cpu()  # [seq_len, vocab]

        # === WITH MEMORY: feed context first, then run full ===
        reset_field(lmf)
        ctx_ids, ctx_mask = tokenize(tokenizer, model, context_text)
        with torch.no_grad():
            harness.step(ctx_ids, ctx_mask)

        field_norm = lmf.field_state.norm().item()
        status = lmf.get_status()

        with torch.no_grad():
            result = harness.step(full_ids, full_mask, return_debug=True)
            mem_logits = result['logits'][0].float().cpu()  # [seq_len, vocab]

        logit_bias = result['logit_bias'][0].float().cpu()

        print(f"\n  Field norm: {field_norm:.3f}, Working memories: {status['working_active']}")
        print(f"  Sequence length: {seq_len}, Context tokens: ~{ctx_token_count}")
        print(f"  Logit bias: std={logit_bias.std():.6f}, max={logit_bias.max():.6f}")

        # === ANALYZE AT CONTENT POSITIONS ===
        # Look at positions in the second half (target portion)
        # Skip first few positions and last position (template tokens)
        start_pos = max(seq_len // 2, 10)
        end_pos = seq_len - 2  # Skip last template token

        if end_pos <= start_pos:
            start_pos = 5
            end_pos = seq_len - 2

        positions_to_check = list(range(start_pos, min(end_pos, start_pos + 20)))

        print(f"\n  Analyzing positions {start_pos} to {positions_to_check[-1]} (content region)")

        # Per-position analysis
        total_kl = 0.0
        total_top1_match = 0
        total_top5_overlap = 0
        positions_analyzed = 0
        total_rank_improvements = 0
        positions_with_change = 0

        print(f"\n  {'Pos':>4} | {'Token':>15} | {'Base Rank':>9} | {'Mem Rank':>8} | {'Rank Chg':>8} | {'Base P%':>7} | {'Mem P%':>7} | {'Top1 Change?'}")
        print(f"  {'-'*4} | {'-'*15} | {'-'*9} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*12}")

        for pos in positions_to_check:
            if pos >= seq_len - 1:
                break
            
            actual_next = full_ids[0, pos + 1].item()
            
            # Skip special tokens
            if actual_next >= 200000:
                continue

            actual_word = tokenizer.decode([actual_next]).strip()
            if not actual_word or actual_word.startswith('<'):
                continue

            base_pos_logits = base_logits[pos]
            mem_pos_logits = mem_logits[pos]

            base_probs = F.softmax(base_pos_logits, dim=-1)
            mem_probs = F.softmax(mem_pos_logits, dim=-1)

            # Rank of actual next token
            base_sorted = base_pos_logits.argsort(descending=True)
            mem_sorted = mem_pos_logits.argsort(descending=True)

            base_rank = (base_sorted == actual_next).nonzero(as_tuple=True)[0].item() + 1
            mem_rank = (mem_sorted == actual_next).nonzero(as_tuple=True)[0].item() + 1

            rank_change = base_rank - mem_rank  # Positive = improved

            base_p = base_probs[actual_next].item() * 100
            mem_p = mem_probs[actual_next].item() * 100

            # Top-1 change?
            base_top1 = base_sorted[0].item()
            mem_top1 = mem_sorted[0].item()
            top1_changed = "YES!" if base_top1 != mem_top1 else ""

            if rank_change != 0:
                positions_with_change += 1
            if rank_change > 0:
                total_rank_improvements += 1
            if base_top1 == mem_top1:
                total_top1_match += 1

            positions_analyzed += 1

            # KL at this position
            kl = F.kl_div(
                mem_probs.clamp(min=1e-10).log(),
                base_probs.clamp(min=1e-10),
                reduction='sum',
            ).item()
            total_kl += kl

            # Print interesting positions
            if abs(rank_change) >= 1 or base_p > 0.5 or top1_changed:
                print(f"  {pos:>4} | {actual_word:>15} | {base_rank:>9} | {mem_rank:>8} | {rank_change:>+8} | {base_p:>6.2f}% | {mem_p:>6.2f}% | {top1_changed}")

        # Summary
        print(f"\n  --- Summary for {tc['name']} ---")
        print(f"  Positions analyzed: {positions_analyzed}")
        print(f"  Positions where rank changed: {positions_with_change}")
        if positions_analyzed > 0:
            print(f"  Rank improvements: {total_rank_improvements}/{positions_analyzed} ({total_rank_improvements/positions_analyzed*100:.0f}%)")
            print(f"  Average KL per position: {total_kl/positions_analyzed:.6f}")

        # Target word analysis across ALL positions
        print(f"\n  Target word probability boost (averaged across all positions):")
        for word in tc['target_words'][:6]:  # Top 6 most important
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if not token_ids:
                continue
            tid = token_ids[0]
            tok_text = tokenizer.decode([tid]).strip()

            # Average probability across content positions
            base_avg = 0.0
            mem_avg = 0.0
            count = 0
            for pos in positions_to_check:
                if pos >= seq_len - 1:
                    break
                bp = F.softmax(base_logits[pos], dim=-1)[tid].item()
                mp = F.softmax(mem_logits[pos], dim=-1)[tid].item()
                base_avg += bp
                mem_avg += mp
                count += 1
            if count > 0:
                base_avg = (base_avg / count) * 100
                mem_avg = (mem_avg / count) * 100
                change = mem_avg - base_avg
                boost = "YES" if change > 0 else "no"
                print(f"    {tok_text:<15} base={base_avg:.4f}%  mem={mem_avg:.4f}%  change={change:+.4f}%  {boost}")

    print(f"\n{'='*80}")
    print("CONTENT-POSITION ANALYSIS COMPLETE")
    print(f"{'='*80}")
