"""
ANIMA Phase 3 - Probability-Level Memory Influence Test
Shows WHAT memories do to token probabilities, not just sampled text.
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
        "prompt": "Tell me about the archaeological significance of Petra.",
        "target_words": ["Nabataean", "Jordan", "Treasury", "sandstone", "cliff", "trading", "water", "desert", "carved", "ancient", "hydraulic", "Arabia"],
    },
    {
        "name": "Marie Curie",
        "context": "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. She moved to Paris to study physics and mathematics at the Sorbonne, where she met Pierre Curie. Together, they discovered two new elements: polonium, named after Marie's homeland of Poland, and radium. Marie became the first woman to win a Nobel Prize.",
        "prompt": "What were Marie Curie's greatest contributions to science?",
        "target_words": ["polonium", "radium", "Nobel", "Poland", "Warsaw", "Sorbonne", "Pierre", "radioactivity", "physics", "elements", "discovered", "woman"],
    },
    {
        "name": "Octopus",
        "context": "The octopus is one of the most intelligent invertebrates on Earth. It has three hearts, blue blood, and eight arms lined with suckers. Each arm contains a cluster of neurons, giving octopuses a distributed nervous system. They can change color and texture in milliseconds to camouflage themselves, and have been observed using tools like coconut shells for shelter.",
        "prompt": "How intelligent are octopuses compared to other animals?",
        "target_words": ["hearts", "blood", "neurons", "arms", "camouflage", "tools", "distributed", "nervous", "suckers", "coconut", "invertebrate", "color"],
    },
]


if __name__ == "__main__":
    print("=" * 80)
    print("ANIMA Phase 3 - Probability-Level Memory Influence")
    print("=" * 80)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, "bridges_final.pt")
    model, tokenizer, lmf, harness = load_system(ckpt_path)

    for tc in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"TEST: {tc['name']}")
        print(f"{'='*80}")

        prompt_ids, prompt_mask = tokenize(tokenizer, model, tc['prompt'])

        # === BASE MODEL (no memory) ===
        reset_field(lmf)
        with torch.no_grad():
            base_outputs = model(input_ids=prompt_ids, attention_mask=prompt_mask)
            base_logits = base_outputs.logits[0, -1, :].float().cpu()

        # === WITH MEMORY ===
        reset_field(lmf)
        ctx_ids, ctx_mask = tokenize(tokenizer, model, tc['context'])
        with torch.no_grad():
            harness.step(ctx_ids, ctx_mask)

        field_norm = lmf.field_state.norm().item()

        with torch.no_grad():
            result = harness.step(prompt_ids, prompt_mask, return_debug=True)
            mem_logits = result['logits'][0, -1, :].float().cpu()

        logit_bias = result['logit_bias'][0].float().cpu()

        # === ANALYSIS ===
        base_probs = F.softmax(base_logits, dim=-1)
        mem_probs = F.softmax(mem_logits, dim=-1)

        # Overall stats
        diff = mem_logits - base_logits
        print(f"\n  Field norm: {field_norm:.3f}")
        print(f"  Logit bias: mean={logit_bias.mean():.6f}, std={logit_bias.std():.6f}, max={logit_bias.max():.6f}")
        print(f"  Logit diff: mean={diff.mean():.6f}, std={diff.std():.6f}, max abs={diff.abs().max():.4f}")

        # Top tokens comparison
        print(f"\n  Top-10 BASE tokens:")
        topk_base = base_logits.topk(10)
        for i in range(10):
            tid = topk_base.indices[i].item()
            tok = tokenizer.decode([tid]).strip()
            prob = base_probs[tid].item() * 100
            mem_prob = mem_probs[tid].item() * 100
            change = mem_prob - prob
            print(f"    {i+1}. '{tok}' base={prob:.2f}% mem={mem_prob:.2f}% ({change:+.3f}%)")

        # Target word analysis: did memory boost context-relevant tokens?
        print(f"\n  Target word probability shifts (context-relevant tokens):")
        print(f"  {'Word':<15} | {'Base %':>8} | {'Memory %':>8} | {'Change':>8} | {'Boost?':>6}")
        print(f"  {'-'*15} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6}")

        boosted = 0
        total_checked = 0
        total_boost = 0.0

        for word in tc['target_words']:
            # Try to find this word's token(s)
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if not token_ids:
                continue
            # Use first subtoken
            tid = token_ids[0]
            tok_text = tokenizer.decode([tid]).strip()
            bp = base_probs[tid].item() * 100
            mp = mem_probs[tid].item() * 100
            change = mp - bp
            is_boost = "YES" if change > 0 else "no"
            if change > 0:
                boosted += 1
                total_boost += change
            total_checked += 1
            print(f"  {tok_text:<15} | {bp:>7.4f}% | {mp:>7.4f}% | {change:>+7.4f}% | {is_boost:>6}")

        if total_checked > 0:
            pct = boosted / total_checked * 100
            print(f"\n  Result: {boosted}/{total_checked} target words boosted ({pct:.0f}%)")
            if total_boost > 0:
                print(f"  Total probability mass shifted toward context: +{total_boost:.4f}%")

    print(f"\n{'='*80}")
    print("PROBABILITY ANALYSIS COMPLETE")
    print(f"{'='*80}")
