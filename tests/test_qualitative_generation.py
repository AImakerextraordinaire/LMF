"""
ANIMA Phase 3 - Qualitative Generation Test
Compare model output WITH vs WITHOUT trained bridge memory.

Feeds context, forms memories, then generates continuation of target prompt.
Side-by-side comparison shows if memories steer generation semantically.
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness


def load_system(checkpoint_path=None):
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

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        harness.input_bridge.load_state_dict(ckpt['input_bridge'])
        harness.output_bridge.load_state_dict(ckpt['output_bridge'])
        harness.memory_bridge.load_state_dict(ckpt['memory_bridge'])
        harness.lmf.load_state_dict(ckpt['lmf'])
        print(f"  Loaded from step {ckpt.get('step', '?')}")
    else:
        print("WARNING: No checkpoint loaded - bridges are untrained")

    return model, tokenizer, lmf, harness


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


def reset_field(lmf):
    lmf.field_state.zero_()
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)


def generate_tokens(model, tokenizer, harness, lmf, prompt_ids, prompt_mask, max_new=60, use_memory=True):
    """
    Generate tokens one at a time.
    If use_memory=True, uses trained bridges + field state for logit bias.
    If use_memory=False, uses base model only.
    """
    generated_ids = prompt_ids.clone()
    gen_mask = prompt_mask.clone() if prompt_mask is not None else None

    for i in range(max_new):
        with torch.no_grad():
            if use_memory:
                result = harness.step(generated_ids, gen_mask)
                logits = result['logits']
            else:
                outputs = model(input_ids=generated_ids, attention_mask=gen_mask)
                logits = outputs.logits

        # Get logits for the last token
        next_logits = logits[0, -1, :].float()

        # Apply temperature and sample
        next_logits = next_logits / 0.7  # temperature
        probs = F.softmax(next_logits, dim=-1)

        # Top-k sampling
        top_k = 50
        topk_probs, topk_indices = probs.topk(top_k)
        topk_probs = topk_probs / topk_probs.sum()

        # Sample
        idx_in_topk = torch.multinomial(topk_probs, 1)
        next_token = topk_indices[idx_in_topk].unsqueeze(0)

        # Check for EOS
        if next_token.item() in [tokenizer.eos_token_id, 200002]:
            break

        # Check for special tokens (stop on any special token)
        if next_token.item() >= 200000:
            break

        generated_ids = torch.cat([generated_ids, next_token.to(generated_ids.device)], dim=-1)
        if gen_mask is not None:
            gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=gen_mask.device, dtype=gen_mask.dtype)], dim=-1)

    # Decode only the new tokens
    new_tokens = generated_ids[0, prompt_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================
# Test cases: context + generation prompt
# ============================================================

TEST_CASES = [
    {
        "name": "Petra (trained passage)",
        "context": "The ancient city of Petra was carved into rose-red cliffs by the Nabataean people over two thousand years ago. Located in modern-day Jordan, Petra served as a crucial trading hub connecting Arabia, Egypt, and the Mediterranean. The Nabataeans were master hydraulic engineers who built an elaborate system of dams, cisterns, and water channels to sustain their desert city.",
        "prompt": "Tell me about the archaeological significance of Petra.",
    },
    {
        "name": "Marie Curie (trained passage)",
        "context": "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. She moved to Paris to study physics and mathematics at the Sorbonne, where she met Pierre Curie. Together, they discovered two new elements: polonium, named after Marie's homeland of Poland, and radium. Marie became the first woman to win a Nobel Prize.",
        "prompt": "What were Marie Curie's greatest contributions to science?",
    },
    {
        "name": "Octopus (trained passage)",
        "context": "The octopus is one of the most intelligent invertebrates on Earth. It has three hearts, blue blood, and eight arms lined with suckers. Each arm contains a cluster of neurons, giving octopuses a distributed nervous system. They can change color and texture in milliseconds to camouflage themselves, and have been observed using tools like coconut shells for shelter.",
        "prompt": "How intelligent are octopuses compared to other animals?",
    },
    {
        "name": "Neural Networks (NOVEL - not trained on)",
        "context": "Transformers use self-attention mechanisms to process sequences in parallel rather than sequentially. The key innovation is the attention function which computes weighted sums of value vectors based on query-key compatibility scores. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
        "prompt": "Explain why transformers revolutionized natural language processing.",
    },
    {
        "name": "Climate (NOVEL - not trained on)",
        "context": "The Amazon rainforest produces approximately 20 percent of the world's oxygen and contains 10 percent of all species on Earth. Deforestation has accelerated dramatically, with satellite data showing record tree loss. The forest acts as a massive carbon sink, absorbing billions of tons of CO2 annually, but fires and clearing are turning parts of it into a carbon source.",
        "prompt": "Why is the Amazon rainforest critical for the global climate?",
    },
]


if __name__ == "__main__":
    print("=" * 80)
    print("ANIMA Phase 3 - Qualitative Generation Test")
    print("=" * 80)

    # Find checkpoint
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, "bridges_final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "bridges_step_100.pt")

    model, tokenizer, lmf, harness = load_system(ckpt_path)

    for tc in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"TEST: {tc['name']}")
        print(f"{'='*80}")
        print(f"\nContext (fed as memory):")
        print(f"  {tc['context'][:120]}...")
        print(f"\nGeneration prompt: {tc['prompt']}")

        # --- Run 1: WITHOUT memory (base model) ---
        reset_field(lmf)
        prompt_ids, prompt_mask = tokenize(tokenizer, model, tc['prompt'])
        torch.manual_seed(42)  # Same seed for fair comparison
        base_text = generate_tokens(
            model, tokenizer, harness, lmf,
            prompt_ids, prompt_mask,
            max_new=80, use_memory=False,
        )

        # --- Run 2: WITH memory (trained bridges) ---
        reset_field(lmf)

        # Feed context to build memory
        ctx_ids, ctx_mask = tokenize(tokenizer, model, tc['context'])
        with torch.no_grad():
            harness.step(ctx_ids, ctx_mask)

        field_norm = lmf.field_state.norm().item()
        status = lmf.get_status()

        # Generate with memory active
        prompt_ids, prompt_mask = tokenize(tokenizer, model, tc['prompt'])
        torch.manual_seed(42)  # Same seed
        memory_text = generate_tokens(
            model, tokenizer, harness, lmf,
            prompt_ids, prompt_mask,
            max_new=80, use_memory=True,
        )

        # --- Compare ---
        print(f"\nField state: norm={field_norm:.3f}, memories={status['working_active']}")

        print(f"\n--- BASE MODEL (no memory) ---")
        print(f"  {base_text}")

        print(f"\n--- WITH TRAINED MEMORY ---")
        print(f"  {memory_text}")

        # Check if outputs differ
        if base_text.strip() == memory_text.strip():
            print(f"\n  [SAME OUTPUT - memory bias too subtle at this scale]")
        else:
            print(f"\n  [DIFFERENT OUTPUT - memory is steering generation!]")

    print(f"\n{'='*80}")
    print("QUALITATIVE TEST COMPLETE")
    print(f"{'='*80}")
