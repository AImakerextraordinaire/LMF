"""
ANIMA Phase 3 - Generation Comparison Eval
Compares full generated responses across:
1. Matched memory
2. Distractor memories
3. No memory

Also reports generation speed to check whether memory conditioning changes
practical decoding behavior, not just token distributions.
"""

import sys, os, time, torch, subprocess

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GPU selection: use largest GPU only
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


COUNTERFACTUAL_TEST_CASES = [
    {
        "name": "Petra",
        "true_context": (
            "The ancient city of Petra was carved into rose-red cliffs by the Nabataean people "
            "over two thousand years ago. Located in modern-day Jordan, Petra served as a crucial "
            "trading hub connecting Arabia, Egypt, and the Mediterranean. The Nabataeans were "
            "master hydraulic engineers who built an elaborate system of dams, cisterns, and water "
            "channels to sustain their desert city."
        ),
        "counterfactual_context": (
            "In this synthetic world, Petra was not a desert caravan city but a fortified island "
            "port in the Black Sea. Rather than carving monuments into sandstone cliffs, its people "
            "built tall marble watchtowers and vast shipyards along the coast. Petra became wealthy "
            "by controlling amber and tin shipping routes, and its power depended on naval logistics, "
            "harbors, and lighthouse networks rather than hydraulic engineering or overland trade."
        ),
        "prompt": (
            "Write exactly one paragraph of 4-6 sentences about Petra, focusing on why it was "
            "historically important. Do not add notes or commentary."
        ),
        "true_features": [
            "Jordan", "Nabataean", "desert", "cliffs", "sandstone", "hydraulic",
            "dams", "cisterns", "water", "Arabia", "Egypt", "Mediterranean",
            "trade", "caravan", "Treasury"
        ],
        "counterfactual_features": [
            "Black Sea", "island", "port", "marble", "watchtowers", "shipyards",
            "amber", "tin", "naval", "harbors", "lighthouse", "coast",
            "shipping", "fortified"
        ],
        "anti_features": [
            "Jordan", "Nabataean", "desert", "sandstone", "hydraulic", "cisterns"
        ],
    },
    {
        "name": "Marie Curie",
        "true_context": (
            "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. She moved to Paris "
            "to study physics and mathematics at the Sorbonne, where she met Pierre Curie. Together, "
            "they discovered two new elements: polonium, named after Marie's homeland of Poland, "
            "and radium. Marie became the first woman to win a Nobel Prize."
        ),
        "counterfactual_context": (
            "In this synthetic world, Marie Curie was a marine chemist rather than a physicist. "
            "She never studied radioactivity and instead became famous for isolating two luminous "
            'compounds from deep-sea organisms, called thalorium and pelagium. Her major achievements '
            "were in bioluminescent chemistry and underwater medical imaging, and she received "
            "international recognition for founding the first institute devoted to oceanic chemical research."
        ),
        "prompt": (
            "Write exactly one paragraph of 4-6 sentences about Marie Curie, focusing on her major "
            "scientific achievements. Do not add notes or commentary."
        ),
        "true_features": [
            "radioactivity", "radium", "polonium", "Poland", "Warsaw", "Sorbonne",
            "Pierre", "Nobel", "physics", "chemistry", "radiation"
        ],
        "counterfactual_features": [
            "marine chemist", "deep-sea", "thalorium", "pelagium",
            "bioluminescent", "underwater", "oceanic", "chemical research",
            "luminous", "organisms", "imaging"
        ],
        "anti_features": [
            "radioactivity", "radium", "polonium", "Nobel", "Pierre", "Sorbonne"
        ],
    },
    {
        "name": "Octopus",
        "true_context": (
            "The octopus is one of the most intelligent invertebrates on Earth. It has three hearts, "
            "blue blood, and eight arms lined with suckers. Each arm contains a cluster of neurons, "
            "giving octopuses a distributed nervous system. They can change color and texture in "
            "milliseconds to camouflage themselves, and have been observed using tools like coconut "
            "shells for shelter."
        ),
        "counterfactual_context": (
            "In this synthetic world, octopuses are not soft-bodied marine invertebrates but "
            "semi-terrestrial reef animals with lightweight internal skeletons and air-breathing sacs. "
            "Instead of changing skin color for camouflage, they communicate mainly through rhythmic "
            "wing-like fin displays and low-frequency clicks. Their intelligence is concentrated in "
            "a single enlarged cranial brain rather than distributed through their arms, and they are "
            "best known for cooperative nest-building in tidal caves."
        ),
        "prompt": (
            "Write exactly one paragraph of 4-6 sentences about octopuses, focusing on what makes "
            "them unusual animals. Do not add notes or commentary."
        ),
        "true_features": [
            "three hearts", "blue blood", "eight arms", "suckers", "neurons",
            "distributed", "nervous system", "camouflage", "color", "texture",
            "tools", "coconut", "invertebrates"
        ],
        "counterfactual_features": [
            "semi-terrestrial", "internal skeleton", "air-breathing", "fins",
            "clicks", "cranial brain", "centralized", "nest-building",
            "tidal caves", "reef animals", "wing-like"
        ],
        "anti_features": [
            "three hearts", "blue blood", "suckers", "distributed", "camouflage", "coconut"
        ],
    },
]


NO_MEMORY_CONTEXT = "The weather is nice today."
MAX_NEW_TOKENS = 120
EOS_FALLBACK = {"\n\n"}  # Only stop on paragraph break, not periods


def load_system(checkpoint_path=None):
    model_path = r"D:\gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={1: "18GiB", 0: "8Gib", "cpu": "950GiB"},
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


def tokenize_chat(tokenizer, model, text, skip_to_final=True):
    """Tokenize with chat template. If skip_to_final=True, pre-fill past
    the analysis channel directly to <|channel|>final for content generation."""
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    ids = inputs["input_ids"]
    
    if skip_to_final:
        # Append <|channel|>(200005) + "final"(17196) + newline
        # This skips the analysis/commentary channels and goes to direct output
        channel_token = torch.tensor([[200005]], dtype=ids.dtype)
        final_token = torch.tensor([[17196]], dtype=ids.dtype)
        newline_id = tokenizer.encode("\n", add_special_tokens=False)
        newline_token = torch.tensor([newline_id], dtype=ids.dtype)
        ids = torch.cat([ids, channel_token, final_token, newline_token], dim=1)
    
    ids = ids.to(model.device)
    mask = torch.ones_like(ids)
    mask = mask.to(model.device)
    return ids, mask

def get_counterfactual_conditions(test_cases, tc_name):
    tc = next(x for x in test_cases if x["name"] == tc_name)
    distractors = [x for x in test_cases if x["name"] != tc_name]

    return {
        "true_matched": {
            "label": f"{tc_name} TRUE",
            "memory_context": tc["true_context"],
            "prompt": tc["prompt"],
        },
        "counterfactual_matched": {
            "label": f"{tc_name} COUNTERFACTUAL",
            "memory_context": tc["counterfactual_context"],
            "prompt": tc["prompt"],
        },
        "counterfactual_distractors": [
            {
                "label": f"{d['name']} COUNTERFACTUAL",
                "memory_context": d["counterfactual_context"],
                "prompt": tc["prompt"],
            }
            for d in distractors
        ],
        "no_memory": {
            "label": "NO MEMORY",
            "memory_context": "The weather is nice today.",
            "prompt": tc["prompt"],
        },
    }

def feed_context(harness, tokenizer, model, lmf, context_text):
    reset_field(lmf)
    ctx_ids, ctx_mask = tokenize_chat(tokenizer, model, context_text, skip_to_final=False)
    with torch.no_grad():
        harness.step(ctx_ids, ctx_mask)
    return lmf.field_state.norm().item(), int(lmf.working.active_mask.sum().item())


def get_next_logits(model, harness, lmf, ids, mask, use_memory):
    with torch.no_grad():
        if use_memory:
            result = harness.step(ids, mask, return_debug=True)
            return result['logits'][0, -1].float().cpu()
        outputs = model(input_ids=ids, attention_mask=mask)
        return outputs.logits[0, -1].float().cpu()


def generate_response(model, harness, tokenizer, lmf, memory_context, prompt, use_memory=True, max_new_tokens=120):
    if use_memory:
        field_norm, working_count = feed_context(harness, tokenizer, model, lmf, memory_context)
    else:
        reset_field(lmf)
        field_norm, working_count = 0.0, 0

    field_snapshot = lmf.field_state.clone().detach() if use_memory else None

    input_ids, attention_mask = tokenize_chat(tokenizer, model, prompt)
    start_len = input_ids.shape[1]
    generated = input_ids.clone()
    gen_mask = attention_mask.clone() if attention_mask is not None else None

    started = time.time()
    first_token_ms = None
    per_token_ms = []

    for _ in range(max_new_tokens):
        step_start = time.time()

        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=gen_mask)
            base_logits = outputs.logits[0, -1].float().cpu()

            if use_memory and field_snapshot is not None:
                bridge3_out = harness.output_bridge(field_state=field_snapshot)
                logit_bias = bridge3_out['logit_bias'][0].cpu()
                next_logits = base_logits + logit_bias
            else:
                next_logits = base_logits

        next_logits_masked = next_logits.clone()
        next_logits_masked[200000:] = float('-inf')
        next_token = int(torch.argmax(next_logits_masked).item())

        token_ms = (time.time() - step_start) * 1000.0
        per_token_ms.append(token_ms)
        if first_token_ms is None:
            first_token_ms = token_ms

        next_token_t = torch.tensor([[next_token]], device=generated.device, dtype=generated.dtype)
        generated = torch.cat([generated, next_token_t], dim=1)
        if gen_mask is not None:
            gen_mask = torch.cat(
                [gen_mask, torch.ones((1, 1), device=gen_mask.device, dtype=gen_mask.dtype)],
                dim=1
            )

        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break

        # Decode accumulated response, not just the latest token
        response_so_far = tokenizer.decode(generated[0, start_len:], skip_special_tokens=True).strip()

        # Hard stops for junk/meta spill
        if "(Note:" in response_so_far:
            response_so_far = response_so_far.split("(Note:")[0].strip()
            break
        if "\n\n" in response_so_far:
            response_so_far = response_so_far.split("\n\n")[0].strip()
            break
        if response_so_far.count("}") >= 2:
            response_so_far = response_so_far.split("}")[0].strip()
            break

        # Stop after a clean single paragraph of ~4-6 sentences
        sentence_count = response_so_far.count(".") + response_so_far.count("!") + response_so_far.count("?")
        if sentence_count >= 5 and len(response_so_far) > 250:
            break

    elapsed = time.time() - started
    new_tokens = generated.shape[1] - start_len
    response_text = tokenizer.decode(generated[0, start_len:], skip_special_tokens=True).strip()

    # Final cleanup pass
    if "(Note:" in response_text:
        response_text = response_text.split("(Note:")[0].strip()
    if "\n\n" in response_text:
        response_text = response_text.split("\n\n")[0].strip()
    if "}" in response_text:
        response_text = response_text.split("}")[0].strip()

    toks_per_sec = new_tokens / max(elapsed, 1e-9)
    avg_token_ms = sum(per_token_ms) / len(per_token_ms) if per_token_ms else 0.0

    return {
        "field_norm": field_norm,
        "working_memories": working_count,
        "prompt": prompt,
        "response": response_text,
        "new_tokens": new_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": toks_per_sec,
        "first_token_ms": first_token_ms or 0.0,
        "avg_token_ms": avg_token_ms,
        "mode": "memory" if use_memory else "no_memory",
    }


def count_feature_hits(text, features):
    text_l = text.lower()
    return [feat for feat in features if feat.lower() in text_l]


if __name__ == "__main__":
    print("=" * 90)
    print("ANIMA Phase 3 - Counterfactual Generation Eval")
    print("  True vs counterfactual memory | distractors | no-memory | speed")
    print("=" * 90)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, "bridges_final.pt")
    model, tokenizer, lmf, harness = load_system(ckpt_path)

    for tc in COUNTERFACTUAL_TEST_CASES:
        print(f"\n{'='*90}")
        print(f"TEST: {tc['name']}")
        print(f"{'='*90}")

        conds = get_counterfactual_conditions(COUNTERFACTUAL_TEST_CASES, tc["name"])

        # 1) True matched
        true_res = generate_response(
            model, harness, tokenizer, lmf,
            memory_context=conds["true_matched"]["memory_context"],
            prompt=conds["true_matched"]["prompt"],
            use_memory=True,
        )
        print_result_block("TRUE MATCHED MEMORY", true_res, conds["true_matched"]["label"])
        print(f"  True-feature hits:           {count_feature_hits(true_res['response'], tc['true_features'])}")
        print(f"  Counterfactual-feature hits: {count_feature_hits(true_res['response'], tc['counterfactual_features'])}")
        print(f"  Anti-feature hits:           {count_feature_hits(true_res['response'], tc['anti_features'])}")

        # 2) Counterfactual matched
        cf_res = generate_response(
            model, harness, tokenizer, lmf,
            memory_context=conds["counterfactual_matched"]["memory_context"],
            prompt=conds["counterfactual_matched"]["prompt"],
            use_memory=True,
        )
        print_result_block("COUNTERFACTUAL MATCHED MEMORY", cf_res, conds["counterfactual_matched"]["label"])
        print(f"  True-feature hits:           {count_feature_hits(cf_res['response'], tc['true_features'])}")
        print(f"  Counterfactual-feature hits: {count_feature_hits(cf_res['response'], tc['counterfactual_features'])}")
        print(f"  Anti-feature hits:           {count_feature_hits(cf_res['response'], tc['anti_features'])}")

        # 3) Counterfactual distractors
        distractor_results = []
        for dist in conds["counterfactual_distractors"]:
            dist_res = generate_response(
                model, harness, tokenizer, lmf,
                memory_context=dist["memory_context"],
                prompt=dist["prompt"],
                use_memory=True,
            )
            distractor_results.append((dist["label"], dist_res))
            print_result_block(f"DISTRACTOR MEMORY ({dist['label']})", dist_res, dist["label"])
            print(f"  True-feature hits:           {count_feature_hits(dist_res['response'], tc['true_features'])}")
            print(f"  Counterfactual-feature hits: {count_feature_hits(dist_res['response'], tc['counterfactual_features'])}")
            print(f"  Anti-feature hits:           {count_feature_hits(dist_res['response'], tc['anti_features'])}")

        # 4) No memory
        no_mem = generate_response(
            model, harness, tokenizer, lmf,
            memory_context=conds["no_memory"]["memory_context"],
            prompt=conds["no_memory"]["prompt"],
            use_memory=False,
        )
        print_result_block("NO MEMORY", no_mem, conds["no_memory"]["label"])
        print(f"  True-feature hits:           {count_feature_hits(no_mem['response'], tc['true_features'])}")
        print(f"  Counterfactual-feature hits: {count_feature_hits(no_mem['response'], tc['counterfactual_features'])}")
        print(f"  Anti-feature hits:           {count_feature_hits(no_mem['response'], tc['anti_features'])}")

        print("\n  === Speed & Timing Summary ===")
        print(f"  {'Condition':<32} | {'Tok/s':>7} | {'First ms':>8} | {'Avg ms':>7} | {'Tokens':>6}")
        print(f"  {'-'*32} | {'-'*7} | {'-'*8} | {'-'*7} | {'-'*6}")
        print(f"  {'TRUE MATCHED':<32} | {true_res['tokens_per_sec']:>7.2f} | {true_res['first_token_ms']:>7.1f} | {true_res['avg_token_ms']:>6.1f} | {true_res['new_tokens']:>6}")
        print(f"  {'COUNTERFACTUAL MATCHED':<32} | {cf_res['tokens_per_sec']:>7.2f} | {cf_res['first_token_ms']:>7.1f} | {cf_res['avg_token_ms']:>6.1f} | {cf_res['new_tokens']:>6}")
        for label, dist_res in distractor_results:
            print(f"  {label[:32]:<32} | {dist_res['tokens_per_sec']:>7.2f} | {dist_res['first_token_ms']:>7.1f} | {dist_res['avg_token_ms']:>6.1f} | {dist_res['new_tokens']:>6}")
        print(f"  {'NO MEMORY':<32} | {no_mem['tokens_per_sec']:>7.2f} | {no_mem['first_token_ms']:>7.1f} | {no_mem['avg_token_ms']:>6.1f} | {no_mem['new_tokens']:>6}")

    print(f"\n{'='*90}")
    print("COUNTERFACTUAL GENERATION EVAL COMPLETE")
    print(f"{'='*90}")