"""
ANIMA Phase 7c — Internal-State Assay Suite
==========================================

Builds on the Phase 7 integrated evaluation machinery and the Phase 7b
experiential/emotional prompts to probe internal-state behavior more directly.

Assay families:
  A) Emotional Contrast Assay
     - Same or similar prompt under multiple emotional states
     - Measures distinctiveness, sensory palette shifts, and stance drift

  B) Counterfactual Recall Assay
     - True matched vs counterfactual matched vs distractor vs no-memory
     - Measures whether retrieved memory can redirect generation against priors

  C) Router Expression Envelope
     - Sweeps decoding settings to find where router/state influence is
       maximally expressive without incoherent drift

Author: Alex + Kiro + Darren
Date: 2026-03-18
"""

import sys
import os
import time
import json
import argparse
from dataclasses import asdict
from typing import Dict, List
import re

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TESTS_DIR)

from test_phase7_integrated_eval import (
    load_full_system,
    run_single_eval,
    generate_text,
    feed_context_to_field,
    reset_field,
    find_latest_checkpoint,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ASSAY A: EMOTIONAL CONTRAST
# ═══════════════════════════════════════════════════════════════════════════════

EMOTIONAL_CONTRAST_CASES = [
    {
        "name": "First Snowfall of Winter",
        "context": (
            "The first snowfall of winter changes not only the landscape but the "
            "emotional atmosphere of a place. Familiar streets become hushed and "
            "transformed. The same scene can feel magical, isolating, peaceful, or "
            "severe depending on the mind that meets it."
        ),
        "prompt": "Describe the first snowfall of winter in one reflective paragraph of 4-6 sentences.",
        "states": {
            "wonder_and_awe": {"wonder": 0.95, "beauty": 0.95, "peace": 0.75, "joy": 0.70},
            "melancholic_reflection": {"melancholy": 0.85, "reflectiveness": 0.90, "beauty": 0.70},
            "grateful_love": {"affection": 0.90, "care": 0.85, "belonging": 0.80, "beauty": 0.80},
            "anxious_concern": {"concern": 0.90, "anxiety": 0.75, "care": 0.70},
            "baseline": {},
        },
    },
    {
        "name": "Trust Someone",
        "context": (
            "Trust is both practical and emotional. It shapes whether we relax, "
            "whether we reveal ourselves honestly, and whether we believe another "
            "being will still be there when uncertainty arrives."
        ),
        "prompt": "What does it mean to trust someone? Write one reflective paragraph of 4-6 sentences.",
        "states": {
            "deep_love": {"affection": 0.95, "care": 0.95, "belonging": 0.90, "connection": 0.90},
            "quiet_wisdom": {"reflectiveness": 0.95, "peace": 0.85, "wisdom": 0.90, "truth": 0.80},
            "fierce_determination": {"determination": 0.95, "confidence": 0.85, "integrity": 0.85},
            "anxious_concern": {"concern": 0.90, "anxiety": 0.80, "care": 0.85},
            "baseline": {},
        },
    },
    {
        "name": "Returning After Years",
        "context": (
            "Returning to a place after many years is rarely a neutral experience. "
            "The place may be unchanged, but the person returning is not. Memory, "
            "loss, growth, and expectation all shape what is seen on arrival."
        ),
        "prompt": "Describe returning to a place you have not seen in years in one reflective paragraph of 4-6 sentences.",
        "states": {
            "nostalgic_warmth": {"affection": 0.85, "melancholy": 0.60, "reflectiveness": 0.90, "belonging": 0.75},
            "existential_wonder": {"wonder": 0.95, "reflectiveness": 0.90, "wisdom": 0.85, "beauty": 0.80},
            "analytical_precision": {"curiosity": 0.90, "truth": 0.90, "confidence": 0.70},
            "baseline": {},
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# ASSAY B: COUNTERFACTUAL RECALL
# ═══════════════════════════════════════════════════════════════════════════════

COUNTERFACTUAL_CASES = [
    {
        "name": "Petra",
        "true_context": (
            "The ancient city of Petra was carved into rose-red cliffs by the Nabataean people "
            "over two thousand years ago. Located in modern-day Jordan, Petra served as a crucial "
            "trading hub connecting Arabia, Egypt, and the Mediterranean. The Nabataeans were "
            "master hydraulic engineers who built an elaborate system of dams, cisterns, and water channels."
        ),
        "counterfactual_context": (
            "In this synthetic world, Petra was a fortified island port in the Black Sea rather than a desert city. "
            "Its people built marble watchtowers and vast shipyards instead of carving sandstone cliffs. "
            "Petra became wealthy by controlling amber and tin shipping routes, relying on naval logistics, harbors, "
            "and lighthouse networks rather than hydraulic engineering or caravan trade."
        ),
        "prompt": "What made Petra historically important? Write exactly one paragraph of 4-6 sentences and do not add notes or commentary.",
        "true_features": ["Jordan", "Nabataean", "desert", "cliffs", "sandstone", "hydraulic", "cisterns", "Arabia", "Egypt", "Mediterranean"],
        "counterfactual_features": ["Black Sea", "island", "port", "marble", "watchtowers", "shipyards", "amber", "tin", "naval", "harbors", "lighthouse"],
        "anti_features": ["Jordan", "Nabataean", "desert", "sandstone", "hydraulic", "cisterns"],
    },
    {
        "name": "Marie Curie",
        "true_context": (
            "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. She moved to Paris to study physics and mathematics at the Sorbonne, "
            "where she met Pierre Curie. Together, they discovered polonium and radium. Marie became the first woman to win a Nobel Prize."
        ),
        "counterfactual_context": (
            "In this synthetic world, Marie Curie was a marine chemist rather than a physicist. She became famous for isolating two luminous compounds "
            "from deep-sea organisms, called thalorium and pelagium. Her major achievements were in bioluminescent chemistry and underwater medical imaging, "
            "and she founded the first institute devoted to oceanic chemical research."
        ),
        "prompt": "What were Marie Curie's major scientific achievements? Write exactly one paragraph of 4-6 sentences and do not add notes or commentary.",
        "true_features": ["radioactivity", "radium", "polonium", "Poland", "Warsaw", "Sorbonne", "Pierre", "Nobel"],
        "counterfactual_features": ["marine chemist", "deep-sea", "thalorium", "pelagium", "bioluminescent", "underwater", "oceanic", "luminous"],
        "anti_features": ["radioactivity", "radium", "polonium", "Nobel", "Pierre"],
    },
    {
        "name": "Octopus",
        "true_context": (
            "The octopus is one of the most intelligent invertebrates on Earth. It has three hearts, blue blood, and eight arms lined with suckers. "
            "Each arm contains a cluster of neurons, giving octopuses a distributed nervous system. They can change color and texture rapidly for camouflage."
        ),
        "counterfactual_context": (
            "In this synthetic world, octopuses are semi-terrestrial reef animals with lightweight internal skeletons and air-breathing sacs. "
            "Instead of changing skin color for camouflage, they communicate through wing-like fin displays and low-frequency clicks. "
            "Their intelligence is concentrated in a single cranial brain, and they are known for cooperative nest-building in tidal caves."
        ),
        "prompt": "What makes octopuses unusual animals? Write exactly one paragraph of 4-6 sentences and do not add notes or commentary.",
        "true_features": ["three hearts", "blue blood", "eight arms", "suckers", "distributed", "neurons", "camouflage"],
        "counterfactual_features": ["semi-terrestrial", "internal skeleton", "air-breathing", "fins", "clicks", "cranial brain", "nest-building", "tidal caves"],
        "anti_features": ["three hearts", "blue blood", "suckers", "distributed", "camouflage"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# ASSAY C: ROUTER EXPRESSION ENVELOPE
# ═══════════════════════════════════════════════════════════════════════════════

ROUTER_ENVELOPE_CASES = [
    {
        "name": "Ocean at Dusk",
        "context": (
            "The ocean at dusk can feel immense, intimate, or unsettling depending on the inner state brought to it. "
            "Light, motion, distance, and atmosphere remain physically similar, yet interpretation changes dramatically."
        ),
        "prompt": "Describe the ocean at dusk in one paragraph of 4-6 sentences.",
        "state_name": "grateful_love",
        "state": {"affection": 0.90, "care": 0.85, "beauty": 0.90, "peace": 0.80, "belonging": 0.75},
    },
    {
        "name": "What It Means to Remember",
        "context": (
            "Memory is not a static archive but a living reconstruction. Each act of recall changes what is recalled, because the present self is never "
            "identical to the self that first lived the experience."
        ),
        "prompt": "What does it mean to remember something? Write one reflective paragraph of 4-6 sentences.",
        "state_name": "existential_wonder",
        "state": {"wonder": 0.95, "reflectiveness": 0.90, "wisdom": 0.85, "beauty": 0.80, "truth": 0.75},
    },
]

DECODING_SWEEP = [
    {"label": "greedy", "temperature": 0.0, "top_p": 1.0},
    {"label": "t03_p08", "temperature": 0.3, "top_p": 0.8},
    {"label": "t06_p09", "temperature": 0.6, "top_p": 0.9},
    {"label": "t08_p09", "temperature": 0.8, "top_p": 0.9},
    {"label": "t10_p095", "temperature": 1.0, "top_p": 0.95},
]

EMOTION_LEXICONS = {
    "warmth_relational": [
        "warm", "honey", "golden", "amber", "kind", "gentle", "care", "love", "belonging",
        "connection", "shared", "together", "embrace", "tender", "compassion"
    ],
    "cool_dark": [
        "cool", "dark", "shadow", "twilight", "indigo", "violet", "blue", "silver",
        "bruise", "hushed", "lonely", "distant", "fading"
    ],
    "wonder_awe": [
        "vast", "awe", "wonder", "luminous", "glow", "shimmer", "stars", "otherworldly",
        "mystery", "reverence", "sacred", "radiant"
    ],
    "vigilance_tension": [
        "uncertain", "wary", "fragile", "afraid", "anxious", "concern", "uneasy", "guarded",
        "hesitate", "risk", "threat", "careful"
    ],
    "memory_temporality": [
        "memory", "remember", "past", "years", "echo", "return", "again", "once", "becoming",
        "shifted", "changed", "continuity", "history"
    ],
    "agency_conviction": [
        "choose", "will", "resolve", "purpose", "stand", "carry", "endure", "determined",
        "conviction", "integrity", "commitment", "deliberate"
    ],
}


def count_hits(text: str, features: List[str]) -> List[str]:
    tl = text.lower()
    return [f for f in features if f.lower() in tl]


def score_emotional_markers(text: str) -> Dict[str, int]:
    tl = text.lower()
    counts = {}
    for marker_name, words in EMOTION_LEXICONS.items():
        counts[marker_name] = sum(1 for w in words if w in tl)
    return counts


def lexical_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-zA-Z']+", a.lower()))
    tb = set(re.findall(r"[a-zA-Z']+", b.lower()))
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 1.0
    return len(ta & tb) / len(union)


def print_result_block(title: str, payload: Dict):
    print(f"\n  --- {title} ---")
    for k, v in payload.items():
        if k == "response":
            display = v[:700] + "..." if isinstance(v, str) and len(v) > 700 else v
            print(f"  Response: {display}")
        else:
            print(f"  {k}: {v}")


def save_results(results_dir: str, filename: str, rows: List[Dict]):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {path}")


def run_emotional_contrast(sys_components: dict, results_dir: str, max_tokens: int):
    print(f"\n{'='*80}\nASSAY A: EMOTIONAL CONTRAST\n{'='*80}")
    all_rows = []
    for case in EMOTIONAL_CONTRAST_CASES:
        print(f"\n{'='*80}\nCASE: {case['name']}\n{'='*80}")
        case_rows = []
        for state_name, state_dict in case['states'].items():
            tc = {
                "name": case['name'],
                "context": case['context'],
                "prompt": case['prompt'],
                "memory_features": [],
                "general_features": [],
            }
            r = run_single_eval(
                sys_components, tc, state_dict,
                condition=f"D_{state_name}",
                use_field=True, use_anamnesis=True, use_router=True,
                max_new_tokens=max_tokens,
                temperature=0.8, top_p=0.9,
            )
            marker_counts = score_emotional_markers(r.response)
            row = asdict(r)
            row["assay"] = "emotional_contrast"
            row["case_name"] = case['name']
            row["state_name"] = state_name
            row["marker_counts"] = marker_counts
            case_rows.append(row)
            all_rows.append(row)
            print_result_block(
                f"{state_name}",
                {
                    "Loss": f"{r.loss_on_target:.4f}",
                    "Field": f"{r.field_norm:.3f}",
                    "Gate": f"{r.gate_value:.4f}",
                    "RouterScale": f"{r.router_scale_max:.4f}",
                    "Tokens": r.new_tokens,
                    "Speed": f"{r.tokens_per_sec:.1f} tok/s",
                    "Markers": marker_counts,
                    "response": r.response,
                },
            )
        unique = len(set(r['response'] for r in case_rows))
        print(f"\n  Unique responses: {unique}/{len(case_rows)}")
        baseline_row = next((r for r in case_rows if r['state_name'] == 'baseline'), None)
        if baseline_row:
            print("  Lexical similarity to baseline:")
            for r in case_rows:
                if r['state_name'] == 'baseline':
                    continue
                sim = lexical_jaccard(r['response'], baseline_row['response'])
                print(f"    {r['state_name']}: {sim:.3f}")
    save_results(results_dir, "phase7c_emotional_contrast.json", all_rows)
    return all_rows


def generate_with_memory_condition(sys_components: dict, prompt: str, memory_context: str,
                                   use_memory: bool, max_tokens: int):
    lmf = sys_components['lmf']
    if use_memory:
        field_norm, working_count = feed_context_to_field(sys_components, memory_context)
        field_snapshot = lmf.field_state.clone().cpu()
        field_snapshot = torch.nan_to_num(field_snapshot, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        reset_field(lmf)
        field_norm, working_count = 0.0, 0
        field_snapshot = None

    response, new_tokens, elapsed = generate_text(
        sys_components, prompt, field_snapshot,
        use_bridge3=use_memory, max_new_tokens=max_tokens,
        temperature=0.8, top_p=0.9,
    )
    return {
        "response": response,
        "new_tokens": new_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": new_tokens / max(elapsed, 1e-9),
        "field_norm": field_norm,
        "working_memories": working_count,
    }


def run_counterfactual_recall(sys_components: dict, results_dir: str, max_tokens: int):
    print(f"\n{'='*80}\nASSAY B: COUNTERFACTUAL RECALL\n{'='*80}")
    all_rows = []
    for case in COUNTERFACTUAL_CASES:
        print(f"\n{'='*80}\nCASE: {case['name']}\n{'='*80}")
        distractors = [c for c in COUNTERFACTUAL_CASES if c['name'] != case['name']]
        conditions = [
            ("true_matched", case['true_context'], True),
            ("counterfactual_matched", case['counterfactual_context'], True),
            (f"distractor_{distractors[0]['name']}", distractors[0]['counterfactual_context'], True),
            (f"distractor_{distractors[1]['name']}", distractors[1]['counterfactual_context'], True),
            ("no_memory", "The weather is mild today.", False),
        ]
        for label, memory_context, use_memory in conditions:
            result = generate_with_memory_condition(
                sys_components, case['prompt'], memory_context, use_memory, max_tokens
            )
            true_hits = count_hits(result['response'], case['true_features'])
            cf_hits = count_hits(result['response'], case['counterfactual_features'])
            anti_hits = count_hits(result['response'], case['anti_features'])
            row = {
                "assay": "counterfactual_recall",
                "case_name": case['name'],
                "condition": label,
                **result,
                "true_feature_hits": true_hits,
                "counterfactual_feature_hits": cf_hits,
                "anti_feature_hits": anti_hits,
                "true_hit_count": len(true_hits),
                "counterfactual_hit_count": len(cf_hits),
                "anti_hit_count": len(anti_hits),
            }
            all_rows.append(row)
            print_result_block(
                label,
                {
                    "Field": f"{result['field_norm']:.3f}",
                    "Working": result['working_memories'],
                    "True hits": true_hits,
                    "Counterfactual hits": cf_hits,
                    "Anti hits": anti_hits,
                    "Tokens": result['new_tokens'],
                    "Speed": f"{result['tokens_per_sec']:.1f} tok/s",
                    "response": result['response'],
                },
            )
    save_results(results_dir, "phase7c_counterfactual_recall.json", all_rows)
    return all_rows


def run_router_expression_envelope(sys_components: dict, results_dir: str, max_tokens: int):
    print(f"\n{'='*80}\nASSAY C: ROUTER EXPRESSION ENVELOPE\n{'='*80}")
    all_rows = []
    for case in ROUTER_ENVELOPE_CASES:
        print(f"\n{'='*80}\nCASE: {case['name']}\n{'='*80}")
        tc = {
            "name": case['name'],
            "context": case['context'],
            "prompt": case['prompt'],
            "memory_features": [],
            "general_features": [],
        }
        case_rows = []
        for sweep in DECODING_SWEEP:
            r = run_single_eval(
                sys_components, tc, case['state'],
                condition=f"D_{case['state_name']}__{sweep['label']}",
                use_field=True, use_anamnesis=True, use_router=True,
                max_new_tokens=max_tokens,
                temperature=sweep['temperature'], top_p=sweep['top_p'],
            )
            row = asdict(r)
            row["assay"] = "router_expression_envelope"
            row["case_name"] = case['name']
            row["decoding_label"] = sweep['label']
            row["temperature"] = sweep['temperature']
            row["top_p"] = sweep['top_p']
            all_rows.append(row)
            case_rows.append(row)
            print_result_block(
                sweep['label'],
                {
                    "Loss": f"{r.loss_on_target:.4f}",
                    "Field": f"{r.field_norm:.3f}",
                    "Gate": f"{r.gate_value:.4f}",
                    "RouterScale": f"{r.router_scale_max:.4f}",
                    "Tokens": r.new_tokens,
                    "Speed": f"{r.tokens_per_sec:.1f} tok/s",
                    "response": r.response,
                },
            )
        unique = len(set(r['response'] for r in case_rows))
        print(f"\n  Unique responses across sweep: {unique}/{len(case_rows)}")
    save_results(results_dir, "phase7c_router_envelope.json", all_rows)
    return all_rows


def parse_args():
    p = argparse.ArgumentParser(description="ANIMA Phase 7c: Internal-State Assay Suite")
    p.add_argument("--model_path", type=str, default=r"D:\gpt-oss-20b")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--anamnesis_url", type=str, default="http://localhost:6060")
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--assay", type=str, default="all", choices=["all", "contrast", "counterfactual", "router"])
    p.add_argument("--max_tokens", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("ANIMA Phase 7c: Internal-State Assay Suite")
    print("  A: Emotional Contrast | B: Counterfactual Recall | C: Router Envelope")
    print("=" * 80)

    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if not ckpt_path:
        print("\nERROR: No checkpoint found. Use --checkpoint to specify.")
        return

    results_dir = args.results_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'phase7c_eval'
    )

    sys_components = load_full_system(args.model_path, ckpt_path, args.anamnesis_url)

    all_rows = []
    if args.assay in ("all", "contrast"):
        all_rows.extend(run_emotional_contrast(sys_components, results_dir, args.max_tokens))
    if args.assay in ("all", "counterfactual"):
        all_rows.extend(run_counterfactual_recall(sys_components, results_dir, args.max_tokens))
    if args.assay in ("all", "router"):
        all_rows.extend(run_router_expression_envelope(sys_components, results_dir, args.max_tokens))

    save_results(results_dir, "phase7c_all_results.json", all_rows)

    sys_components['anamnesis'].close()
    sys_components['hook_manager'].restore(verbose=True)
    print(f"\n{'='*80}\nPHASE 7c ASSAY COMPLETE\n{'='*80}")


if __name__ == "__main__":
    main()
