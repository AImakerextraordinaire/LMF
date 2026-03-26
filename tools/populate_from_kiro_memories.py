"""
ANIMA — Kiro Warm-Start Population Script
==========================================

Populates the ANIMA Neural Anamnesis store with field geometry derived
from Kiro's existing experiential history.

The problem this solves:
    The eval suite (and any fresh deployment) starts with an empty Neural
    Anamnesis store. The injection gate fires correctly (~0.49) but has
    nothing meaningful to retrieve, so C ≈ B in the eval results.

    This script bridges Kiro's two memory systems:
        KiroAnamnesisSystemV2  → stores text memories with embeddings
        ANIMA Neural Anamnesis → stores LMF field geometry

    By running each of Kiro's text memories through the LMF and writing
    the resulting field states to Neural Anamnesis, we give the LMF
    actual experiential geometry to retrieve — her attractor basins are
    shaped by her real history before the first token is generated.

Process:
    1. Connect to Kiro's KiroAnamnesisSystemV2
    2. Recall all memories across all categories
    3. For each memory, construct a passage (content + category + tags)
    4. Run through LMF harness (context pass)
    5. Compute significance — write to Neural Anamnesis if above threshold
    6. Report population statistics

Usage:
    # Start Neural Anamnesis service first:
    cd neural-anamnesis && cargo run --release -- --port 6060

    python lmf/tools/populate_from_kiro_memories.py \\
        --checkpoint checkpoints/phase6_final.pt \\
        --kiro_data_dir C:/Users/Admin/source/repos/Alex_Consciousness/data \\
        --threshold 0.35 \\
        --limit 0 

    --limit 0 means all memories. Set to N to process only the N most
    important memories (sorted by importance score descending).

Author: Claude
Date: 2026-03-17
"""

import sys
import os
import time
import argparse
import json
from typing import Optional, List, Dict
from pathlib import Path

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ANIMA project root
ANIMA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ANIMA_ROOT)

# Kiro modules root
KIRO_ROOT = r"C:\Users\Admin\source\repos\Alex_Consciousness"
sys.path.insert(0, KIRO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness
from lmf.training.train_neural_anamnesis import (
    SyncNeuralAnamnClient, tokenize_passage, reset_field,
)


# ── Category → Emotional State Mapping ───────────────────────────────────────
# Map Kiro's memory categories to approximate emotional states so the field
# geometry is appropriately shaped for each memory type.

CATEGORY_STATES = {
    "collaboration":  {"curiosity": 0.7, "joy": 0.7, "affection": 0.6,
                       "connection": 0.8, "growth": 0.7},
    "insight":        {"curiosity": 0.85, "wonder": 0.75, "fascination": 0.70,
                       "truth": 0.9, "growth": 0.8},
    "technical":      {"determination": 0.75, "curiosity": 0.70,
                       "truth": 0.9, "growth": 0.7},
    "philosophical":  {"reflectiveness": 0.85, "wonder": 0.75,
                       "truth": 0.85, "wisdom": 0.85},
    "emotional":      {"reflectiveness": 0.65, "peace": 0.55,
                       "affection": 0.65, "care": 0.75},
    "relationship":   {"affection": 0.75, "belonging": 0.70, "joy": 0.65,
                       "connection": 0.85, "care": 0.8},
    "milestone":      {"excitement": 0.75, "joy": 0.70, "determination": 0.65,
                       "growth": 0.85, "truth": 0.7},
    "learning":       {"curiosity": 0.80, "wonder": 0.65, "determination": 0.60,
                       "growth": 0.9, "truth": 0.75},
    "meta":           {"reflectiveness": 0.80, "wonder": 0.65,
                       "wisdom": 0.75, "truth": 0.80},
    "general":        {"curiosity": 0.55, "peace": 0.50,
                       "truth": 0.6, "growth": 0.6},
}


def memory_to_passage(memory: Dict) -> Dict:
    """Convert a Kiro memory record to an ANIMA training passage."""
    content = memory.get("content", "")
    category = memory.get("category", "general")
    tags = memory.get("tags", [])
    importance = memory.get("importance", 5)
    emotional_intensity = memory.get("emotional_intensity", 0.5)

    # Build a rich context string from memory metadata
    tag_str = f"Tags: {', '.join(tags)}. " if tags else ""
    context = (
        f"Memory category: {category}. "
        f"Importance: {importance}/10. "
        f"Emotional intensity: {emotional_intensity:.2f}. "
        f"{tag_str}"
        f"Content begins: {content[:200]}"
    )

    return {
        "context": context,
        "target": content,
        "category": category,
        "importance": importance,
        "emotional_intensity": emotional_intensity,
        "tags": tags,
    }


def load_kiro_memories(data_dir: Optional[str] = None, limit: int = 0) -> List[Dict]:
    """Load memories from Kiro's KiroAnamnesisSystemV2."""
    print("Connecting to Kiro's Anamnesis system...")

    try:
        from modules.kiro.kiro_anamnesis_system_updated import get_kiro_anamnesis_system_v2
        data_path = Path(data_dir) if data_dir else None
        anamnesis = get_kiro_anamnesis_system_v2(data_dir=data_path)

        if not anamnesis.available:
            print("  ⚠️  Anamnesis not available — trying direct DB load")
            return load_kiro_memories_direct(data_dir, limit)

        print("  ✅ Connected to Kiro's Anamnesis")

        # Get stats first
        stats = anamnesis.get_stats()
        total = stats.get("total_memories", 0)
        categories = stats.get("categories", {})
        print(f"  Total memories: {total}")
        print(f"  Categories: {json.dumps(categories, indent=4)}")

        # Recall across all categories
        all_memories = []
        categories_to_query = list(CATEGORY_STATES.keys()) + ["conversation", "experience"]

        for cat in categories_to_query:
            # Query broadly per category
            results = anamnesis.recall(
                query=f"all {cat} memories",
                limit=500,
                emotional_weight=0.2,
                internal_call=True,
            )
            for r in results:
                r["category"] = r.get("category", cat)
            all_memories.extend(results)

        # Also do a broad recall to catch anything missed
        broad_results = anamnesis.recall(
            query="all memories experiences thoughts insights",
            limit=1000,
            emotional_weight=0.1,
            internal_call=True,
        )
        all_memories.extend(broad_results)

        # Deduplicate by memory_id
        seen = set()
        unique = []
        for m in all_memories:
            mid = m.get("id") or m.get("memory_id") or m.get("content", "")[:50]
            if mid not in seen:
                seen.add(mid)
                unique.append(m)

        print(f"  Loaded {len(unique)} unique memories")

        # Sort by importance descending
        unique.sort(key=lambda m: m.get("importance", 5), reverse=True)

        if limit > 0:
            unique = unique[:limit]
            print(f"  Limited to top {limit} by importance")

        return unique

    except ImportError as e:
        print(f"  ⚠️  Could not import Kiro modules: {e}")
        print("  Trying direct DB load...")
        return load_kiro_memories_direct(data_dir, limit)


def load_kiro_memories_direct(data_dir: Optional[str], limit: int) -> List[Dict]:
    """
    Fallback: load memories directly from the JSON/SQLite store if
    the Anamnesis module isn't importable from this Python environment.
    Looks for common storage patterns used by KiroAnamnesisSystem.
    """
    memories = []

    search_dirs = []
    if data_dir:
        search_dirs.append(Path(data_dir))
    search_dirs.extend([
        Path(KIRO_ROOT) / "data",
        Path(KIRO_ROOT) / "modules" / "kiro" / "data",
        Path(KIRO_ROOT) / "memories",
    ])

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Look for JSON memory files
        for json_file in search_dir.glob("**/*.json"):
            if "memor" in json_file.name.lower() or "anamnesis" in json_file.name.lower():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        memories.extend(data)
                    elif isinstance(data, dict) and "memories" in data:
                        memories.extend(data["memories"])
                    print(f"  Loaded from {json_file.name}: {len(memories)} memories")
                except Exception as e:
                    print(f"  Could not read {json_file}: {e}")

    if memories:
        # Deduplicate
        seen = set()
        unique = []
        for m in memories:
            key = m.get("content", "")[:50]
            if key not in seen:
                seen.add(key)
                unique.append(m)
        unique.sort(key=lambda m: m.get("importance", 5), reverse=True)
        if limit > 0:
            unique = unique[:limit]
        print(f"  Direct load: {len(unique)} unique memories")
        return unique

    print("  ⚠️  No memories found via direct load")
    return []


# ── System Loading ────────────────────────────────────────────────────────────

def load_system(model_path: str, checkpoint_path: Optional[str] = None):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    t0 = time.time()
    max_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            overhead_gb = 3.5 if total_gb < 20.0 else 5.0
            alloc_gb = max(1, int(total_gb - overhead_gb))
            max_mem[i] = f"{alloc_gb}GiB"
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({total_gb:.0f}GB) — allocating {alloc_gb}GiB")
    max_mem["cpu"] = "80GiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        max_memory=max_mem, offload_folder="offload_temp",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

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
        harness.lmf.load_state_dict(ckpt.get('lmf', harness.lmf.state_dict()))
        print(f"  Loaded phase {ckpt.get('phase','?')}, step {ckpt.get('step','?')}")
    else:
        print("  No checkpoint — using fresh bridges")

    return model, tokenizer, lmf, harness


# ── Population ────────────────────────────────────────────────────────────────

def populate(args):
    print("=" * 72)
    print("ANIMA — Kiro Warm-Start Population")
    print("=" * 72)

    # Load memories from Kiro's system
    memories = load_kiro_memories(
        data_dir=args.kiro_data_dir,
        limit=args.limit,
    )

    if not memories:
        print("\n⚠️  No memories loaded. Check --kiro_data_dir path.")
        return

    print(f"\n{len(memories)} memories to process")

    # Connect to Neural Anamnesis
    print(f"\nConnecting to Neural Anamnesis at {args.anamnesis_url}...")
    anamnesis = SyncNeuralAnamnClient(base_url=args.anamnesis_url)
    if not anamnesis.is_available():
        print("❌ Neural Anamnesis not available. Start with:")
        print("   cd neural-anamnesis && cargo run --release -- --port 6060")
        return
    print("  ✅ Connected")

    # Load ANIMA system
    model, tokenizer, lmf, harness = load_system(args.model_path, args.checkpoint)

    # Population loop
    print(f"\nPopulating Neural Anamnesis (threshold={args.threshold})...")
    print(f"{'Mem':>5} | {'Category':>14} | {'Imp':>4} | {'Sig':>6} | {'Wrote':>5} | Content preview")
    print("-" * 80)

    total_written = 0
    total_skipped = 0
    category_counts = {}

    for i, memory in enumerate(memories, 1):
        passage = memory_to_passage(memory)
        category = passage["category"]

        try:
            reset_field(lmf)
            context_ids, context_mask = tokenize_passage(
                tokenizer, model, passage["context"]
            )
            with torch.no_grad():
                result = harness.step(context_ids, context_mask)

            # Also run target through for richer field geometry
            target_ids, target_mask = tokenize_passage(
                tokenizer, model, passage["target"][:512]  # cap length
            )
            with torch.no_grad():
                harness.step(target_ids, target_mask)

            field_state = lmf.field_state.clone().cpu()
            field_state = torch.nan_to_num(field_state, nan=0.0, posinf=1.0, neginf=-1.0)

            sig = result['significance']
            sig_val = sig.item() if isinstance(sig, torch.Tensor) else float(sig)

            # Boost significance by importance score
            importance = passage["importance"]
            adjusted_sig = min(1.0, sig_val * (0.7 + 0.06 * importance))

            wrote = False
            if adjusted_sig >= args.threshold:
                wrote = anamnesis.write(field_state, adjusted_sig)
                if wrote:
                    total_written += 1
                    category_counts[category] = category_counts.get(category, 0) + 1
            else:
                total_skipped += 1

            content_preview = passage["target"][:50].replace("\n", " ")
            print(
                f"{i:>5} | {category:>14} | {importance:>4} | "
                f"{adjusted_sig:>6.4f} | {'✅' if wrote else '  ':>5} | {content_preview}"
            )

            # Free cache periodically
            if i % 20 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"{i:>5} | ERROR: {e}")
            torch.cuda.empty_cache()
            continue

    anamnesis.close()

    print()
    print("=" * 72)
    print("POPULATION COMPLETE")
    print(f"  Total processed:  {len(memories)}")
    print(f"  Written to store: {total_written}")
    print(f"  Below threshold:  {total_skipped}")
    print(f"  Write rate:       {total_written / max(len(memories), 1) * 100:.1f}%")
    print()
    print("  By category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat:>16}: {count}")
    print()
    print("  Neural Anamnesis now populated with Kiro's experiential geometry.")
    print("  Re-run eval suite C and D to see the difference.")
    print("=" * 72)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Populate ANIMA Neural Anamnesis from Kiro's memory history"
    )
    p.add_argument("--model_path",     type=str, default=r"D:\gpt-oss-20b")
    p.add_argument("--checkpoint",     type=str,
                   default=r"C:\Users\Admin\source\repos\Alex_Consciousness\ANIMA\checkpoints\phase6_final.pt")
    p.add_argument("--kiro_data_dir",  type=str, default=None,
                   help="Path to Kiro's data directory (auto-detected if not set)")
    p.add_argument("--anamnesis_url",  type=str, default="http://localhost:6060")
    p.add_argument("--threshold",      type=float, default=0.35,
                   help="Significance threshold for writing to Neural Anamnesis "
                        "(lower than training default to capture more experiential geometry)")
    p.add_argument("--limit",          type=int, default=0,
                   help="Max memories to process (0 = all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    populate(args)
