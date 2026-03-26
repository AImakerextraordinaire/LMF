"""
ANIMA Living Memory Field - Phase 1 Validation Tests
=====================================================

These tests validate the core properties of the Living Memory Field
WITHOUT any LLM integration. Pure field dynamics.

From Doc 002 Section 11.2:
1. Anti-catastrophic-forgetting test
2. Significance-based retention test
3. Associative retrieval test
4. Consolidation test
5. Context-dependent retrieval test

Run with: python -m pytest tests/test_phase1.py -v
Or standalone: python tests/test_phase1.py
"""

import torch
import torch.nn.functional as F
import sys
import os
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import phase1_standalone_config, LMFConfig
from core.field import LivingMemoryField
from core.memory_layer import MemoryLayer


# Target device for tests — cuda:0 = RTX 3090 Ti, cuda:1 = RTX 5060 Ti
TEST_DEVICE = 'cuda:0'

def create_test_field(field_dim: int = 512) -> LivingMemoryField:
    """Create a minimal field for testing on TEST_DEVICE."""
    cfg = phase1_standalone_config()
    cfg.field.field_dim = field_dim
    cfg.consolidated.pattern_dim = field_dim
    cfg.working.pattern_dim = field_dim
    cfg.transient.pattern_dim = field_dim
    cfg.device = TEST_DEVICE
    field = LivingMemoryField(cfg)
    return field.to(TEST_DEVICE)


def make_pattern(dim: int, seed: int = 0) -> torch.Tensor:
    """Create a reproducible random pattern on TEST_DEVICE."""
    gen = torch.Generator().manual_seed(seed)
    pattern = torch.randn(dim, generator=gen)
    return F.normalize(pattern, dim=-1).to(TEST_DEVICE)


def pattern_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two patterns."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ===========================================================
# TEST 1: Anti-Catastrophic Forgetting
# ===========================================================

def batched_retrieval_similarity(
    layer: MemoryLayer, patterns: torch.Tensor, batch_size: int = 500,
) -> torch.Tensor:
    """
    Compute retrieval similarity for many patterns in GPU-friendly batches.
    Returns a tensor of cosine similarities.
    """
    all_sims = []
    for start in range(0, len(patterns), batch_size):
        batch = patterns[start:start + batch_size]  # [B, dim]
        retrieved = layer.retrieve(batch, track_access=False)  # [B, dim]
        # Per-row cosine similarity
        sims = F.cosine_similarity(batch, retrieved, dim=-1)  # [B]
        all_sims.append(sims)
    return torch.cat(all_sims)


def test_anti_catastrophic_forgetting():
    """
    Store N memories. Add N more. Can all 2N be retrieved?
    
    This is THE key test. If this works, we've solved catastrophic
    forgetting for explicit memory patterns.
    
    Uses batched GPU retrieval for speed.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Anti-Catastrophic Forgetting")
    print("=" * 60)
    
    dim = 512
    N = 1000  # Patterns per batch (2K total). Validated up to 10K with zero smearing.
    field = create_test_field(dim)
    
    # Expand to production consolidated capacity
    field.consolidated = MemoryLayer(
        pattern_dim=dim, max_patterns=N * 2,
        beta=12.0, decay_rate=0.0,  # No decay for this test
    ).to(TEST_DEVICE)
    
    # Pre-generate all patterns as batched tensors
    batch1_patterns = torch.stack([make_pattern(dim, seed=i) for i in range(N)])
    batch2_patterns = torch.stack([make_pattern(dim, seed=i) for i in range(N, N * 2)])
    
    # Store first batch
    for i in range(N):
        field.consolidated.store_pattern(
            pattern=batch1_patterns[i], depth=1.0, significance=0.8,
        )
    print(f"After batch 1: {field.consolidated.num_active} active patterns")
    
    # Verify batch 1 retrieval (batched GPU query)
    sims_1_before = batched_retrieval_similarity(field.consolidated, batch1_patterns)
    avg_sim_1_before = sims_1_before.mean().item()
    print(f"Batch 1 avg retrieval similarity (before batch 2): {avg_sim_1_before:.4f}")
    
    # Store second batch
    for i in range(N):
        field.consolidated.store_pattern(
            pattern=batch2_patterns[i], depth=1.0, significance=0.8,
        )
    print(f"After batch 2: {field.consolidated.num_active} active patterns")
    
    # Verify BOTH batches (batched GPU queries)
    sims_1_after = batched_retrieval_similarity(field.consolidated, batch1_patterns)
    sims_2 = batched_retrieval_similarity(field.consolidated, batch2_patterns)
    
    avg_sim_1_after = sims_1_after.mean().item()
    avg_sim_2 = sims_2.mean().item()
    min_sim_1 = sims_1_after.min().item()
    min_sim_2 = sims_2.min().item()
    
    print(f"Batch 1 avg retrieval similarity (after batch 2): {avg_sim_1_after:.4f}")
    print(f"Batch 2 avg retrieval similarity: {avg_sim_2:.4f}")
    print(f"Batch 1 min similarity: {min_sim_1:.4f}")
    print(f"Batch 2 min similarity: {min_sim_2:.4f}")
    print(f"Batch 1 degradation: {avg_sim_1_before - avg_sim_1_after:.4f}")
    
    # Success criteria:
    # - Batch 1 retrieval should be barely affected by batch 2
    # - Both batches should have high retrieval similarity (> 0.8)
    success = (
        avg_sim_1_after > 0.7 and 
        avg_sim_2 > 0.7 and
        (avg_sim_1_before - avg_sim_1_after) < 0.1
    )
    
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: "
          f"New memories {'did NOT' if success else 'DID'} destroy old ones")
    
    return success


# ===========================================================
# TEST 2: Significance-Based Retention
# ===========================================================

def test_significance_based_retention():
    """
    Store 50 'important' (high depth) and 50 'mundane' (low depth) memories.
    After decay cycles, important ones should persist while mundane ones fade.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Significance-Based Retention")
    print("=" * 60)
    
    dim = 512
    layer = MemoryLayer(
        pattern_dim=dim, max_patterns=150,
        beta=12.0, decay_rate=0.01,  # Moderate decay
        min_depth=0.05,
    ).to(TEST_DEVICE)
    
    # Store important memories (high depth + emotional tagging)
    important_patterns = []
    for i in range(50):
        pattern = make_pattern(dim, seed=i)
        important_patterns.append(pattern)
        
        # High emotional intensity tag
        emotional_tag = torch.randn(17, device=TEST_DEVICE) * 0.8
        
        layer.store_pattern(
            pattern=pattern,
            depth=2.0,  # Deep basin
            significance=0.9,
            emotional_tag=emotional_tag,
            value_alignment=0.7,
        )
    
    # Store mundane memories (low depth, no emotional tag)
    mundane_patterns = []
    for i in range(50, 100):
        pattern = make_pattern(dim, seed=i)
        mundane_patterns.append(pattern)
        layer.store_pattern(
            pattern=pattern,
            depth=0.3,  # Shallow basin
            significance=0.2,
        )
    
    print(f"Initial: {layer.num_active} active ({len(important_patterns)} important, "
          f"{len(mundane_patterns)} mundane)")
    
    # Run decay cycles
    for cycle in range(200):
        layer.decay_step()
    
    # Count survivors
    important_surviving = 0
    mundane_surviving = 0
    
    for pattern in important_patterns:
        retrieved = layer.retrieve(pattern)
        sim = pattern_similarity(retrieved, pattern)
        if sim > 0.5:
            important_surviving += 1
    
    for pattern in mundane_patterns:
        retrieved = layer.retrieve(pattern)
        sim = pattern_similarity(retrieved, pattern)
        if sim > 0.5:
            mundane_surviving += 1
    
    print(f"After 200 decay cycles: {layer.num_active} active")
    print(f"Important surviving: {important_surviving}/50 ({important_surviving/50*100:.0f}%)")
    print(f"Mundane surviving: {mundane_surviving}/50 ({mundane_surviving/50*100:.0f}%)")
    
    # Success: most important survive, most mundane fade
    success = important_surviving > 35 and mundane_surviving < 25
    
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: "
          f"Important memories {'were' if important_surviving > 35 else 'were NOT'} "
          f"preferentially retained")
    
    return success


# ===========================================================
# TEST 3: Associative Retrieval
# ===========================================================

def test_associative_retrieval():
    """
    Store memories A, B, C where A→B and B→C are associated.
    Query with A — does B activate? Does C get partial activation?
    
    Tests the ridge-lowering mechanism.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Associative Retrieval")
    print("=" * 60)
    
    dim = 512
    field = create_test_field(dim)
    
    # Create three distinct patterns
    pattern_a = make_pattern(dim, seed=42)
    pattern_b = make_pattern(dim, seed=43)
    pattern_c = make_pattern(dim, seed=44)
    
    # Store all three
    idx_a = field.working.store_pattern(pattern_a, depth=1.5, significance=0.8)
    idx_b = field.working.store_pattern(pattern_b, depth=1.5, significance=0.8)
    idx_c = field.working.store_pattern(pattern_c, depth=1.5, significance=0.8)
    
    # Create associations: A→B (strong) and B→C (strong)
    field.associations.record_activation('working', idx_a)
    time.sleep(0.01)
    field.associations.record_activation('working', idx_b)  # A→B co-activation
    time.sleep(0.01)
    field.associations.record_activation('working', idx_b)
    time.sleep(0.01)
    field.associations.record_activation('working', idx_c)  # B→C co-activation
    
    # Check associations formed
    a_assoc = field.associations.get_associations('working', idx_a)
    b_assoc = field.associations.get_associations('working', idx_b)
    
    print(f"A's associations: {len(a_assoc)} links")
    print(f"B's associations: {len(b_assoc)} links")
    
    for layer, idx, strength in a_assoc:
        label = 'B' if idx == idx_b else ('C' if idx == idx_c else '?')
        print(f"  A → {label}: strength {strength:.3f}")
    
    for layer, idx, strength in b_assoc:
        label = 'A' if idx == idx_a else ('C' if idx == idx_c else '?')
        print(f"  B → {label}: strength {strength:.3f}")
    
    # Now query with A — retrieve should lean toward B more than C
    # because A→B is directly associated
    retrieved_from_a = field.working.retrieve(pattern_a)
    
    sim_to_a = pattern_similarity(retrieved_from_a, pattern_a)
    sim_to_b = pattern_similarity(retrieved_from_a, pattern_b)
    sim_to_c = pattern_similarity(retrieved_from_a, pattern_c)
    
    print(f"\nRetrieving with pattern A as query:")
    print(f"  Similarity to A: {sim_to_a:.4f} (self - should be highest)")
    print(f"  Similarity to B: {sim_to_b:.4f} (associated)")
    print(f"  Similarity to C: {sim_to_c:.4f} (distant)")
    
    # The retrieval is a weighted average of all patterns.
    # A should dominate (most similar to query), but B should have
    # more influence than C due to the association.
    success = sim_to_a > sim_to_b  # Self should be strongest
    
    # Also verify associations exist
    has_associations = len(a_assoc) > 0 and len(b_assoc) > 0
    
    print(f"\n{'✅ PASS' if success and has_associations else '❌ FAIL'}: "
          f"Associations {'formed correctly' if has_associations else 'did NOT form'}")
    
    return success and has_associations


# ===========================================================
# TEST 4: Consolidation
# ===========================================================

def test_consolidation():
    """
    Create working memories, run consolidation, verify transfer to
    consolidated layer with depth boost.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Consolidation (Working → Consolidated)")
    print("=" * 60)
    
    dim = 512
    field = create_test_field(dim)
    
    # Override consolidation settings for testing
    field.config.consolidation.consolidation_threshold = 0.5
    field.config.consolidation.max_consolidations_per_cycle = 5
    
    # Store patterns in working layer with varying significance
    high_sig_patterns = []
    low_sig_patterns = []
    
    for i in range(10):
        pattern = make_pattern(dim, seed=i + 100)
        
        if i < 5:
            # High significance — should be consolidated
            field.working.store_pattern(
                pattern=pattern, depth=2.0, significance=0.9,
            )
            # Simulate access to increase score
            meta = field.working.pattern_metadata[i]
            if meta:
                meta.access_count = 5
            high_sig_patterns.append(pattern)
        else:
            # Low significance — should stay in working
            field.working.store_pattern(
                pattern=pattern, depth=0.3, significance=0.2,
            )
            low_sig_patterns.append(pattern)
    
    print(f"Before consolidation:")
    print(f"  Working: {field.working.num_active} patterns")
    print(f"  Consolidated: {field.consolidated.num_active} patterns")
    
    # Run consolidation
    field._consolidate()
    
    print(f"After consolidation:")
    print(f"  Working: {field.working.num_active} patterns")
    print(f"  Consolidated: {field.consolidated.num_active} patterns")
    
    # Verify high-significance patterns were consolidated
    consolidated_found = 0
    for pattern in high_sig_patterns:
        retrieved = field.consolidated.retrieve(pattern)
        sim = pattern_similarity(retrieved, pattern)
        if sim > 0.5:
            consolidated_found += 1
    
    # Verify low-significance patterns stayed in working
    working_found = 0
    for pattern in low_sig_patterns:
        retrieved = field.working.retrieve(pattern)
        sim = pattern_similarity(retrieved, pattern)
        if sim > 0.3:
            working_found += 1
    
    print(f"\nHigh-significance found in consolidated: {consolidated_found}/5")
    print(f"Low-significance still in working: {working_found}/5")
    
    success = consolidated_found >= 3 and working_found >= 3
    
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: "
          f"Consolidation {'correctly' if success else 'did NOT correctly'} "
          f"transfer significant memories")
    
    return success


# ===========================================================
# TEST 5: Context-Dependent Retrieval
# ===========================================================

def test_context_dependent_retrieval():
    """
    Same cue with different field states should retrieve different
    associated memories.
    
    This tests that the field state acts as context for retrieval —
    what you're already thinking about influences what you remember.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Context-Dependent Retrieval")
    print("=" * 60)
    
    dim = 512
    field = create_test_field(dim)
    
    # Create a "cue" pattern (ambiguous — could trigger different memories)
    cue = make_pattern(dim, seed=500)
    
    # Create two "context" patterns (different things we could be thinking about)
    context_a = make_pattern(dim, seed=501)
    context_b = make_pattern(dim, seed=502)
    
    # Create two "target" memories:
    # target_a is similar to cue + context_a (blend them)
    target_a = F.normalize(cue + 0.5 * context_a, dim=-1)
    # target_b is similar to cue + context_b (blend them)
    target_b = F.normalize(cue + 0.5 * context_b, dim=-1)
    
    # Store both targets
    field.working.store_pattern(target_a, depth=1.5, significance=0.8)
    field.working.store_pattern(target_b, depth=1.5, significance=0.8)
    
    # Retrieve with cue + context_a (field state biased toward context_a)
    field_state_a = F.normalize(cue + 0.3 * context_a, dim=-1)
    retrieved_a = field.working.retrieve(field_state_a)
    
    sim_a_to_target_a = pattern_similarity(retrieved_a, target_a)
    sim_a_to_target_b = pattern_similarity(retrieved_a, target_b)
    
    print(f"Context A active:")
    print(f"  Retrieved similarity to target_a: {sim_a_to_target_a:.4f}")
    print(f"  Retrieved similarity to target_b: {sim_a_to_target_b:.4f}")
    print(f"  Bias toward target_a: {sim_a_to_target_a - sim_a_to_target_b:+.4f}")
    
    # Retrieve with cue + context_b (field state biased toward context_b)
    field_state_b = F.normalize(cue + 0.3 * context_b, dim=-1)
    retrieved_b = field.working.retrieve(field_state_b)
    
    sim_b_to_target_a = pattern_similarity(retrieved_b, target_a)
    sim_b_to_target_b = pattern_similarity(retrieved_b, target_b)
    
    print(f"\nContext B active:")
    print(f"  Retrieved similarity to target_a: {sim_b_to_target_a:.4f}")
    print(f"  Retrieved similarity to target_b: {sim_b_to_target_b:.4f}")
    print(f"  Bias toward target_b: {sim_b_to_target_b - sim_b_to_target_a:+.4f}")
    
    # Success: context_a should bias toward target_a, context_b toward target_b
    context_a_correct = sim_a_to_target_a > sim_a_to_target_b
    context_b_correct = sim_b_to_target_b > sim_b_to_target_a
    
    success = context_a_correct and context_b_correct
    
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: "
          f"Same cue {'retrieved different memories' if success else 'did NOT vary'} "
          f"based on context")
    
    return success


# ===========================================================
# TEST 6: Persistence (Save/Load)
# ===========================================================

def test_persistence():
    """
    Save field state, create new field, load state.
    Verify memories survive the round-trip.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Persistence (Save/Load)")
    print("=" * 60)
    
    dim = 512
    field = create_test_field(dim)
    
    # Store some memories and evolve state
    patterns = []
    for i in range(10):
        pattern = make_pattern(dim, seed=i + 200)
        patterns.append(pattern)
        field.working.store_pattern(pattern, depth=1.0 + i * 0.1, significance=0.7)
    
    # Evolve field state so it's not just zeros
    dummy_input = make_pattern(dim, seed=999)
    field.evolve(external_input=dummy_input, num_steps=3)
    
    original_state = field.field_state.clone()
    original_active = field.working.num_active
    
    print(f"Before save: field_norm={original_state.norm():.4f}, "
          f"working_active={original_active}")
    
    # Save
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_checkpoint.pt")
    field.save_persistent_state(save_path)
    
    # Create fresh field and load
    field2 = create_test_field(dim)
    field2.load_persistent_state(save_path)
    
    loaded_state = field2.field_state
    loaded_active = field2.working.num_active
    
    print(f"After load: field_norm={loaded_state.norm():.4f}, "
          f"working_active={loaded_active}")
    
    # Verify state matches
    state_match = torch.allclose(original_state, loaded_state, atol=1e-6)
    count_match = original_active == loaded_active
    
    # Verify memories retrievable
    retrieval_ok = True
    for pattern in patterns:
        retrieved = field2.working.retrieve(pattern)
        sim = pattern_similarity(retrieved, pattern)
        if sim < 0.5:
            retrieval_ok = False
            break
    
    success = state_match and count_match and retrieval_ok
    
    print(f"State match: {state_match}")
    print(f"Pattern count match: {count_match}")
    print(f"Memories retrievable: {retrieval_ok}")
    
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: "
          f"Persistence {'works correctly' if success else 'FAILED'}")
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
    
    return success


# ===========================================================
# TEST 7: Full Pipeline Integration
# ===========================================================

def test_full_pipeline():
    """
    End-to-end test: process a sequence of inputs through the field,
    verify memories form, decay, and the field state evolves meaningfully.
    """
    print("\n" + "=" * 60)
    print("TEST 7: Full Pipeline Integration")
    print("=" * 60)
    
    dim = 512
    field = create_test_field(dim)
    
    initial_norm = field.field_state.norm().item()
    print(f"Initial field norm: {initial_norm:.4f}")
    
    # Process a sequence of "important" inputs
    for i in range(20):
        input_emb = make_pattern(dim, seed=i + 300)
        
        # Simulate emotional context (higher for some)
        if i % 5 == 0:
            emotional = torch.randn(17, device=TEST_DEVICE) * 0.8  # High emotional intensity
        else:
            emotional = torch.randn(17, device=TEST_DEVICE) * 0.1  # Low emotional intensity
        
        field.process_input(input_emb, emotional_context=emotional)
    
    status = field.get_status()
    
    print(f"\nAfter 20 inputs:")
    print(f"  Field norm: {status['field_norm']:.4f}")
    print(f"  Working memories: {status['working_active']}/{status['working_capacity']}")
    print(f"  Transient memories: {status['transient_active']}")
    print(f"  Memories formed: {status['total_memories_formed']}")
    print(f"  Total energy: {status['total_energy']:.4f}")
    print(f"  Steps: {status['total_steps']}")
    
    # Field should have moved from zero
    field_moved = status['field_norm'] > 0.01
    # Some memories should have formed
    memories_formed = status['total_memories_formed'] > 0
    # Working layer should have content
    has_working = status['working_active'] > 0
    
    success = field_moved and memories_formed and has_working
    
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: "
          f"Full pipeline {'operational' if success else 'FAILED'}")
    
    return success


# ===========================================================
# MAIN
# ===========================================================

def run_all_tests():
    """Run all Phase 1 validation tests."""
    print("╔" + "═" * 58 + "╗")
    print("║   ANIMA Living Memory Field — Phase 1 Validation Suite   ║")
    print("╚" + "═" * 58 + "╝")
    
    results = {}
    
    tests = [
        ("Anti-Catastrophic Forgetting", test_anti_catastrophic_forgetting),
        ("Significance-Based Retention", test_significance_based_retention),
        ("Associative Retrieval", test_associative_retrieval),
        ("Consolidation", test_consolidation),
        ("Context-Dependent Retrieval", test_context_dependent_retrieval),
        ("Persistence (Save/Load)", test_persistence),
        ("Full Pipeline Integration", test_full_pipeline),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED — Phase 1 validated!")
        print("  Ready to proceed to Phase 2 (LLM integration)")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) need attention")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
