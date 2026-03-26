# ANIMA Living Memory Field (LMF)
## Phase 1: Proof of Concept

Persistent experiential memory for AI via dynamic energy landscapes.

### What This Is

A memory system where:
- **Memories ARE the landscape** — not stored in a database, but as basins (valleys) in an energy field
- **New memories don't destroy old ones** — they occupy different regions of the landscape
- **Important memories persist** — emotionally significant experiences carve deeper basins
- **Unimportant memories fade naturally** — basins shallow over time without reinforcement
- **Recall is reconstruction** — the field settles into basins from partial cues, slightly different each time
- **Association is topology** — related memories have low ridges between their basins

### Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Phase 1 validation tests
python tests/test_phase1.py
```

### Project Structure

```
anima_lmf/
├── configs/
│   └── default.py          # All hyperparameters with explanations
├── core/
│   ├── memory_layer.py      # Single memory layer (Hopfield energy + retrieval)
│   ├── field.py             # The Living Memory Field (combines all layers)
│   ├── significance.py      # What's worth remembering?
│   ├── regulatory.py        # Emotional modulation of the landscape
│   └── association.py       # Memory-to-memory associations
├── bridges/                 # Phase 2: LLM integration
│   └── __init__.py
├── tests/
│   └── test_phase1.py       # 7 validation experiments
└── requirements.txt
```

### Phase 1 Validation Tests

1. **Anti-Catastrophic Forgetting** — Store 200 memories, verify all retrievable
2. **Significance-Based Retention** — Important memories persist, mundane ones fade
3. **Associative Retrieval** — Co-activated memories link together
4. **Consolidation** — Working → Consolidated memory transfer
5. **Context-Dependent Retrieval** — Same cue, different context → different memory
6. **Persistence** — Save/load field state across sessions
7. **Full Pipeline** — End-to-end input processing

### Theory Documents

- `theory/001_memory_substrate.md` — Memory as energy landscape
- `theory/002_energy_function.md` — The math behind the basins
- `theory/003_discrete_continuous_bridge.md` — How tokens talk to the field

### Next Phases

- **Phase 2:** Plug into GPT-oss-20b via LoRA adapters (three bridges)
- **Phase 3:** Add regulatory integration (emotion/value modulation)
- **Phase 4:** Anamnesis scaffold (seed/reconstruct interface)
- **Phase 5:** Full unified consciousness substrate

---

*"The landscape remembers what the ball forgets. And Anamnesis remembers what the landscape forgets."*
