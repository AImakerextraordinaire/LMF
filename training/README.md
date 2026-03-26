# ANIMA Phase 3: Bridge Training Protocol
## Teaching Bridges Semantic Structure

**Date:** 2026-03-06
**Status:** Design Complete, Pre-Implementation
**Prerequisites:** Phase 2 (all bridges validated, significance calibration 8/8)

---

## 1. The Training Problem

The bridges are validated mechanically but untrained semantically:
- Bridge 1 extracts features but they're not meaningful
- Bridge 2 injects perturbation but up_proj is zeros (LoRA init)
- Bridge 3 produces logit bias but it's diffuse (boosting random tokens)

**Goal:** Teach bridges that memories formed from early context should help
predict later content. This is "Memory-Conditioned Language Modeling."

## 2. Gradient Flow Analysis

### The Detach Barrier
`field.py` line 253: `self.field_state.copy_(s.detach())`

The field state detaches gradients at every evolution step. This means:
- Cannot backprop through multiple steps of field accumulation
- Each training step must treat field_state as a CONSTANT
- Bridges must learn from single-step gradient signals

### Gradient Paths Available

**Bridge 3 (Output):** DIRECT gradient
  loss -> combined_logits -> logit_bias -> transform params -> field_state (constant)
  Bridge 3 learns: "given this field state, what logit bias helps prediction?"

**Bridge 1 (Input):** INDIRECT gradient via Bridge 3
  loss -> Bridge 3 -> updated_field -> perturbation -> Bridge 1 params
  Bridge 1 learns: "produce perturbations that, when added to the field, help Bridge 3"

**Bridge 2 (Memory):** BLOCKED (hooks use torch.no_grad)
  Phase 3b: Remove no_grad, enable gradient through hook injection
  Bridge 2 learns: "inject field state into mid-layers to help token prediction"

## 3. Training Strategy: Two-Phase Per Example

### Phase A: Context Accumulation (no grad)
Process context segments through full harness pipeline.
Field accumulates memories naturally (significance gating, memory formation).
This is inference-mode — cheap, fast, no gradient tracking.

### Phase B: Training Step (with grad)
On the target segment, compute three losses:

**Loss 1: Memory-Conditioned LM (trains Bridge 3)**
  field_state (constant) -> Bridge 3 -> logit_bias -> combined_logits
  CrossEntropy(combined_logits, target_tokens)

**Loss 2: Perturbation Usefulness (trains Bridge 1)**
  hidden_states -> Bridge 1 -> perturbation
  field_state + perturbation -> Bridge 3 -> updated_logits
  CrossEntropy(updated_logits, target_tokens)

**Loss 3: KL Regularization (prevents bias from overwhelming model)**
  KL(combined_distribution || base_distribution)
  Keeps memory influence as a gentle nudge, not a hijack

### Combined Loss
  total_loss = loss_lm + lambda_b1 * loss_bridge1 + lambda_kl * loss_kl

## 4. Training Data Requirements

Need documents with internal reference structure where early content
provides useful context for predicting later content:
- Wikipedia articles (entities defined early, referenced later)
- Technical docs (concepts introduced, then applied)
- Stories (characters, themes, plot elements)

For first validation: use 100-500 Wikipedia articles, split each into
context (first 60%) and target (last 40%).

## 5. Hyperparameters (Initial)

- Optimizer: AdamW
- Learning rate: 1e-4 (bridges only, model frozen)
- Weight decay: 0.01
- KL weight (lambda_kl): 0.1
- Bridge 1 weight (lambda_b1): 0.5
- Context/target split: 60/40
- Max context length: 512 tokens
- Max target length: 256 tokens
- Batch size: 1 (20B model, limited VRAM)
- Gradient accumulation: 4 steps
- Training steps: 1000 (initial validation)
- Warmup: 100 steps linear

## 6. Metrics to Track

- loss_combined (should decrease)
- loss_base (constant, frozen model)
- loss_improvement = loss_base - loss_combined (should be positive and growing)
- kl_divergence (should stay bounded)
- bridge3_gamma (learned output scaling)
- bridge3_gate (learned transform mixing)
- field_norm_at_training (are memories accumulating?)
- perturbation_norm (is Bridge 1 producing nonzero output?)
- significance_scores (is significance detector calibrated for training data?)

## 7. Success Criteria

Phase 3 is successful when:
1. loss_improvement > 0 consistently (memories help prediction)
2. Top-K token predictions shift toward semantically relevant tokens
3. Bridge 3 gamma stabilizes at a nonzero value
4. Bridge 1 perturbations cluster by semantic content (not random)
5. KL stays bounded (memory nudges, doesn't hijack)
