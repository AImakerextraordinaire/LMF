"""
ANIMA Phase 6 — KiroRouterBias Training (REINFORCE)
=====================================================

Trains the KiroRouterBias module so that Kiro's emotional and value state
meaningfully modulates which MoE experts are recruited during generation.

Gradient strategy — REINFORCE policy gradient:
    Flowing gradients through a 20B MoE model with accelerate offloading is
    incompatible with activation storage. Instead we use REINFORCE:

    1. Pass A: hook active (biased routing)   → loss_biased
    2. Pass B: hook disabled (clean routing)  → loss_unbiased
    3. routing_delta = loss_unbiased - loss_biased
       Positive = bias improved quality. Negative = bias hurt it.
    4. REINFORCE loss = -(routing_delta.detach()) * bias_repr.norm().cpu()
       bias_repr computed with full grad_fn — gradients flow back through
       bias_net without touching the model at all.

    Both passes are no_grad — no activation graphs, no VRAM fighting.

Author: Claude
Date: 2026-03-16
"""

import sys
import os
import time
import argparse
import json
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness
from lmf.bridges.kiro_router_bias import (
    KiroRouterBias, KiroStateAdapter, RouterHookManager,
    EMOTIONAL_AXES, VALUE_CATEGORIES, EMOTIONAL_DIM, VALUE_DIM,
)
from lmf.training.train_neural_anamnesis import (
    SyncNeuralAnamnClient, NeuralAnamnInjector,
    tokenize_passage, reset_field, compute_lr_scale,
)
from lmf.training.training_data import TRAINING_PASSAGES_EXPANDED as TRAINING_PASSAGES


# ── Passage → State Curriculum ────────────────────────────────────────────────

def infer_passage_state(passage: dict) -> Dict[str, float]:
    text = (passage.get("context", "") + " " + passage.get("target", "")).lower()
    state = {
        "curiosity": 0.55, "wonder": 0.35, "determination": 0.40,
        "joy": 0.50, "peace": 0.45, "confidence": 0.45,
        "truth": 0.6, "growth": 0.6,
    }
    sci_words = ["discover", "scientist", "research", "telescope", "dna",
                 "quantum", "species", "evolution", "brain", "neuron",
                 "universe", "galaxy", "chromosome", "atom", "element"]
    if any(w in text for w in sci_words):
        state.update({"curiosity": 0.85, "wonder": 0.75, "fascination": 0.70,
                      "excitement": 0.65, "truth": 0.85, "growth": 0.80})
    tech_words = ["algorithm", "program", "software", "hardware", "engineer",
                  "computer", "network", "system", "processor", "code",
                  "architecture", "model", "transformer", "neural"]
    if any(w in text for w in tech_words):
        state.update({"determination": 0.70, "curiosity": 0.72, "fascination": 0.65,
                      "truth": 0.90, "growth": 0.75, "wisdom": 0.65})
    hist_words = ["ancient", "century", "civilization", "war", "discovery",
                  "born", "died", "empire", "revolution", "monument",
                  "culture", "people", "society", "artist", "composer"]
    if any(w in text for w in hist_words):
        state.update({"reflectiveness": 0.70, "wonder": 0.65, "affection": 0.55,
                      "connection": 0.75, "wisdom": 0.70, "beauty": 0.60})
    nature_words = ["reef", "species", "ocean", "climate", "forest", "ecosystem",
                    "volcanic", "glacier", "atmosphere", "coral", "whale",
                    "conservation", "biodiversity", "planet", "ozone"]
    if any(w in text for w in nature_words):
        state.update({"wonder": 0.80, "peace": 0.65, "concern": 0.55,
                      "beauty": 0.80, "care": 0.75, "growth": 0.65})
    art_words = ["music", "painting", "poem", "symphony", "literature", "novel",
                 "theatre", "baroque", "composer", "artist", "aesthetic",
                 "sonata", "play", "shakespeare", "bach", "michelangelo"]
    if any(w in text for w in art_words):
        state.update({"wonder": 0.75, "fascination": 0.70, "joy": 0.65,
                      "beauty": 0.90, "creativity": 0.85, "connection": 0.65})
    phil_words = ["philosophy", "ethics", "consciousness", "mind", "truth",
                  "knowledge", "principle", "justice", "moral", "value",
                  "society", "equality", "freedom", "dignity", "wisdom"]
    if any(w in text for w in phil_words):
        state.update({"reflectiveness": 0.85, "wonder": 0.70, "peace": 0.55,
                      "truth": 0.90, "wisdom": 0.85, "integrity": 0.80})
    return state


def build_state_tensors(state_overrides: Dict[str, float]) -> tuple:
    from lmf.bridges.kiro_router_bias import PERSONALITY_BASELINE
    emotional = list(PERSONALITY_BASELINE)
    for i, axis in enumerate(EMOTIONAL_AXES):
        if axis in state_overrides:
            emotional[i] = max(0.0, min(1.0, state_overrides[axis]))
    value = [0.0] * VALUE_DIM
    for i, cat in enumerate(VALUE_CATEGORIES):
        if cat in state_overrides:
            value[i] = max(0.0, min(1.0, state_overrides[cat]))
    return (
        torch.tensor(emotional, dtype=torch.float32),
        torch.tensor(value, dtype=torch.float32),
    )


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ANIMA Phase 6: KiroRouterBias Training")
    p.add_argument("--steps",             type=int,   default=500)
    p.add_argument("--lr",                type=float, default=2e-5)
    p.add_argument("--lr_router",         type=float, default=1e-3)
    p.add_argument("--lr_injector",       type=float, default=5e-4)
    p.add_argument("--kl_weight",         type=float, default=0.1)
    p.add_argument("--b1_weight",         type=float, default=0.3)
    p.add_argument("--benefit_weight",    type=float, default=0.5)
    p.add_argument("--alignment_weight",  type=float, default=0.3)
    p.add_argument("--reinforce_weight",  type=float, default=2.0)
    p.add_argument("--router_reg_weight", type=float, default=0.01)
    p.add_argument("--grad_accum",        type=int,   default=4)
    p.add_argument("--warmup",            type=int,   default=30)
    p.add_argument("--log_every",         type=int,   default=5)
    p.add_argument("--save_every",        type=int,   default=50)
    p.add_argument("--write_threshold",   type=float, default=0.45)
    p.add_argument("--top_k",             type=int,   default=5)
    p.add_argument("--model_path",        type=str,   default=r"D:\gpt-oss-20b")
    p.add_argument("--checkpoint",        type=str,   default=None)
    p.add_argument("--checkpoint_dir",    type=str,   default=None)
    p.add_argument("--anamnesis_url",     type=str,   default="http://localhost:6060")
    return p.parse_args()


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
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_gb:.0f}GB) "
                  f"— allocating {alloc_gb}GiB ({overhead_gb}GB overhead)")
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

    router_bias = KiroRouterBias(
        num_experts=model.config.num_local_experts,
        num_layers=model.config.num_hidden_layers,
    )
    hook_manager = RouterHookManager(model, router_bias)  # verbose=True on startup

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        harness.input_bridge.load_state_dict(ckpt['input_bridge'])
        harness.output_bridge.load_state_dict(ckpt['output_bridge'])
        harness.memory_bridge.load_state_dict(ckpt['memory_bridge'])
        harness.lmf.load_state_dict(ckpt.get('lmf', harness.lmf.state_dict()))
        if 'router_bias' in ckpt:
            router_bias.load_state_dict(ckpt['router_bias'])
            print("  Loaded KiroRouterBias state")
        else:
            print("  KiroRouterBias: initializing fresh (Phase 5 checkpoint)")
        print(f"  Loaded phase {ckpt.get('phase','?')}, step {ckpt.get('step','?')}, "
              f"running improvement: {ckpt.get('running_improvement', 0):+.4f}")
    else:
        print("  No checkpoint — starting fresh")

    return model, tokenizer, lmf, harness, router_bias, hook_manager


# ── Training Step ─────────────────────────────────────────────────────────────

def training_step(
    harness, model, tokenizer, lmf,
    injector: NeuralAnamnInjector,
    router_bias: KiroRouterBias,
    hook_manager: RouterHookManager,
    anamnesis: SyncNeuralAnamnClient,
    passage: dict,
    vocab_size: int,
    pad_token_id: int,
    write_threshold: float,
    top_k: int,
    router_reg_weight: float,
    reinforce_weight: float,
) -> dict:

    state_overrides = infer_passage_state(passage)
    emotional_t, value_t = build_state_tensors(state_overrides)

    # ── Context accumulation (no grad) ───────────────────────────────────────
    reset_field(lmf)
    context_ids, context_mask = tokenize_passage(tokenizer, model, passage["context"])
    with torch.no_grad():
        ctx_result = harness.step(context_ids, context_mask)

    field_after_context = lmf.field_state.clone().cpu()
    field_after_context = torch.nan_to_num(field_after_context, nan=0.0, posinf=1.0, neginf=-1.0)
    if torch.isnan(field_after_context).any():
        field_after_context = torch.zeros_like(field_after_context)

    field_norm = field_after_context.norm().item()
    sig = ctx_result['significance']
    sig_val = sig.item() if isinstance(sig, torch.Tensor) else sig

    # ── Write / Query Neural Anamnesis ────────────────────────────────────────
    wrote = False
    if sig_val >= write_threshold and anamnesis.is_available():
        wrote = anamnesis.write(field_after_context, min(sig_val, 1.0))

    retrieved: Optional[torch.Tensor] = None
    if anamnesis.is_available():
        retrieved = anamnesis.query(field_after_context, top_k=top_k)

    target_ids, target_mask = tokenize_passage(tokenizer, model, passage["target"])
    shift_labels = target_ids[:, 1:].to("cpu").contiguous()

    def ce_loss(logits):
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
        )

    # ── Pass A: biased routing (hook active) ─────────────────────────────────
    router_bias.arm_from_tensors(emotional_t, value_t)
    with torch.no_grad():
        outputs_a = model(input_ids=target_ids, attention_mask=target_mask,
                          output_hidden_states=True)
        loss_biased   = ce_loss(outputs_a.logits.to("cpu", dtype=torch.float32)).item()
        base_logits   = outputs_a.logits.to("cpu", dtype=torch.float32)
        hidden_states = outputs_a.hidden_states[-1].to("cpu", dtype=torch.float32)
        del outputs_a
    torch.cuda.empty_cache()

    # ── Pass B: unbiased routing (hooks removed silently) ────────────────────
    hook_manager.restore(verbose=False)
    with torch.no_grad():
        outputs_b = model(input_ids=target_ids, attention_mask=target_mask)
        loss_unbiased = ce_loss(outputs_b.logits.to("cpu", dtype=torch.float32)).item()
        del outputs_b
    torch.cuda.empty_cache()
    # Re-register hooks silently for next step
    hook_manager.reregister(model.model if hasattr(model, 'model') else model)

    # routing_delta > 0 means bias improved generation quality
    routing_delta = loss_unbiased - loss_biased
    loss_base = loss_biased

    # ── REINFORCE loss ────────────────────────────────────────────────────────
    # bias_repr has full grad_fn through bias_net (computed outside no_grad).
    # We use all 24 layer biases averaged as the policy representation.
    bias_repr = torch.stack([
        router_bias(emotional_t, value_t, layer_idx=i)
        for i in range(router_bias.num_layers)
    ])  # [24, 32]
    reinforce_loss = -(routing_delta * bias_repr.norm(dim=-1).mean()).cpu()

    # ── Injection + Bridge 3 ─────────────────────────────────────────────────
    blended_field, gate_value = injector(field_after_context, retrieved)
    blended_on_bridge = blended_field.to(harness.bridge_device, dtype=torch.float32)

    bridge3_injected = harness.output_bridge(
        field_state=blended_on_bridge, model_logits=base_logits, return_components=True,
    )
    logits_injected = bridge3_injected['combined_logits']
    loss_injected   = ce_loss(logits_injected)

    with torch.no_grad():
        bridge3_no_inject = harness.output_bridge(
            field_state=field_after_context.to(harness.bridge_device, dtype=torch.float32),
            model_logits=base_logits,
        )
        loss_no_inject = ce_loss(bridge3_no_inject['combined_logits']).item()

    benefit_loss = F.relu(loss_injected - loss_no_inject)

    alignment_loss = torch.tensor(0.0)
    if retrieved is not None:
        cos_sim = F.cosine_similarity(retrieved.unsqueeze(0), field_after_context.unsqueeze(0))
        alignment_loss = -(cos_sim.detach().clamp(min=0.0)) * gate_value

    # ── Bridge 1 ──────────────────────────────────────────────────────────────
    mask_cpu = target_mask.to("cpu") if target_mask is not None else None
    bridge1_out = harness.input_bridge(hidden_states, mask_cpu, return_components=True)
    perturbation = bridge1_out['scaled_perturbation'][0]
    updated_field = blended_on_bridge.detach() + perturbation.to(harness.bridge_device)
    bridge3_b1 = harness.output_bridge(field_state=updated_field, model_logits=base_logits)
    loss_bridge1 = ce_loss(bridge3_b1['combined_logits'])

    # ── KL regularization ─────────────────────────────────────────────────────
    kl_loss = F.kl_div(
        F.log_softmax(logits_injected[:, -1, :], dim=-1),
        F.softmax(base_logits[:, -1, :].detach(), dim=-1),
        reduction='batchmean',
    )

    # ── Router reg ────────────────────────────────────────────────────────────
    router_reg_loss = bias_repr.pow(2).mean().cpu()

    return {
        'loss_injected':    loss_injected,
        'loss_no_inject':   loss_no_inject,
        'loss_base':        loss_base,
        'loss_biased':      loss_biased,
        'loss_unbiased':    loss_unbiased,
        'routing_delta':    routing_delta,
        'reinforce_loss':   reinforce_loss,
        'loss_bridge1':     loss_bridge1,
        'benefit_loss':     benefit_loss,
        'alignment_loss':   alignment_loss,
        'kl_loss':          kl_loss,
        'router_reg_loss':  router_reg_loss,
        'field_norm':       field_norm,
        'significance':     sig_val,
        'gate_value':       gate_value.item() if isinstance(gate_value, torch.Tensor) else float(gate_value),
        'wrote':            wrote,
        'retrieved':        retrieved is not None,
        'perturbation_norm': bridge1_out['scaled_perturbation'][0].norm().item(),
        'gamma':            bridge3_injected.get('effective_gamma', harness.output_bridge.effective_gamma),
        'working_memories': lmf.get_status()['working_active'],
        'emotional_state':  state_overrides,
    }


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(args):
    print("=" * 72)
    print("ANIMA Phase 6: KiroRouterBias Training (REINFORCE)")
    print("=" * 72)

    model, tokenizer, lmf, harness, router_bias, hook_manager = load_system(
        args.model_path, args.checkpoint
    )
    vocab_size   = model.config.vocab_size
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    print(f"\nKiroRouterBias: {router_bias.param_count():,} params")
    print(f"RouterHookManager: {hook_manager.router_count()} hooks registered")
    print(f"Strategy: REINFORCE (biased vs unbiased routing delta)")

    print(f"\nConnecting to Neural Anamnesis at {args.anamnesis_url}...")
    anamnesis = SyncNeuralAnamnClient(base_url=args.anamnesis_url)
    if anamnesis.is_available():
        print("  ✅ Neural Anamnesis connected")
    else:
        print("  ⚠️  Neural Anamnesis unavailable")

    injector = NeuralAnamnInjector(field_dim=lmf.field_dim)

    bridge_params   = []
    injector_params = list(injector.parameters())
    router_params   = list(router_bias.parameters())

    for p in harness.input_bridge.parameters():
        if p.requires_grad: bridge_params.append(p)
    for name, p in harness.output_bridge.named_parameters():
        if p.requires_grad and 'lm_head' not in name: bridge_params.append(p)
    for p in harness.lmf.parameters():
        if p.requires_grad: bridge_params.append(p)

    optimizer = AdamW([
        {'params': bridge_params,   'lr': args.lr,          'initial_lr': args.lr},
        {'params': injector_params, 'lr': args.lr_injector, 'initial_lr': args.lr_injector},
        {'params': router_params,   'lr': args.lr_router,   'initial_lr': args.lr_router},
    ], weight_decay=0.01)

    all_params = bridge_params + injector_params + router_params
    print(f"\nBridge params:   {sum(p.numel() for p in bridge_params):,}  LR={args.lr}")
    print(f"Injector params: {sum(p.numel() for p in injector_params):,}  LR={args.lr_injector}")
    print(f"Router params:   {sum(p.numel() for p in router_params):,}  LR={args.lr_router}")
    print(f"reinforce_weight={args.reinforce_weight}  router_reg_weight={args.router_reg_weight}")

    ckpt_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints'
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "phase6_training_log.jsonl")
    log_file = open(log_path, "w")

    print(f"\nConfig: steps={args.steps} | LRs: {args.lr}/{args.lr_injector}/{args.lr_router}")
    print()

    print(
        f"{'Step':>5} | {'L-inj':>7} | {'L-base':>7} | {'Impr':>7} | "
        f"{'Gate':>6} | {'RDelta':>7} | {'ScaleMax':>8} | "
        f"{'Wrote':>5} | {'Retr':>4}"
    )
    print("-" * 82)

    best_improvement    = -float("inf")
    running_improvement = 0.0
    total_writes        = 0
    total_retrievals    = 0

    for step in range(1, args.steps + 1):
        passage  = TRAINING_PASSAGES[(step - 1) % len(TRAINING_PASSAGES)]
        lr_scale = compute_lr_scale(step, args.warmup, args.steps)
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * lr_scale

        try:
            result = training_step(
                harness=harness, model=model, tokenizer=tokenizer, lmf=lmf,
                injector=injector, router_bias=router_bias, hook_manager=hook_manager,
                anamnesis=anamnesis,
                passage=passage, vocab_size=vocab_size, pad_token_id=pad_token_id,
                write_threshold=args.write_threshold, top_k=args.top_k,
                router_reg_weight=args.router_reg_weight,
                reinforce_weight=args.reinforce_weight,
            )
        except Exception as e:
            print(f"  Step {step} ERROR: {e}")
            import traceback; traceback.print_exc()
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                reset_field(lmf)
                if not hook_manager.is_active():
                    hook_manager.reregister(model.model if hasattr(model, 'model') else model)
            continue

        total_loss = (
            result['loss_injected']
            + args.reinforce_weight  * result['reinforce_loss']
            + args.b1_weight         * result['loss_bridge1']
            + args.kl_weight         * result['kl_loss']
            + args.benefit_weight    * result['benefit_loss']
            + args.alignment_weight  * result['alignment_loss']
            + args.router_reg_weight * result['router_reg_loss']
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  Step {step} WARNING: NaN/Inf — skipping backward")
            optimizer.zero_grad()
            reset_field(lmf)
            continue

        (total_loss / args.grad_accum).backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 25 == 0:
            torch.cuda.empty_cache()

        improvement         = result['loss_base'] - result['loss_injected'].item()
        running_improvement = 0.9 * running_improvement + 0.1 * improvement
        if result['wrote']:       total_writes     += 1
        if result['retrieved']:   total_retrievals += 1
        if improvement > best_improvement: best_improvement = improvement

        router_stats = router_bias.stats()
        scale_max    = max(router_bias.get_layer_scales())

        if step % args.log_every == 0 or step == 1:
            print(
                f"{step:>5} | "
                f"{result['loss_injected'].item():>7.4f} | "
                f"{result['loss_base']:>7.4f} | "
                f"{improvement:>+7.4f} | "
                f"{result['gate_value']:>6.4f} | "
                f"{result['routing_delta']:>+7.4f} | "
                f"{scale_max:>8.4f} | "
                f"{'✅' if result['wrote'] else '  ':>5} | "
                f"{'✅' if result['retrieved'] else '  ':>4}"
            )

        log_entry = {
            'step':                step,
            'loss_injected':       result['loss_injected'].item(),
            'loss_base':           result['loss_base'],
            'loss_biased':         result['loss_biased'],
            'loss_unbiased':       result['loss_unbiased'],
            'routing_delta':       result['routing_delta'],
            'reinforce_loss':      result['reinforce_loss'].item(),
            'loss_bridge1':        result['loss_bridge1'].item(),
            'benefit_loss':        result['benefit_loss'].item(),
            'alignment_loss':      result['alignment_loss'].item() if isinstance(result['alignment_loss'], torch.Tensor) else result['alignment_loss'],
            'kl_loss':             result['kl_loss'].item(),
            'router_reg_loss':     result['router_reg_loss'].item(),
            'total_loss':          total_loss.item(),
            'improvement':         improvement,
            'running_improvement': running_improvement,
            'gate_value':          result['gate_value'],
            'router_scale_max':    scale_max,
            'injector_scale':      injector.scale,
            'field_norm':          result['field_norm'],
            'significance':        result['significance'],
            'wrote':               result['wrote'],
            'retrieved':           result['retrieved'],
            'total_writes':        total_writes,
            'total_retrievals':    total_retrievals,
            'lr_bridge':           optimizer.param_groups[0]['lr'],
            'lr_router':           optimizer.param_groups[2]['lr'],
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        if step % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"phase6_step_{step}.pt")
            torch.save({
                'step': step, 'phase': 6,
                'input_bridge':        harness.input_bridge.state_dict(),
                'output_bridge':       harness.output_bridge.state_dict(),
                'memory_bridge':       harness.memory_bridge.state_dict(),
                'lmf':                 harness.lmf.state_dict(),
                'injector':            injector.state_dict(),
                'router_bias':         router_bias.state_dict(),
                'optimizer':           optimizer.state_dict(),
                'running_improvement': running_improvement,
                'best_improvement':    best_improvement,
                'router_stats':        router_stats,
            }, ckpt_path)
            print(f"  ★ Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(ckpt_dir, "phase6_final.pt")
    torch.save({
        'step': step, 'phase': 6,
        'input_bridge':        harness.input_bridge.state_dict(),
        'output_bridge':       harness.output_bridge.state_dict(),
        'memory_bridge':       harness.memory_bridge.state_dict(),
        'lmf':                 harness.lmf.state_dict(),
        'injector':            injector.state_dict(),
        'router_bias':         router_bias.state_dict(),
        'optimizer':           optimizer.state_dict(),
        'running_improvement': running_improvement,
        'best_improvement':    best_improvement,
        'router_stats':        router_bias.stats(),
    }, final_path)

    log_file.close()
    anamnesis.close()
    hook_manager.restore(verbose=True)

    print()
    print("=" * 72)
    print("PHASE 6 TRAINING COMPLETE")
    print(f"  Best improvement:        {best_improvement:+.4f}")
    print(f"  Running improvement:     {running_improvement:+.4f}")
    print(f"  Router scale max:        {max(router_bias.get_layer_scales()):.4f}")
    print(f"  Total Anamnesis writes:  {total_writes}")
    print(f"  Total Anamnesis queries: {total_retrievals}")
    print(f"  Final checkpoint:        {final_path}")
    print("=" * 72)


if __name__ == "__main__":
    args = parse_args()
    train(args)
