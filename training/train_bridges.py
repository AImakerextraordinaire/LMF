"""
ANIMA Phase 3 - Bridge Training Script
Memory-Conditioned Language Modeling

Teaches bridges that memories from context should help predict target text.
Base model is FROZEN. Only bridges + LMF are trainable.

Training loop:
  1. Context accumulation (no grad): process first 60% of document
  2. Training step (with grad): compute loss on last 40%
     - Bridge 3 loss: field_state -> logit_bias helps next-token prediction
     - Bridge 1 loss: perturbation improves field -> better logit bias
     - KL regularization: memory nudges, doesn't hijack

Run:
    cd ANIMA repo root
    python lmf/training/train_bridges.py --steps 100 --lr 1e-4
"""

import sys
import os
import time
import argparse
import json

# GPU selection: set CUDA_VISIBLE_DEVICES before launching (see run_training.bat)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GPU selection handled at top of file via CUDA_VISIBLE_DEVICES
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness


# ============================================================
# Training data: simple multi-segment passages
# ============================================================

from lmf.training.training_data import TRAINING_PASSAGES_EXPANDED as TRAINING_PASSAGES

# Original 8 passages kept as fallback
TRAINING_PASSAGES_ORIGINAL = [
    # Each passage has context that should help predict the target
    {
        "context": "The ancient city of Petra was carved into rose-red cliffs by the Nabataean people over two thousand years ago. Located in modern-day Jordan, Petra served as a crucial trading hub connecting Arabia, Egypt, and the Mediterranean. The Nabataeans were master hydraulic engineers who built an elaborate system of dams, cisterns, and water channels to sustain their desert city.",
        "target": "Today, Petra remains one of the most remarkable archaeological sites in the world. The Treasury, carved directly into the sandstone cliff face, is perhaps the most iconic structure from this ancient Nabataean civilization. Visitors can still see traces of the sophisticated water management systems that once sustained life in this desert trading center."
    },
    {
        "context": "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. She moved to Paris to study physics and mathematics at the Sorbonne, where she met Pierre Curie. Together, they discovered two new elements: polonium, named after Marie's homeland of Poland, and radium. Marie became the first woman to win a Nobel Prize.",
        "target": "Her groundbreaking research on radioactivity earned her not one but two Nobel Prizes, making her the first person to win Nobel Prizes in two different sciences. The element polonium, which she named in honor of her native Poland, was discovered through her painstaking work isolating radioactive compounds. Marie Curie's legacy in physics and chemistry continues to inspire scientists worldwide."
    },
    {
        "context": "The octopus is one of the most intelligent invertebrates on Earth. It has three hearts, blue blood, and eight arms lined with suckers. Each arm contains a cluster of neurons, giving octopuses a distributed nervous system. They can change color and texture in milliseconds to camouflage themselves, and have been observed using tools like coconut shells for shelter.",
        "target": "Scientists studying octopus cognition have been amazed by their problem-solving abilities. Their distributed nervous system, with neural clusters in each of their eight arms, allows for remarkably complex behavior. These creatures with three hearts and blue blood can open jars, navigate mazes, and even recognize individual human faces, demonstrating intelligence that rivals many vertebrates."
    },
    {
        "context": "The Great Barrier Reef stretches over 2,300 kilometers along the northeast coast of Australia. It is the largest living structure on Earth, visible from space. The reef is home to over 1,500 species of fish, 400 types of coral, and countless other marine organisms. Rising ocean temperatures pose the greatest threat to this ecosystem through coral bleaching.",
        "target": "Conservation efforts for the Great Barrier Reef have intensified as ocean warming accelerates coral bleaching events along Australia's northeastern coastline. The reef, which spans more than two thousand kilometers and supports extraordinary biodiversity including thousands of fish species, faces an uncertain future. Protecting this massive living structure, the largest on Earth, requires global action on climate change."
    },
    {
        "context": "Johann Sebastian Bach composed over a thousand works during his lifetime, including the Brandenburg Concertos, the Mass in B minor, and The Well-Tempered Clavier. Working in Germany during the Baroque period, Bach served as a church musician and court composer. His mastery of counterpoint and harmony laid the foundation for Western classical music theory.",
        "target": "The influence of Bach on Western music cannot be overstated. His works, from the Brandenburg Concertos to The Well-Tempered Clavier, established fundamental principles of harmony and counterpoint that composers still study today. This Baroque-era church musician and court composer from Germany created a body of over a thousand compositions that represents one of the greatest achievements in the history of music."
    },
    {
        "context": "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. The prefrontal cortex handles executive function and decision-making, while the hippocampus is crucial for memory formation and spatial navigation. During sleep, the brain consolidates memories by replaying experiences and transferring them from short-term to long-term storage.",
        "target": "Memory consolidation during sleep is one of the most fascinating processes in neuroscience. The hippocampus replays recently formed memories while we sleep, gradually transferring important experiences to long-term storage across the cortex. This process, occurring in a brain with billions of neurons and trillions of synaptic connections, helps explain why adequate sleep is essential for learning and memory."
    },
    {
        "context": "Rust is a systems programming language focused on safety, speed, and concurrency. Created by Graydon Hoare at Mozilla, Rust uses an ownership system with borrowing rules enforced at compile time to prevent data races and memory leaks. Unlike C and C++, Rust achieves memory safety without garbage collection.",
        "target": "The ownership model that defines Rust has made it increasingly popular for systems where reliability is critical. By enforcing borrowing rules at compile time rather than relying on garbage collection, Rust eliminates entire categories of bugs including data races and memory leaks. This systems language, originally developed at Mozilla, proves that safety and performance need not be opposing goals."
    },
    {
        "context": "Mount Everest stands at 8,849 meters above sea level, making it the highest point on Earth. Located in the Himalayas on the border of Nepal and Tibet, it was first summited by Edmund Hillary and Tenzing Norgay in 1953. The mountain is known by several names: Chomolungma in Tibetan and Sagarmatha in Nepali.",
        "target": "Since Hillary and Norgay first reached its summit in 1953, thousands of climbers have attempted to scale this Himalayan peak. Known as Sagarmatha in Nepal and Chomolungma in Tibet, the mountain that straddles their border rises nearly nine thousand meters above sea level. Mount Everest, the highest point on Earth, continues to draw adventurers from around the world."
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="ANIMA Phase 3 Bridge Training")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL regularization weight")
    parser.add_argument("--b1_weight", type=float, default=0.5, help="Bridge 1 loss weight")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--log_every", type=int, default=5, help="Log every N steps")
    parser.add_argument("--bridge2", action="store_true", help="Enable Bridge 2 gradient training (Phase 3b)")
    parser.add_argument("--contrastive", action="store_true", help="Enable contrastive loss (penalize wrong-memory improvement)")
    parser.add_argument("--contrastive_weight", type=float, default=0.3, help="Weight for contrastive loss term")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--model_path", type=str, default=r"D:\gpt-oss-20b")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    return parser.parse_args()


def load_system(model_path):
    """Load model, tokenizer, LMF, and harness."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    t0 = time.time()
    # Detect GPU memory and allocate accordingly
    max_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            # Reserve ~2GB for overhead
            alloc_gb = int(total_gb - 2)
            max_mem[i] = f"{alloc_gb}GiB"
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} - allocating {alloc_gb}GiB")
    max_mem["cpu"] = "80GiB"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        max_memory=max_mem,
        offload_folder="offload_temp",
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    config = gpt_oss_20b_config()
    config.device = "cpu"
    lmf = LivingMemoryField(config)
    harness = BridgeHarness(model=model, lmf=lmf, bridge_device="cpu")

    return model, tokenizer, lmf, harness


def tokenize_passage(tokenizer, model, text, max_length=512):
    """Tokenize text with chat template."""
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, return_dict=True,
        max_length=max_length, truncation=True,
    )
    ids = inputs["input_ids"].to(model.device)
    mask = inputs.get("attention_mask")
    if mask is not None:
        mask = mask.to(model.device)
    return ids, mask


def reset_field(lmf):
    """Reset LMF to clean state."""
    lmf.field_state.zero_()
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)


def get_trainable_params(harness):
    """Collect all trainable parameters from bridges + LMF."""
    params = []

    # Bridge 1 (Input)
    for p in harness.input_bridge.parameters():
        if p.requires_grad:
            params.append(p)

    # Bridge 3 (Output) - only new params, not lm_head
    for name, p in harness.output_bridge.named_parameters():
        if p.requires_grad and 'lm_head' not in name:
            params.append(p)

    # LMF trainable params (A matrix, layer weights, etc.)
    for p in harness.lmf.parameters():
        if p.requires_grad:
            params.append(p)

    # Bridge 2 (Memory) - only if Phase 3b enabled
    if hasattr(harness, '_train_bridge2') and harness._train_bridge2:
        for p in harness.memory_bridge.parameters():
            if p.requires_grad:
                params.append(p)
        print(f"  Bridge 2 training ENABLED ({sum(p.numel() for p in harness.memory_bridge.parameters()):,} params)")

    total = sum(p.numel() for p in params)
    print(f"Trainable parameters: {total:,}")
    return params


def compute_lr_scale(step, warmup_steps, total_steps):
    """Linear warmup, cosine decay."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())


def training_step(
    harness, model, tokenizer, lmf,
    passage, vocab_size, pad_token_id,
    distractor_passage=None,
):
    """
    One training step: context accumulation + loss computation.
    
    If distractor_passage is provided, also computes contrastive loss:
    penalizes improvement from wrong-memory conditioning.

    Returns dict with losses and metrics.
    """
    # === Phase 0: Compute distractor loss FIRST (if contrastive enabled) ===
    # Must happen before the main forward pass to avoid in-place graph conflicts
    # when accelerate moves tensors between devices during model forward calls.
    distractor_loss_val = None
    if distractor_passage is not None:
        with torch.no_grad():
            reset_field(lmf)
            dist_ctx_ids, dist_ctx_mask = tokenize_passage(tokenizer, model, distractor_passage["context"])
            harness.step(dist_ctx_ids, dist_ctx_mask)
            
            # Now run target with WRONG memory
            target_ids_d, target_mask_d = tokenize_passage(tokenizer, model, passage["target"])
            base_out_d = model(input_ids=target_ids_d, attention_mask=target_mask_d)
            base_logits_d = base_out_d.logits.to(device="cpu", dtype=torch.float32)
            
            dist_field = lmf.field_state.clone()
            dist_b3 = harness.output_bridge(
                field_state=dist_field,
                model_logits=base_logits_d,
            )
            dist_combined = dist_b3['combined_logits']
            shift_dist = dist_combined[:, :-1, :].contiguous()
            shift_labels_d = target_ids_d[:, 1:].to(device="cpu").contiguous()
            
            distractor_loss_val = F.cross_entropy(
                shift_dist.view(-1, vocab_size),
                shift_labels_d.view(-1),
                ignore_index=pad_token_id,
            ).item()

    # === Phase A: Context Accumulation (no grad) ===
    reset_field(lmf)

    context_ids, context_mask = tokenize_passage(tokenizer, model, passage["context"])

    with torch.no_grad():
        harness.step(context_ids, context_mask)

    field_norm = lmf.field_state.norm().item()
    status = lmf.get_status()

    # === Phase B: Training Step (with grad on bridges) ===
    target_ids, target_mask = tokenize_passage(tokenizer, model, passage["target"])
    shift_labels = target_ids[:, 1:].to(device="cpu").contiguous()

    train_bridge2 = getattr(harness, '_train_bridge2', False)

    if train_bridge2:
        # === Phase 3b: Full pipeline with Bridge 2 gradients ===
        # Run through harness with grad enabled - Bridge 2 hooks inject with gradients
        result = harness.step(
            target_ids, target_mask,
            return_debug=True,
            enable_grad=True,
        )

        # combined_logits includes BOTH Bridge 2 (mid-layer injection) and Bridge 3 (logit bias)
        combined_logits = result['logits'].to(device="cpu", dtype=torch.float32)
        model_logits = result['model_logits'].to(device="cpu", dtype=torch.float32)
        hidden_states = result.get('hidden_states')
        if hidden_states is not None:
            hidden_states = hidden_states.to(device="cpu", dtype=torch.float32)

        # Base loss: model without ANY bridge influence (for comparison only)
        with torch.no_grad():
            base_outputs = model(input_ids=target_ids, attention_mask=target_mask)
            base_logits = base_outputs.logits.to(device="cpu", dtype=torch.float32)
            shift_base = base_logits[:, :-1, :].contiguous()
            loss_base = F.cross_entropy(
                shift_base.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id,
            )

        # Loss 1: Combined (Bridge 2 + Bridge 3) memory-conditioned LM
        shift_combined = combined_logits[:, :-1, :].contiguous()
        loss_combined = F.cross_entropy(
            shift_combined.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
        )

        # Loss 2: Bridge 1 perturbation usefulness
        field_snapshot = lmf.field_state.clone().detach()
        if hidden_states is not None:
            mask_cpu = target_mask.to("cpu") if target_mask is not None else None
            bridge1_out = harness.input_bridge(hidden_states.detach(), mask_cpu, return_components=True)
            perturbation = bridge1_out['scaled_perturbation'][0]
            updated_field = field_snapshot + perturbation
            bridge3_updated = harness.output_bridge(
                field_state=updated_field,
                model_logits=base_logits,
            )
            shift_updated = bridge3_updated['combined_logits'][:, :-1, :].contiguous()
            loss_bridge1 = F.cross_entropy(
                shift_updated.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id,
            )
        else:
            loss_bridge1 = loss_combined  # Fallback

        # Loss 3: KL regularization (memory nudges, not hijacks)
        kl_loss = F.kl_div(
            F.log_softmax(combined_logits[:, -1, :], dim=-1),
            F.softmax(base_logits[:, -1, :].detach(), dim=-1),
            reduction='batchmean',
        )

        bridge3_out = {'effective_gamma': harness.output_bridge.effective_gamma, 'gate_value': 0.0}
        bridge1_sig = result.get('significance', torch.tensor(0.0))
        bridge1_pnorm = torch.tensor(0.0)
        logit_bias_std = 0.0

    else:
        # === Phase 3a: Original path (no Bridge 2 gradients) ===
        # Get base model output (frozen)
        with torch.no_grad():
            outputs = model(
                input_ids=target_ids,
                attention_mask=target_mask,
                output_hidden_states=True,
            )
            base_logits = outputs.logits.to(device="cpu", dtype=torch.float32)
            hidden_states = outputs.hidden_states[-1].to(device="cpu", dtype=torch.float32)

        # --- Loss 1: Bridge 3 (Memory-Conditioned LM) ---
        field_snapshot = lmf.field_state.clone().detach()

        bridge3_out = harness.output_bridge(
            field_state=field_snapshot,
            model_logits=base_logits,
            return_components=True,
        )
        combined_logits = bridge3_out['combined_logits']

        # Shift for next-token prediction
        shift_combined = combined_logits[:, :-1, :].contiguous()
        shift_base = base_logits[:, :-1, :].contiguous()

        loss_combined = F.cross_entropy(
            shift_combined.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
        )

        with torch.no_grad():
            loss_base = F.cross_entropy(
                shift_base.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id,
            )

        # --- Loss 2: Bridge 1 (Perturbation Usefulness) ---
        mask_cpu = target_mask.to("cpu") if target_mask is not None else None
        bridge1_out = harness.input_bridge(hidden_states, mask_cpu, return_components=True)
        perturbation = bridge1_out['scaled_perturbation'][0]

        updated_field = field_snapshot + perturbation
        bridge3_updated = harness.output_bridge(
            field_state=updated_field,
            model_logits=base_logits,
        )
        updated_logits = bridge3_updated['combined_logits']

        shift_updated = updated_logits[:, :-1, :].contiguous()
        loss_bridge1 = F.cross_entropy(
            shift_updated.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
        )

        # --- Loss 3: KL Regularization ---
        kl_loss = F.kl_div(
            F.log_softmax(combined_logits[:, -1, :], dim=-1),
            F.softmax(base_logits[:, -1, :], dim=-1),
            reduction='batchmean',
        )

        bridge1_sig = bridge1_out['significance']
        bridge1_pnorm = perturbation.norm()
        logit_bias_std = bridge3_out['logit_bias'][0].std().item()

    # --- Loss 4: Contrastive (matched memory should beat distractor memory) ---
    # Distractor loss was pre-computed in Phase 0 (before any graph was built).
    # Now we just compare: penalize when matched loss > distractor loss.
    # Gradients flow through loss_combined only (matched path).
    contrastive_loss = torch.tensor(0.0)
    if distractor_loss_val is not None:
        contrastive_loss = F.relu(loss_combined - distractor_loss_val)

    return {
        'loss_combined': loss_combined,
        'loss_base': loss_base.item() if isinstance(loss_base, torch.Tensor) else loss_base,
        'loss_bridge1': loss_bridge1,
        'kl_loss': kl_loss,
        'contrastive_loss': contrastive_loss,
        'field_norm': field_norm,
        'working_memories': status['working_active'],
        'significance': bridge1_sig.item() if isinstance(bridge1_sig, torch.Tensor) else bridge1_sig,
        'gamma': bridge3_out.get('effective_gamma', 0.0) if isinstance(bridge3_out, dict) else bridge3_out,
        'gate': bridge3_out.get('gate_value', 0.0) if isinstance(bridge3_out, dict) else 0.0,
        'perturbation_norm': bridge1_pnorm.item() if isinstance(bridge1_pnorm, torch.Tensor) else bridge1_pnorm,
        'logit_bias_std': logit_bias_std if isinstance(logit_bias_std, float) else 0.0,
    }


def train(args):
    """Main training loop."""
    print("=" * 70)
    print("ANIMA Phase 3: Bridge Training")
    print("=" * 70)

    # Load system
    model, tokenizer, lmf, harness = load_system(args.model_path)

    vocab_size = model.config.vocab_size
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Setup optimizer
    params = get_trainable_params(harness)
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)

    # Checkpoint dir
    ckpt_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints'
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training log
    log_path = os.path.join(ckpt_dir, "training_log.jsonl")
    log_file = open(log_path, "w")

    # Phase 3b: Enable Bridge 2 training
    harness._train_bridge2 = args.bridge2
    if args.bridge2:
        print("\n  *** PHASE 3b: Bridge 2 gradient training ENABLED ***")
    
    print(f"\nTraining config:")
    print(f"  Steps: {args.steps}")
    print(f"  LR: {args.lr}")
    print(f"  KL weight: {args.kl_weight}")
    print(f"  Bridge 1 weight: {args.b1_weight}")
    print(f"  Grad accumulation: {args.grad_accum}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Passages: {len(TRAINING_PASSAGES)}")
    print(f"  Bridge 2: {'ENABLED' if args.bridge2 else 'disabled'}")
    print(f"  Contrastive: {'ENABLED (weight=' + str(args.contrastive_weight) + ')' if args.contrastive else 'disabled'}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print()

    # Training header
    print(f"{'Step':>5} | {'Loss':>7} | {'Base':>7} | {'Impr':>7} | {'B1':>7} | {'KL':>7} | {'Norm':>6} | {'Mem':>3} | {'Gamma':>6} | {'PNorm':>7}")
    print(f"{'-'*5} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*3} | {'-'*6} | {'-'*7}")

    best_improvement = -float('inf')
    running_improvement = 0.0

    for step in range(1, args.steps + 1):
        # Select passage (cycle through)
        passage_idx = (step - 1) % len(TRAINING_PASSAGES)
        passage = TRAINING_PASSAGES[passage_idx]
        
        # Select distractor (different passage) for contrastive loss
        distractor = None
        if args.contrastive:
            import random
            dist_idx = random.choice([i for i in range(len(TRAINING_PASSAGES)) if i != passage_idx])
            distractor = TRAINING_PASSAGES[dist_idx]

        # Compute LR with warmup + cosine decay
        lr_scale = compute_lr_scale(step, args.warmup, args.steps)
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_scale

        # Forward pass
        try:
            result = training_step(
                harness, model, tokenizer, lmf,
                passage, vocab_size, pad_token_id,
                distractor_passage=distractor,
            )
        except Exception as e:
            print(f"  Step {step} ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        # Combined loss
        total_loss = (
            result['loss_combined']
            + args.b1_weight * result['loss_bridge1']
            + args.kl_weight * result['kl_loss']
        )
        effective_cw = 0.0
        if args.contrastive:
            # Warmup the contrastive weight to prevent over-separation early
            # Ramps from 0 to full weight over the first 40% of training
            # This lets bridges learn "memory helps" before learning "wrong memory shouldn't"
            contrastive_ramp = min(1.0, step / (args.steps * 0.4))
            effective_cw = args.contrastive_weight * contrastive_ramp
            if result['contrastive_loss'].item() > 0:
                total_loss = total_loss + effective_cw * result['contrastive_loss']

        # Backward (accumulate)
        scaled_loss = total_loss / args.grad_accum
        scaled_loss.backward()

        # Optimizer step every grad_accum steps
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Metrics
        improvement = result['loss_base'] - result['loss_combined'].item()
        running_improvement = 0.9 * running_improvement + 0.1 * improvement

        # Log
        if step % args.log_every == 0 or step == 1:
            print(
                f"{step:>5} | "
                f"{result['loss_combined'].item():>7.4f} | "
                f"{result['loss_base']:>7.4f} | "
                f"{improvement:>+7.4f} | "
                f"{result['loss_bridge1'].item():>7.4f} | "
                f"{result['kl_loss'].item():>7.4f} | "
                f"{result['field_norm']:>6.3f} | "
                f"{result['working_memories']:>3d} | "
                f"{result['gamma']:>6.4f} | "
                f"{result['perturbation_norm']:>7.4f}"
            )

        # Write log entry
        log_entry = {
            'step': step,
            'loss_combined': result['loss_combined'].item(),
            'loss_base': result['loss_base'],
            'loss_bridge1': result['loss_bridge1'].item(),
            'kl_loss': result['kl_loss'].item(),
            'contrastive_loss': result['contrastive_loss'].item() if args.contrastive else 0.0,
            'contrastive_weight_effective': effective_cw if args.contrastive else 0.0,
            'total_loss': total_loss.item(),
            'improvement': improvement,
            'running_improvement': running_improvement,
            'field_norm': result['field_norm'],
            'working_memories': result['working_memories'],
            'significance': result['significance'],
            'gamma': result['gamma'],
            'gate': result['gate'],
            'perturbation_norm': result['perturbation_norm'],
            'logit_bias_std': result['logit_bias_std'],
            'lr': args.lr * lr_scale,
            'bridge2_enabled': getattr(harness, '_train_bridge2', False),
            'bridge2_gates': harness.memory_bridge.get_gate_values() if getattr(harness, '_train_bridge2', False) else {},
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # Save checkpoint
        if step % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"bridges_step_{step}.pt")
            torch.save({
                'step': step,
                'input_bridge': harness.input_bridge.state_dict(),
                'output_bridge': harness.output_bridge.state_dict(),
                'memory_bridge': harness.memory_bridge.state_dict(),
                'lmf': harness.lmf.state_dict(),
                'optimizer': optimizer.state_dict(),
                'running_improvement': running_improvement,
                'bridge2_gates': harness.memory_bridge.get_gate_values(),
            }, ckpt_path)
            print(f"  ** Checkpoint saved: {ckpt_path}")

        # Track best
        if improvement > best_improvement:
            best_improvement = improvement

    # Final save
    final_path = os.path.join(ckpt_dir, "bridges_final.pt")
    torch.save({
        'step': step,
        'input_bridge': harness.input_bridge.state_dict(),
        'output_bridge': harness.output_bridge.state_dict(),
        'memory_bridge': harness.memory_bridge.state_dict(),
        'lmf': harness.lmf.state_dict(),
        'optimizer': optimizer.state_dict(),
        'running_improvement': running_improvement,
        'bridge2_gates': harness.memory_bridge.get_gate_values(),
    }, final_path)

    log_file.close()

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best improvement: {best_improvement:+.4f}")
    print(f"  Running improvement: {running_improvement:+.4f}")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Training log: {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    train(args)
