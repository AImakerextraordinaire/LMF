def load_full_system(
    model_path: str,
    checkpoint_path: str,
    anamnesis_url: str = "http://localhost:6060",
):
    """
    Load the complete Phase 6 stack.

    NOTE: Do NOT use BitsAndBytes NF4 quantization here. The model uses native
    MXFP4 quantization on expert weights. Stripping MXFP4 and applying BnB NF4
    destroys the expert weights (gate_up_proj_blocks → randomly initialized
    gate_up_proj). Load with dtype=bfloat16 + accelerate offloading exactly
    as in the training scripts. The KV cache fix in generate_text() already
    solves the VRAM issue at long generation lengths.
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    t0 = time.time()
    max_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            # Same overhead as training: 5GB on 24GB card, 3.5GB on 16GB card
            overhead_gb = 3.5 if total_gb < 20.0 else 5.0
            alloc_gb = max(1, int(total_gb - overhead_gb))
            max_mem[i] = f"{alloc_gb}GiB"
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_gb:.0f}GB) "
                  f"— allocating {alloc_gb}GiB")
    max_mem["cpu"] = "80GiB"

    # Load with native MXFP4 intact — same as training pipeline
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        offload_folder="offload_temp",
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
    hook_manager = RouterHookManager(model, router_bias, verbose=False)
    injector = NeuralAnamnInjector(field_dim=lmf.field_dim)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        harness.input_bridge.load_state_dict(ckpt['input_bridge'])
        harness.output_bridge.load_state_dict(ckpt['output_bridge'])
        harness.memory_bridge.load_state_dict(ckpt['memory_bridge'])
        if 'lmf' in ckpt:
            harness.lmf.load_state_dict(ckpt['lmf'])
        if 'injector' in ckpt:
            injector.load_state_dict(ckpt['injector'])
        if 'router_bias' in ckpt:
            router_bias.load_state_dict(ckpt['router_bias'])
            scales = router_bias.get_layer_scales()
            print(f"  RouterBias loaded (scale_max={max(scales):.4f})")
        print(f"  Phase {ckpt.get('phase','?')}, step {ckpt.get('step','?')}, "
              f"improvement={ckpt.get('running_improvement', 0):+.4f}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")

    print(f"Connecting to Neural Anamnesis at {anamnesis_url}...")
    anamnesis = SyncNeuralAnamnClient(base_url=anamnesis_url)
    if anamnesis.is_available():
        print("  Neural Anamnesis connected")
    else:
        print("  Neural Anamnesis unavailable")

    return {
        'model': model, 'tokenizer': tokenizer, 'lmf': lmf,
        'harness': harness, 'injector': injector,
        'router_bias': router_bias, 'hook_manager': hook_manager,
        'anamnesis': anamnesis,
        'vocab_size': model.config.vocab_size,
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
