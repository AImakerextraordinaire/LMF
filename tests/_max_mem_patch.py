    print("Loading model...")
    t0 = time.time()
    max_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            # During eval, MXFP4 dequantization at load time allocates ~2GB
            # temporary buffers on GPU. Use higher overhead than training to
            # give the dequantizer room — 7GB on 24GB card, 4GB on 16GB card.
            overhead_gb = 4.0 if total_gb < 20.0 else 7.0
            alloc_gb = max(1, int(total_gb - overhead_gb))
            max_mem[i] = f"{alloc_gb}GiB"
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_gb:.0f}GB) "
                  f"— allocating {alloc_gb}GiB ({overhead_gb}GB overhead for MXFP4 dequant)")
    max_mem["cpu"] = "80GiB"