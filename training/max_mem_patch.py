    max_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            # Reserve more headroom on smaller GPUs (Neural Anamnesis Rust
            # service + CUDA context overhead eats into available memory).
            # <20GB cards get 3.5GB overhead, larger cards get 3GB.
            overhead_gb = 3.5 if total_gb < 20.0 else 3.0
            alloc_gb = max(1, int(total_gb - overhead_gb))
            max_mem[i] = f"{alloc_gb}GiB"
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_gb:.0f}GB) — allocating {alloc_gb}GiB")
    max_mem["cpu"] = "80GiB"
