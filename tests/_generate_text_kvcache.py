def generate_text(sys: dict, prompt: str, field_snapshot: Optional[torch.Tensor],
                  use_bridge3: bool, max_new_tokens: int = 120) -> Tuple[str, int, float]:
    """
    Generate text with optional field-state modulated logits.
    Uses KV cache (past_key_values) so each step only processes the new token
    rather than the full growing sequence — O(n) VRAM instead of O(n²).
    Returns (text, token_count, elapsed_seconds).
    """
    model = sys['model']
    tokenizer = sys['tokenizer']
    harness = sys['harness']

    input_ids, attention_mask = tokenize_chat(tokenizer, model, prompt)
    start_len = input_ids.shape[1]

    t0 = time.time()

    # ── Prefill pass — process full prompt, seed KV cache ────────────────
    with torch.no_grad():
        prefill_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = prefill_out.past_key_values
        base_logits = prefill_out.logits[0, -1].float().cpu()

        if use_bridge3 and field_snapshot is not None:
            bridge3_out = harness.output_bridge(field_state=field_snapshot)
            logit_bias = bridge3_out['logit_bias'][0].cpu()
            next_logits = base_logits + logit_bias
        else:
            next_logits = base_logits

    next_logits[200000:] = float('-inf')
    next_token = int(torch.argmax(next_logits).item())
    generated_ids = [next_token]

    # ── Decode loop — one new token per step, KV cache handles context ───
    for step in range(1, max_new_tokens):
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break

        next_t = torch.tensor(
            [[next_token]], device=input_ids.device, dtype=input_ids.dtype
        )
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device,
                                        dtype=attention_mask.dtype)],
            dim=1,
        )

        with torch.no_grad():
            step_out = model(
                input_ids=next_t,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = step_out.past_key_values
            base_logits = step_out.logits[0, -1].float().cpu()

            if use_bridge3 and field_snapshot is not None:
                bridge3_out = harness.output_bridge(field_state=field_snapshot)
                logit_bias = bridge3_out['logit_bias'][0].cpu()
                next_logits = base_logits + logit_bias
            else:
                next_logits = base_logits

        next_logits[200000:] = float('-inf')
        next_token = int(torch.argmax(next_logits).item())
        generated_ids.append(next_token)

        # Early stop — only after 90% of budget
        if step > int(max_new_tokens * 0.9):
            response_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if "(Note:" in response_so_far:
                break
            if response_so_far.endswith(('.', '!', '?', '"')) and "\n\n" in response_so_far[-50:]:
                break

        # Free KV cache memory periodically for very long generations
        if step % 200 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    new_tokens = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    del past_key_values
    torch.cuda.empty_cache()

    for stop in ["(Note:", "}"]:
        if stop in text:
            text = text.split(stop)[0].strip()

    return text, new_tokens, elapsed