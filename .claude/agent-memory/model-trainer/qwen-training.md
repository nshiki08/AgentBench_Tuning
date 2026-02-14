# Qwen2.5-7B-Instruct Training Details

## QLoRA Hyperparameters (SFT baseline - v1)
- LoRA rank: 64, alpha: 128, dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Learning rate: 2e-5, cosine schedule, warmup ratio 0.05
- Effective batch: 16 (per_device=2, grad_accum=8)
- Epochs: 3, max_seq_length: 4096
- bf16, gradient checkpointing, adamw_torch_fused
- Weight decay: 0.01, max grad norm: 1.0
- ~1.4% trainable parameters with these LoRA settings

## Memory Estimates (A100 40GB)
- 4-bit quantized model: ~4-5 GB
- LoRA adapters: ~0.5 GB
- Optimizer states + gradients: ~3-4 GB
- Activations (with grad checkpointing, bs=2, seq=4096): ~8-12 GB
- Total estimated: ~18-22 GB -- fits A100 40GB comfortably

## Key Implementation Details
- Use `DataCollatorForCompletionOnlyLM` with tokenized response_template IDs
- `formatting_func` applies chat template per sample; returns list[str]
- `remove_unused_columns=False` in TrainingArguments when using formatting_func
- `gradient_checkpointing_kwargs={"use_reentrant": False}` for compatibility
- `packing=False` to preserve conversation boundaries
- SampleGenerationCallback on_evaluate for qualitative monitoring

## Tokenizer Notes
- Qwen2.5 vocabulary size: 151,936
- Special tokens: <|im_start|>, <|im_end|>, <|endoftext|>
- eos_token: <|im_end|> (used as pad_token)
- padding_side: "right" for training
