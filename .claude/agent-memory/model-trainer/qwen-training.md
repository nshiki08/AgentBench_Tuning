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

## QLoRA Hyperparameters (DPO - Plan C)
- LoRA rank: 32, alpha: 64, dropout: 0.05 (lower rank than SFT for stability)
- Target modules: same as SFT (q/k/v/o_proj, gate/up/down_proj)
- Learning rate: 5e-6, cosine schedule, warmup ratio 0.1
- Effective batch: 16 (per_device=2, grad_accum=8)
- Epochs: 1 per round
- beta: 0.1 (KL penalty), loss_type: "sigmoid" (standard DPO)
- max_length: 4096, max_prompt_length: 2048
- bf16, gradient checkpointing
- Weight decay: 0.01, max grad norm: 1.0
- `remove_unused_columns=False` to preserve task_type/source metadata

## DPO Implementation Details
- DPOTrainer from trl; pass peft_config to let it apply fresh LoRA
- Reference model: implicit via PEFT adapter disable (no extra VRAM)
- SFT adapter loaded via AutoPeftModelForCausalLM -> merge_and_unload()
- Fallback to base model if SFT adapter not found
- padding_side: "left" for DPO (generation consistency)
- Input format: prompt/chosen/rejected as message lists (DPOTrainer applies template)
- DPOSampleGenerationCallback for qualitative monitoring during eval

## Tokenizer Notes
- Qwen2.5 vocabulary size: 151,936
- Special tokens: <|im_start|>, <|im_end|>, <|endoftext|>
- eos_token: <|im_end|> (used as pad_token)
- padding_side: "right" for SFT training, "left" for DPO
