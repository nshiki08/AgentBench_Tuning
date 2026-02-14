# Model Trainer Agent Memory

## Marimo Notebook Conventions
- PEP 723 `# /// script` block at top with all dependencies
- `__generated_with = "0.10.0"` and `app = marimo.App(width="medium")`
- First cell imports `marimo as mo` and returns `(mo,)`
- `mo.md()` for markdown sections; no `mo.ui.*` (Colab compat)
- Each `@app.cell` must return all variables used by downstream cells as a tuple
- Use `_()` or `_(dependency)` for cell function signatures
- Avoid name collisions in return tuples across cells (e.g. `Path` vs `PathFinal`)
- `if __name__ == "__main__": app.run()` at bottom
- See [marimo-patterns.md](marimo-patterns.md) for details

## Qwen2.5-7B-Instruct SFT Notes
- Chat template: ChatML with `<|im_start|>` / `<|im_end|>` tokens
- Response template for completion-only loss: `<|im_start|>assistant\n`
- Pass response_template as token IDs (not string) to DataCollatorForCompletionOnlyLM
- pad_token defaults to eos_token for Qwen; set explicitly for safety
- `trust_remote_code=True` needed; `attn_implementation="flash_attention_2"` for A100
- `model.config.use_cache = False` required when using gradient checkpointing
- `prepare_model_for_kbit_training()` before `get_peft_model()` for QLoRA
- See [qwen-training.md](qwen-training.md) for hyperparameters

## Qwen2.5-7B-Instruct DPO Notes
- DPO notebook: `training/notebooks/02_dpo_qwen25_7b.py` (Plan C)
- Input: JSONL with `prompt` (list[dict]), `chosen` (list[dict]), `rejected` (list[dict])
- DPOTrainer handles chat template application internally when given message lists
- `padding_side="left"` for tokenizer in DPO (generation consistency)
- LoRA rank 32 (lower than SFT 64) for stability on preference learning
- LR 5e-6 (lower than SFT 2e-5); beta=0.1 (KL penalty); 1 epoch per round
- DPOTrainer + peft_config: ref model via adapter disable (no extra VRAM)
- `AutoPeftModelForCausalLM` for loading SFT adapter -> `merge_and_unload()` -> fresh DPO LoRA
- Fallback: load base model directly if no SFT adapter available
- See [qwen-training.md](qwen-training.md) for full hyperparameter details

## Key Files
- SFT notebook: `training/notebooks/01_sft_qwen25_7b.py`
- DPO notebook: `training/notebooks/02_dpo_qwen25_7b.py`
- Dataset creation: `dataset/notebooks/01_build_training_set_v1.py` (expected JSONL output)
- DPO pairs: `dataset/output/dpo_pairs_v1.jsonl` (from `03_generate_dpo_pairs_v1.py`)
- Export script: `scripts/export_to_ipynb.sh`

## Ruff Linting
- Avoid f-strings without interpolation (ruff F541)
- Line length 120 chars; target Python 3.12
- Rules: E, F, I, N, W, UP
