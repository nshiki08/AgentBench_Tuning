# Dataset Creator Agent Memory

## Project Structure
- Marimo notebooks live in `dataset/notebooks/` and `training/notebooks/`
- Example notebook pattern: see `dataset/notebooks/example_dataset_creation.py`
- Export to ipynb via `./scripts/export_to_ipynb.sh`
- Output datasets go to `dataset/output/` (not tracked in git)

## Marimo Notebook Conventions
- PEP 723 `# /// script` block at top for inline dependencies
- `import marimo` + `app = marimo.App(width="medium")` at module level
- Each cell is `@app.cell` decorated function
- Cell dependencies are passed via function parameters (marimo reactive system)
- Return tuple must contain all names that downstream cells need
- `mo.md()` for section headers only; NO `mo.ui.*` (Colab compat)
- DO NOT use `from __future__ import annotations` in cells -- it breaks marimo's reactive type resolution
- `if __name__ == "__main__": app.run()` at the end
- `__generated_with = "0.10.0"` matches the marimo version

## Dataset Schema (AgentBench Training V1)
- ChatML format (Qwen-compatible): `{"messages": [...], "task_type": "db"|"env"|"general", "source": "..."}`
- Sources: sparc, cosql, spider, synthetic_sql, webshop, scienceworld, synthetic_react, general_chat
- Target: ~6,000 samples, agent:general = 1:2
- JSONL output with companion metadata.json

## DPO Pairs Schema (V1)
- Output: `{"prompt": [...], "chosen": [...], "rejected": [...], "chosen_score": float, "rejected_score": float, "task_type": "db"|"env", "source": "..."}`
- K=8 completions per prompt, temperature=0.8, top_p=0.95, max_new_tokens=1024
- MIN_SCORE_DIFF=0.1 threshold for pair selection
- DB reward: SQL execution match (exact=1.0, partial=Jaccard*0.5, format-only=0.3/-0.3, error=-0.5)
- Env reward: ReAct format (0.4) + action validity (0.3) + reasoning quality (0.3)
- Notebook: `dataset/notebooks/03_generate_dpo_pairs_v1.py`

## RL Prompts Dataset (Plan C: Offline DPO)
- `dataset/notebooks/02_build_rl_prompts_v1.py` -- Prompt-only dataset for GRPO/DPO
- Output: `dataset/output/rl_prompts_v1.jsonl` + `rl_prompts_v1_metadata.json`
- Schema: `{prompt: [{role,content}...], task_type, source, gold_sql, db_path}`
- DB prompts (500-800): Spider/SParC/CoSQL -- system+user only, gold_sql + db_path for reward
- Env prompts (300-500): ETO (WebShop/ScienceWorld) + 40 synthetic templates with parameterization

## Contamination Prevention
- BANNED keywords: alfworld, alf_world, textworld, text_world, tw_, tw-
- Triple-layer check: per-row filter, _make_chatml check, final_contamination_check
- ETO dataset (`agent-eto/eto-sft-trajectory`) contains ALFWorld -- must filter

## Key Files
- `dataset/notebooks/01_build_training_set_v1.py` -- Main dataset builder notebook (SFT)
- `dataset/notebooks/02_build_rl_prompts_v1.py` -- RL prompt dataset builder (DPO/GRPO)
- `dataset/notebooks/03_generate_dpo_pairs_v1.py` -- DPO pair generation notebook
- `training/notebooks/01_sft_qwen25_7b.py` -- SFT training notebook (QLoRA, Qwen2.5-7B)
- `pyproject.toml` -- ruff config: line-length=120, target-version=py312

## Ruff Lint Notes
- Selected rules: E, F, I, N, W, UP
- Per-file-ignores for notebooks: N803, N806 are ignored
- Watch for F841 (unused variables) -- e.g. loop variables, batch_end assigned but not used
- Avoid duplicate local variable definitions within same function scope
- Colab detection pattern uses `# noqa: F401` for the google.colab import check

## Model Loading Pattern (for generation notebooks)
- Use `padding_side="left"` for batched generation (vs "right" for training)
- Sub-batch generation (4 at a time) for memory efficiency with K=8 completions
- Graceful fallback when SFT adapter not found (use base model only, log warning)
