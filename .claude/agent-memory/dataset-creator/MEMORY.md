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

## Contamination Prevention
- BANNED keywords: alfworld, alf_world, textworld, text_world, tw_, tw-
- Triple-layer check: per-row filter, _make_chatml check, final_contamination_check
- ETO dataset (`agent-eto/eto-sft-trajectory`) contains ALFWorld -- must filter

## HuggingFace Dataset Candidates
- SParC: `aherntech/sparc` (fallback: `xlangai/sparc`)
- CoSQL: `aherntech/cosql`, `xlangai/cosql`, `parkervg/cosql`
- Spider: `xlangai/spider`
- Synthetic SQL: `gretelai/synthetic_text_to_sql`
- ETO trajectories: `agent-eto/eto-sft-trajectory`
- General chat: `OpenAssistant/oasst2`, `lmsys/chatbot_arena_conversations` (fallback: `lmsys/lmsys-chat-1m`)

## Key Files
- `dataset/notebooks/01_build_training_set_v1.py` -- Main dataset builder notebook
- `pyproject.toml` -- ruff config: line-length=120, target-version=py312
