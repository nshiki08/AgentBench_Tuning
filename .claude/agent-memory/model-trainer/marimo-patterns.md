# Marimo Notebook Patterns

## File Structure
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", ...]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    mo.md("# Title")
    return (mo,)

# ... more cells ...

if __name__ == "__main__":
    app.run()
```

## Cell Rules
- Each cell is a function decorated with `@app.cell`
- Cell function signature declares dependencies: `def _(mo, cfg, torch):`
- Return tuple declares what this cell provides: `return cfg, model, tokenizer`
- Use `_` as function name (marimo convention)
- Markdown-only cells can `return` with no value
- Imports inside cells (not at module level) -- marimo requirement

## Colab Compatibility
- No `mo.ui.*` widgets (won't work in ipynb)
- Use `mo.md()` for section explanations only
- Include pip install via subprocess for Colab detection
- Export via: `./scripts/export_to_ipynb.sh training`

## Name Collision Avoidance
- Each cell's returned names must be unique across the notebook
- If `Path` is used in cell 3, use `PathFinal` alias in cell 7
- Same for loop variables like `i`, `messages` -- return them to avoid warnings
