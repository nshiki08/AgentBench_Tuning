# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "datasets",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("# AgentBench Dataset Creation\n\nThis notebook is a template for creating datasets for AgentBench tuning.")
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## Workflow

    1. Load raw data
    2. Preprocess and format for training
    3. Export as HuggingFace Dataset
    """)
    return


@app.cell
def _():
    # TODO: Implement dataset creation logic
    pass
    return


if __name__ == "__main__":
    app.run()
