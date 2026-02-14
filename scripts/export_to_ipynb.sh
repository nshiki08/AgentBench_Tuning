#!/bin/bash
# Export all marimo notebooks (.py) to Jupyter notebooks (.ipynb)
# for execution via the Google Colab VS Code extension.
#
# Usage:
#   ./scripts/export_to_ipynb.sh                 # Export all
#   ./scripts/export_to_ipynb.sh dataset          # Export dataset notebooks only
#   ./scripts/export_to_ipynb.sh training         # Export training notebooks only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

TARGET="${1:-all}"

export_notebooks() {
    local dir="$1"
    local notebook_dir="$PROJECT_ROOT/$dir/notebooks"

    if [ ! -d "$notebook_dir" ]; then
        echo "Directory not found: $notebook_dir"
        return
    fi

    echo "=== Exporting $dir notebooks ==="

    for py_file in "$notebook_dir"/*.py; do
        [ -f "$py_file" ] || continue

        filename=$(basename "$py_file" .py)
        ipynb_file="$notebook_dir/${filename}.ipynb"

        echo "  $py_file -> $ipynb_file"
        uv run marimo export ipynb "$py_file" -o "$ipynb_file" --sort top-down
    done

    echo ""
}

case "$TARGET" in
    all)
        export_notebooks "dataset"
        export_notebooks "training"
        ;;
    dataset|training)
        export_notebooks "$TARGET"
        ;;
    *)
        echo "Usage: $0 [all|dataset|training]"
        exit 1
        ;;
esac

echo "Done. Exported .ipynb files can be opened in VS Code with the Colab extension."
