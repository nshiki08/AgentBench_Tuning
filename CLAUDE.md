# AgentBench Tuning

AgentBenchの精度向上を目的としたリポジトリ。データセット作成とモデル学習の2つのフェーズで構成される。

## プロジェクト構成

```
AgentBench_Tuning/
├── .devcontainer/          # Devcontainer設定（Claude Code対応）
├── dataset/                # データセット作成
│   └── notebooks/          # marimo notebooks (.py)
├── training/               # モデル学習
│   └── notebooks/          # marimo notebooks (.py)
├── scripts/                # ユーティリティスクリプト
│   └── export_to_ipynb.sh  # marimo → ipynb 変換
├── pyproject.toml          # Python依存関係（uv管理）
└── CLAUDE.md               # このファイル
```

## 開発環境

- **OS**: macOS（Devcontainer経由でLinux上で開発）
- **Python**: 3.12+
- **パッケージ管理**: uv
- **ノートブック**: marimo（開発用）→ ipynb（Colab実行用）に変換
- **実行環境**: Google Colab（VS Code拡張機能で接続）
- **AI支援**: Claude Code（Devcontainer内で使用）

## 開発ワークフロー

### 1. ノートブック開発（marimo）

```bash
# marimo エディタを起動
uv run marimo edit dataset/notebooks/example_dataset_creation.py
uv run marimo edit training/notebooks/example_training.py
```

marimo notebookは `.py` ファイルとして保存される（Gitフレンドリー）。

### 2. Colab用に変換（ipynb）

```bash
# 全ノートブックを変換
./scripts/export_to_ipynb.sh

# 個別に変換
./scripts/export_to_ipynb.sh dataset
./scripts/export_to_ipynb.sh training
```

### 3. Colab上で実行

1. VS Codeで変換された `.ipynb` を開く
2. カーネル選択で「Colab」を選択
3. GPU/TPUランタイムを選択
4. セルを実行（Colabのクラウドリソースで実行される）

### 注意事項

- **ソース管理**: marimo `.py` ファイルが正（`.ipynb` は `.gitignore` に含まれる）
- **変換時の制約**: `mo.ui.*` 等のmarimo固有UIはColab上では動作しない。データ処理・学習ロジックに集中すること
- **PEP 723**: 各ノートブックの先頭に `# /// script` ブロックで依存関係をインラインで記述する

## コーディング規約

- **フォーマッター**: ruff（`pyproject.toml` に設定済み）
- **行長**: 120文字
- **Python**: 型ヒントを活用する
- **命名**: snake_case（Python標準に準拠）

## uv コマンド

```bash
uv sync                    # 依存関係をインストール
uv add <package>           # パッケージを追加
uv run <command>           # 仮想環境内でコマンド実行
uv run marimo edit <file>  # marimoエディタを起動
uv run pytest              # テスト実行
```
