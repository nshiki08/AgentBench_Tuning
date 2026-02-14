# AgentBench Tuning

AgentBenchの精度向上を目的としたリポジトリ。
データセット作成とモデル学習の2フェーズで構成され、[marimo](https://marimo.io/) で開発し [Google Colab](https://colab.google/) 上で実行するワークフローを採用している。

## プロジェクト構成

```
AgentBench_Tuning/
├── .devcontainer/          # Devcontainer設定（Claude Code対応）
│   ├── devcontainer.json
│   └── Dockerfile
├── dataset/                # データセット作成
│   └── notebooks/          # marimo notebooks (.py)
├── docs/
│   └── plans/              # 学習計画ドキュメント
├── training/               # モデル学習
│   └── notebooks/          # marimo notebooks (.py)
├── scripts/
│   └── export_to_ipynb.sh  # marimo → ipynb 変換スクリプト
├── pyproject.toml          # 依存関係（uv管理）
├── CLAUDE.md               # Claude Code向けガイドライン
└── README.md
```

## 学習計画

- [RL中心計画 v1](docs/plans/rl_training_plan_v1.md) — 現行計画。SFTウォームスタート → GRPO強化学習
- [包括的計画 v0](docs/plans/comprehensive_training_plan_v0.md) — 初期版。5フェーズ構成の全体設計

## 前提条件

以下がMacにインストール済みであること。

| ツール | インストール方法 | 確認コマンド |
|--------|------------------|--------------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | 公式サイトからダウンロード | `docker --version` |
| [VS Code](https://code.visualstudio.com/) | 公式サイトからダウンロード | `code --version` |
| [Git](https://git-scm.com/) | Xcode CLT に同梱（`xcode-select --install`） | `git --version` |

## 環境構築（Mac + VS Code）

### 1. VS Code 拡張機能のインストール

VS Code を開き、以下の拡張機能をインストールする。

```bash
# ターミナルから一括インストール
code --install-extension ms-vscode-remote.remote-containers
code --install-extension Google.colab
```

- **Dev Containers** (`ms-vscode-remote.remote-containers`) — Devcontainer を使うために必須
- **Google Colab** (`Google.colab`) — Colab ランタイムに接続して ipynb を実行する

> その他の拡張機能（Python, Ruff, marimo, Claude Code 等）は Devcontainer 起動時に自動インストールされる。

### 2. リポジトリのクローン

```bash
git clone https://github.com/nshiki08/AgentBench_Tuning.git
cd AgentBench_Tuning
```

### 3. Devcontainer の起動

1. VS Code でクローンしたフォルダを開く
   ```bash
   code .
   ```
2. 左下の `><` アイコンをクリック → **「Reopen in Container」** を選択
   （またはコマンドパレット `Cmd + Shift + P` → `Dev Containers: Reopen in Container`）
3. 初回はDockerイメージのビルドが行われる（数分かかる）
4. ビルド完了後、コンテナ内で自動的に `uv sync` が実行され依存関係がインストールされる

> 起動が完了すると、ターミナルに `AgentBench Tuning devcontainer ready.` と表示される。

### 4. 動作確認

Devcontainer 内のターミナルで以下を実行し、環境が正しく構築されたことを確認する。

```bash
# Python
python --version          # Python 3.12.x

# uv
uv --version              # uv x.x.x

# marimo
uv run marimo --version   # marimo x.x.x

# Claude Code
claude --version          # claude-code x.x.x
```

## 開発ワークフロー

### marimo でノートブックを開発する

```bash
# データセット作成
uv run marimo edit dataset/notebooks/example_dataset_creation.py

# 学習
uv run marimo edit training/notebooks/example_training.py
```

marimo notebook は `.py` ファイルとして保存される（Git diff しやすい）。

### Colab で実行する

marimo notebook を ipynb に変換し、VS Code の Colab 拡張機能で実行する。

```bash
# 全ノートブックを一括変換
./scripts/export_to_ipynb.sh

# ディレクトリ指定で変換
./scripts/export_to_ipynb.sh dataset
./scripts/export_to_ipynb.sh training
```

変換後の手順:

1. VS Code で生成された `.ipynb` ファイルを開く
2. 右上のカーネル選択 → **「Connect to Google Colab」** を選択
3. Google アカウントで認証
4. ランタイムのタイプ（GPU / TPU）を選択
5. セルを実行（Colab のクラウドリソース上で実行される）

### パッケージ管理

```bash
uv add <package>           # パッケージ追加
uv remove <package>        # パッケージ削除
uv sync                    # lockfile から環境を同期
uv run <command>           # 仮想環境内でコマンド実行
```

## 注意事項

- **ソース管理**: marimo `.py` ファイルのみコミットする。`.ipynb` は `.gitignore` に含まれており、`export_to_ipynb.sh` で都度再生成する
- **marimo 固有UI**: `mo.ui.*` 等のインタラクティブ要素は Colab 上では動作しない。データ処理・学習ロジックに集中すること
- **PEP 723**: 各ノートブック先頭の `# /// script` ブロックにインライン依存関係を記述できる（`uv run` が自動解決する）
