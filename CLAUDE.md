# AgentBench Tuning

AgentBenchの精度向上を目的としたリポジトリ。データセット作成とモデル学習の2つのフェーズで構成される。

## プロジェクト構成

```
AgentBench_Tuning/
├── .claude/
│   └── agents/             # サブエージェント定義
├── .devcontainer/          # Devcontainer設定（Claude Code対応）
├── dataset/                # データセット作成
│   └── notebooks/          # marimo notebooks (.py)
├── docs/
│   └── plans/              # 学習計画の詳細ドキュメント
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

## サブエージェント

`.claude/agents/` に定義された専門エージェント群。Claude Code の Task ツール経由で起動される。

| エージェント | 役割 | 使用場面 |
|---|---|---|
| **literature-survey-agent** | 学術文献サーベイ | AgentBench（DB Bench / ALF World）の精度向上に関する研究手法・論文の調査 |
| **dataset-creator** | データセット作成 | 学習・評価用データの生成・加工・品質管理・フォーマット変換 |
| **model-trainer** | モデル学習 | Qwen2.5-7B / Qwen3-4B の SFT・RL（GRPO, DPO 等）学習パイプライン実装 |
| **llm-ecosystem-engineer** | 環境構築・依存解決 | LLM関連ライブラリの依存競合解決、CUDA/GPU問題、バージョン互換性の対応 |

### エージェント間の連携フロー

```
literature-survey-agent  →  研究知見を提供
        ↓
dataset-creator          →  学習データを作成
        ↓
model-trainer            →  モデルを学習
        ↑
llm-ecosystem-engineer   →  環境・依存問題を随時解決
```

## 学習計画

**制約**: コンペのため AgentBench のデータ・環境を学習に直接使用することは禁止。
汎用的な「DBマルチターン対話能力」と「環境ゴール達成マルチターン能力」を強化する。

**計画ドキュメント**:
- [RL中心計画 v1](docs/plans/rl_training_plan_v1.md) — 現行計画。報酬関数・ハイパーパラメータ・VRAM見積もり等
- [包括的計画 v0](docs/plans/comprehensive_training_plan_v0.md) — 初期版。5フェーズ構成のサーベイ・データ収集・SFT・RL全体設計

### 合成データ・蒸留に使用可能なモデル（ホワイトリスト）

以下のモデルのみ、合成データ生成・蒸留に使用可能。

| カテゴリ | モデル |
|---------|--------|
| **学習対象** | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507` |
| **GPT-OSS 120B** | `openai/gpt-oss-120b`, `unsloth/gpt-oss-120b-GGUF`, `unsloth/gpt-oss-120b-unsloth-bnb-4bit` |
| **Qwen2.5-72B** | `Qwen/Qwen2.5-72B-Instruct`, `-AWQ`, `-GGUF`, `-GPTQ-Int4`, `-GPTQ-Int8` |
| **Nemotron-3-Nano-30B** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `-NVFP4`, `-FP8`, `unsloth/...-GGUF` |
| **Qwen3-Coder-30B** | `Qwen/Qwen3-Coder-30B-A3B-Instruct`, `-FP8`, `unsloth/...-GGUF`, `-1M-GGUF` |
| **Qwen3-30B** | `Qwen/Qwen3-30B-A3B-Instruct-2507`, `-FP8`, `unsloth/...-GGUF` |
| **GPT-OSS 20B** | `openai/gpt-oss-20b`, `unsloth/gpt-oss-20b-GGUF`, `unsloth/gpt-oss-20b-unsloth-bnb-4bit` |
| **GLM-4.7-Flash** | `zai-org/GLM-4.7-Flash`, `unsloth/GLM-4.7-Flash-GGUF`, `-FP8-Dynamic` |
| **Qwen3-32B** | `Qwen/Qwen3-32B`, `-FP8`, `-AWQ`, `-GGUF`, `unsloth/...-unsloth-bnb-4bit` |
| **Qwen3-14B** | `Qwen/Qwen3-14B`, `-FP8`, `-AWQ`, `-GGUF`, `unsloth/...-unsloth-bnb-4bit` |

### 対象モデル・ベンチマーク

- **モデル**: Qwen2.5-7B-Instruct / Qwen3-4B-Instruct-2507
- **評価**: AgentBench（DB Bench, ALFWorld）— 評価のみに使用

### 方針: RL中心アプローチ

SFTウォームスタート（1 epoch）→ GRPO による強化学習で完結する。

```
SFTデータ (6K件)  →  SFT 1 epoch  →  GRPOプロンプト (800-1,300件)  →  GRPO  →  最終モデル
```

### 3つのオプション（詳細は [計画ファイル](docs/plans/rl_training_plan_v1.md) 参照）

| オプション | 概要 | VRAM | シンプルさ |
|-----------|------|------|-----------|
| **A: SFT+GRPO（推奨）** | 既存SFTデータで1 epochウォームスタート → GRPO | ~24-30 GB | 最高 |
| **B: 蒸留+GRPO** | ホワイトリストモデルで高品質トラジェクトリ生成 → SFT → GRPO | ~24-30 GB | 中 |
| **C: SFT+DPO（フォールバック）** | SFT → オフラインでペア生成 → DPO 反復 | ~18-22 GB | 中 |

### 報酬設計

- **DB**: SQL実行正否（SQLiteで自動検証）+ 部分一致Jaccardスコア
- **Env**: ReActフォーマット準拠 + 行動妥当性 + 推論品質（ヒューリスティック複合）
- 任意拡張: ホワイトリストモデルによるLLM-as-Judge

### ノートブック一覧

```
dataset/notebooks/
├── 01_build_training_set_v1.py       # SFTデータセット（実装済み）
└── 02_build_rl_prompts_v1.py         # GRPOプロンプトデータ

training/notebooks/
├── 01_sft_qwen25_7b.py               # SFTウォームスタート（実装済み）
├── 02_grpo_qwen25_7b.py              # GRPO学習 7B
└── 03_grpo_qwen3_4b.py               # GRPO学習 4B
```

### 参考文献

| 論文 | 要点 |
|------|------|
| SCoRe (2025) | 7BがGRPOで72B教師に匹敵 (arXiv:2509.14257) |
| DeepSeek-R1 (2025) | コールドスタートSFT → GRPO の2段階が有効 |
| Embodied Planner-R1 (2025) | 純RLでALFWorld 97.78% (arXiv:2506.23127) |
| RAGEN (2025) | マルチターンRL、StarPOフレームワーク (arXiv:2504.20073) |
| AgentTuning (Zeng+ 2023) | トラジェクトリSFT + 汎用データ混合 (arXiv:2310.12823) |
| Agent-FLAN (Chen+ 2024) | ネガティブサンプルで幻覚削減 (arXiv:2403.12881) |
| ReAct (Yao+ 2023) | Thought→Action→Observation形式 (arXiv:2210.03629) |

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
