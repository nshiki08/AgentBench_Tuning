# 包括的学習計画 v0（初期版）

> RL中心計画に移行する前に策定した、5フェーズ構成の包括的計画。
> サーベイ結果と最新研究を反映。現在は [RL中心計画 v1](rl_training_plan_v1.md) に移行済み。

## Context

AgentBench の DB Bench と ALFWorld のスコアを、Qwen2.5-7B-Instruct および Qwen3-4B-Instruct-2507 で向上させることが目的。文献サーベイおよびWeb調査の結果を統合し、具体的な学習計画を策定する。

---

## サーベイ結果の要約

### 現状のベースライン（AgentBench）

| モデル | Overall | DB | ALFWorld |
|--------|---------|-------|----------|
| GPT-4 | 4.01 | 高 | ~70-78% SR |
| GPT-3.5-turbo | 2.45 | 中 | ~50-60% SR |
| LLaMA-2-70B-chat | 0.55 | 低 | 低 |
| AgentLM-70B (AgentTuning後) | 3.19 | - | - |
| AgentLM-7B (AgentTuning後) | ~1.08 | - | - |
| Qwen2.5-7B-Instruct | 未公表 | 未公表 | 未公表 |

- 7Bクラスのモデルは素の状態ではエージェントタスクがほぼ不可能（成功率一桁台）
- AgentTuning等のfine-tuning後は7Bでも大幅改善（LLaMA-2-7Bで~1.08、13Bで~1.96）
- Qwen2.5-7BはLLaMA-2-7Bより基礎能力が高いため、同手法でより高いスコアが期待できる

### 効果が実証された主要手法

1. **AgentTuning** (Zeng+ 2023): GPT-4トラジェクトリSFT + 汎用データ混合（比率1:3〜5）
2. **Agent-FLAN** (Chen+ 2024): ネガティブサンプルで幻覚削減、ReAct:会話 = 1:9 の混合
3. **SCoRe** (2025): 7BモデルがGRPOで72B教師に匹敵する性能（+6.3 over GRPO baseline）
4. **AgentArk** (2026): R-SFT → DA → PAD(PRM+GRPO) の3段階蒸留
5. **Structured Agent Distillation** (2025): Reason/Act分離型loss、ALFWorld/WebShopで有効
6. **AgentBank** (2024): 50K+トラジェクトリの大規模データで汎化能力向上
7. **StateAct** (2025): 明示的状態表現でALFWorld成功率+10-30%改善

### タスク別の最適戦略

**DB Bench:**
- スキーマ理解 + SQL生成 + エラー回復のマルチステップ推論
- CodeS (SIGMOD 2024) が小規模モデル向けText-to-SQLの最善
- 正しいSQLを生成→実行→結果確認→修正の一連のトラジェクトリが学習データとして有効
- DPO/GRPOでSQL実行正否を報酬として活用可能

**ALFWorld:**
- ReAct形式（Thought → Action → Observation）のトラジェクトリ学習
- 明示的な状態追跡（StateAct方式）で長期計画を改善
- Reflexion式の自己修正パターンも学習データに含める
- タスクタイプ別（pick, put, clean, heat, cool, examine）にバランス

---

## 学習計画

### Phase 0: 環境構築・AgentBench評価基盤

**目的**: ベースラインスコアの計測と再現可能な評価環境の構築

| タスク | 詳細 |
|--------|------|
| AgentBench環境セットアップ | THUDM/AgentBench リポジトリのセットアップ |
| ベースライン評価 | Qwen2.5-7B-Instruct / Qwen3-4B-Instruct-2507 の素のスコアを計測 |
| 評価パイプライン構築 | DB Bench / ALFWorld のスコアを自動計測できるスクリプト |

### Phase 1: データ収集（AgentBench不使用）

**制約**: コンペのため AgentBench のデータ・環境を学習に直接使用することは禁止。

#### 1a. DBマルチターン対話データ

| データソース | 特徴 | 件数 |
|-------------|------|------|
| **SParC** (Yale) | Spider のマルチターン版、4,298対話 | 全量使用 |
| **CoSQL** (Yale) | 会話型 Text-to-SQL、3K+対話 | 全量使用 |
| **Spider** | 複雑SQL、200DB/138ドメイン | マルチターン形式に変換 |
| **BIRD** | 実世界DB、大規模 | マルチターン形式に変換 |
| **合成データ** | ホワイトリストモデルでDB対話トラジェクトリを生成 | 500〜1,000 |

#### 1b. 環境ゴール達成マルチターンデータ

ALFWorld / TextWorld は使用禁止。

| データソース | 特徴 | 件数 |
|-------------|------|------|
| **ScienceWorld** | テキストベース科学実験タスク | 環境から生成 |
| **WebShop** | Web操作によるゴール達成 | 既存データ活用 |
| **Jericho** | テキストアドベンチャーゲーム | 環境から生成 |
| **BabyAI** | グリッド環境での指示追従 | テキスト化して使用 |
| **合成 ReAct トラジェクトリ** | 多段階ゴール達成タスクをReAct形式で生成 | 500〜1,000 |
| **ToolBench** | ツール使用の汎用マルチステップデータ | 品質フィルタ後使用 |

#### 1c. ネガティブサンプル・汎用データ

| データ種別 | 用途 | 件数 |
|-----------|------|------|
| ネガティブサンプル（Agent-FLAN方式） | 不正ツール呼び出し・フォーマット違反等 | 300〜500 |
| 汎用チャットデータ（ShareGPT等） | 汎用能力維持のための混合 | ~4,000 |

### Phase 2: データ加工・品質管理

| タスク | 詳細 |
|--------|------|
| ChatML形式統一 | Qwen互換のChatML + ReAct形式に変換 |
| 品質フィルタリング | 不完全・矛盾するトラジェクトリを除去 |
| マルチターン変換 | 単発データを対話形式に拡張 |
| 状態追跡の付与 | StateAct方式で明示的状態（目標/現在地/インベントリ）を追加 |
| DPOペア作成 | 成功/失敗トラジェクトリからchosen/rejectedペアを構築 |
| Qwen3用変換 | `<think>Thought: ...</think>` 形式（推論モード活用） |
| 汎用データ混合 | agent:general = 1:4 の比率で混合 |

**最終データセット構成**:
- SFT用: エージェント~2,000件 + 汎用~4,000件 = **合計~6,000件**
- DPO用: chosen/rejectedペア **~500ペア**

### Phase 3: SFT（Supervised Fine-Tuning）

| パラメータ | Qwen2.5-7B | Qwen3-4B |
|-----------|------------|----------|
| 手法 | QLoRA (4-bit) | QLoRA (4-bit) |
| LoRA rank / alpha | 64 / 128 | 64 / 128 |
| Target modules | q,k,v,o,gate,up,down_proj | 同左 |
| Learning rate | 2e-5 → cosine decay | 5e-5 → cosine decay |
| Effective batch size | 16 | 16 |
| Epochs | 3 | 3 |
| Max seq length | 4096 | 4096 |
| Warmup ratio | 0.05 | 0.05 |
| Precision | bf16 | bf16 |
| Loss masking | Response-only | 同左 |
| GPU | Colab A100 40GB | Colab T4 16GB |

**要点**:
- Gradient checkpointing 有効化で VRAM 節約
- Agent-FLAN 方式: ReAct形式 10% + 会話形式 90% の混合
- Reason/Act 分離型 loss (Structured Agent Distillation) の適用を検討

### Phase 4: DPO / GRPO（選好最適化）

**戦略A: DPO（安定・実装容易）**

| パラメータ | 値 |
|-----------|------|
| Learning rate | 5e-7 |
| Beta (KL coefficient) | 0.1 |
| Epochs | 1 |
| Batch size | 4 |
| Max length | 4096 |

**戦略B: GRPO（より高い改善が期待）**

| パラメータ | 値 |
|-----------|------|
| Learning rate | 1e-6 |
| KL coefficient | 0.05 |
| Num generations/prompt | 4 |
| Max new tokens | 1024 |
| Temperature (rollout) | 0.8 |

**報酬設計**:
- DB Bench: SQL実行の正否（バイナリ）+ 部分一致スコア
- ALFWorld: タスク完了の成否（バイナリ）+ 中間ステップ進捗

### Phase 5: 評価・反復

| タスク | 詳細 |
|--------|------|
| AgentBench評価 | DB Bench / ALFWorld のスコア計測 |
| ベースラインとの比較 | Phase 0 の結果と比較 |
| エラー分析 | 失敗ケースの分類と原因分析 |
| 反復改善 | エラーパターンに基づく追加データ収集・再学習 |

---

## 当初計画のノートブック一覧

```
dataset/notebooks/
├── 01_agentbench_eval_setup.py      # Phase 0: 評価環境構築
├── 02_trajectory_collection_db.py   # Phase 1: DB Bench トラジェクトリ収集
├── 03_trajectory_collection_alf.py  # Phase 1: ALFWorld トラジェクトリ収集
├── 04_data_processing.py            # Phase 2: データ加工・品質管理
└── 05_dpo_data_creation.py          # Phase 2: DPO ペア作成

training/notebooks/
├── 01_sft_qwen25_7b.py              # Phase 3: Qwen2.5-7B SFT
├── 02_sft_qwen3_4b.py               # Phase 3: Qwen3-4B SFT
├── 03_dpo_training.py               # Phase 4: DPO 学習
├── 04_grpo_training.py              # Phase 4: GRPO 学習
└── 05_evaluation.py                 # Phase 5: 評価
```

---

## 参考文献

- AgentBench (Liu+ 2023): arXiv:2308.03688
- AgentTuning (Zeng+ 2023): arXiv:2310.12823
- Agent-FLAN (Chen+ 2024): arXiv:2403.12881, ACL 2024 Findings
- FireAct (Chen+ 2023): arXiv:2305.16291
- AgentBank (2024): arXiv:2410.07706, EMNLP 2024 Findings
- SCoRe (2025): arXiv:2509.14257 — 7B が GRPO で 72B 教師に匹敵
- AgentArk (2026): arXiv:2602.03955 — R-SFT→DA→PAD(PRM+GRPO)
- Structured Agent Distillation (2025): arXiv:2505.13820 — Reason/Act 分離 loss
- StateAct (Rozanov & Rei, 2025): 明示的状態表現で ALFWorld +10-30%
- ReAct (Yao+ 2023): arXiv:2210.03629
- Reflexion (Shinn+ 2023): arXiv:2303.11366
- CodeS (Li+ 2024): SIGMOD 2024, 小規模モデル向け Text-to-SQL
- Qwen2.5 Technical Report (2025): arXiv:2412.15115
- Qwen3 Technical Report (2025): arXiv:2505.09388

## 検証方法

1. Phase 0 でベースラインスコアを計測し記録
2. 各 Phase 完了後に DB Bench / ALFWorld のスコアを再計測
3. SFT 後 → DPO/GRPO 後の段階的なスコア向上を確認
4. 最終的に AgentBench 公式評価で検証
