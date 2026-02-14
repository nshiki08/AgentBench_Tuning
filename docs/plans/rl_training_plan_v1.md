# RL中心の学習計画 v1

> AgentBench (DB Bench / ALFWorld) のスコアを Qwen2.5-7B-Instruct / Qwen3-4B-Instruct-2507 で向上させるための、強化学習中心の学習計画。

## 前提条件・制約

| 項目 | 内容 |
|------|------|
| 対象モデル | Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507 |
| 評価ベンチマーク | AgentBench（DB Bench, ALFWorld）— **評価のみ、学習には使用禁止** |
| 禁止データ | AgentBench / ALFWorld / TextWorld のデータ・環境 |
| 蒸留許可モデル | [ホワイトリスト](../../CLAUDE.md)参照 |
| GPU環境 | Colab A100 40GB（7B用）/ T4 16GB（4B用） |
| フレームワーク | trl >= 0.22.0, peft >= 0.13.0, bitsandbytes >= 0.44.0 |

## 既存成果物

| ファイル | 状態 | 内容 |
|---------|------|------|
| `dataset/notebooks/01_build_training_set_v1.py` | 実装済み | SFTデータセット ~6,000件（DB 1K + Env 1K + 汎用 4K） |
| `training/notebooks/01_sft_qwen25_7b.py` | 実装済み | QLoRA SFT（response-only loss masking） |

---

## 3つのRL計画オプション

### 比較表

| 基準 | Option A: SFT+GRPO | Option B: 蒸留+GRPO | Option C: SFT+DPO |
|------|--------------------|--------------------|-------------------|
| **シンプルさ** | **最高** | 中 | 中 |
| **期待スコア向上** | 中〜高 | **最高** | 中 |
| **API依存** | 低（LLM判定のみ任意） | 高（トラジェクトリ生成） | 低 |
| **VRAM（A100）** | ~24-30 GB | ~24-30 GB | **~18-22 GB** |
| **実行時間** | ~6-10時間 | ~10-16時間 | ~8-14時間（3ラウンド） |
| **研究的裏付け** | 強い（SCoRe, DeepSeek-R1） | 強い（蒸留+RL） | 中程度 |
| **新規Notebook数** | 2 | 3 | 3 |

---

## Option A: SFTウォームスタート → GRPO（推奨）

### 概要

既存SFTデータで1 epochだけウォームスタートし、エージェント形式の基本能力を獲得した上で、
GRPOによるオンライン強化学習で報酬最大化する。DeepSeek-R1のコールドスタート手法に近い。
SFTなしの素のQwen2.5-7Bはエージェントタスクの成功率が低すぎ（<10%）、
報酬分散がゼロに近くなりGRPOが機能しないため、最低限のSFTが必要。

### パイプライン

```
01_build_training_set_v1.py  →  training_set_v1.jsonl (6K samples)
                                       ↓
01_sft_qwen25_7b.py (1 epoch) →  sft_adapter/
                                       ↓
02_build_rl_prompts_v1.py    →  rl_prompts_v1.jsonl (800-1,300 prompts)
                                       ↓
02_grpo_qwen25_7b.py         →  grpo_adapter/ (最終モデル)
```

### Phase 1: SFTウォームスタート

既存の `training/notebooks/01_sft_qwen25_7b.py` をそのまま使用。ただし **1 epoch** に変更。

| パラメータ | 値 |
|-----------|-----|
| Epochs | **1**（元は3。形式学習のみで十分） |
| Learning rate | 2e-5 |
| LoRA rank / alpha | 64 / 128 |
| Effective batch size | 16 |
| Max seq length | 4096 |

### Phase 2: GRPOプロンプトデータセット作成

新規: `dataset/notebooks/02_build_rl_prompts_v1.py`

GRPOTrainerは **プロンプトのみ** を入力とし、モデル自身が応答を生成して報酬で最適化する。
既存のSFTデータとは異なり、完了済み会話ではなく「問題文+スキーマ」だけを含むデータセット。

**DBプロンプト（500-800件）**:

Spider / SParC / CoSQL の訓練セットから抽出。各プロンプトにはゴールドSQLと
SQLiteファイルパスをメタデータとして付与（報酬関数がSQL実行検証に使用）。

```json
{
  "prompt": [
    {"role": "system", "content": "You are an expert database assistant..."},
    {"role": "user", "content": "Database schema:\n<schema>\n\nQuestion: <question>"}
  ],
  "gold_sql": "SELECT ...",
  "db_path": "/path/to/database.sqlite",
  "task_type": "db"
}
```

**Envプロンプト（300-500件）**:

WebShop / ScienceWorld のタスク記述 + 合成タスクテンプレート（家事・ナビゲーション・ツール使用）を拡張。
ALFWorld / TextWorld 由来のプロンプトは **完全に排除**。

```json
{
  "prompt": [
    {"role": "system", "content": "You are an autonomous agent operating in an interactive environment..."},
    {"role": "user", "content": "Task: <goal>\n\nYou are in an interactive environment. Complete the task."}
  ],
  "task_type": "env"
}
```

### Phase 3: 報酬関数

GRPOTrainerに渡す2つの報酬関数。タスクタイプに応じて適用される。

#### DB報酬（SQL実行ベース）

```python
def db_reward_func(completions, gold_sql, db_path, task_type, **kwargs):
    """SQL実行結果の一致で報酬を計算。非DBタスクはスキップ。"""
    rewards = []
    for completion, gold, db, tt in zip(completions, gold_sql, db_path, task_type):
        if tt != "db":
            rewards.append(0.0)
            continue

        # 応答からSQLを抽出
        content = completion[0]["content"] if isinstance(completion, list) else completion
        sql_match = re.search(r"```sql\n(.*?)\n```", content, re.DOTALL)
        if not sql_match:
            sql_match = re.search(r"SELECT\s.+?;", content, re.DOTALL | re.IGNORECASE)
        if not sql_match:
            rewards.append(-0.5)  # フォーマット違反
            continue

        pred_sql = sql_match.group(1).strip() if sql_match.lastindex else sql_match.group(0).strip()

        try:
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            pred_result = set(map(tuple, cur.execute(pred_sql).fetchall()))
            gold_result = set(map(tuple, cur.execute(gold).fetchall()))
            conn.close()

            if pred_result == gold_result:
                rewards.append(1.0)   # 完全一致
            elif pred_result & gold_result:
                jaccard = len(pred_result & gold_result) / len(pred_result | gold_result)
                rewards.append(jaccard * 0.5)  # 部分一致
            else:
                rewards.append(-0.25)  # 不正解
        except Exception:
            rewards.append(-0.5)  # 実行エラー

    return rewards
```

#### Env報酬（ヒューリスティック）

ALFWorld環境にアクセスできないため、フォーマット準拠・行動妥当性・推論品質の
複合ヒューリスティックで報酬を計算。

```python
def env_reward_func(completions, task_type, **kwargs):
    """ReActフォーマット準拠 + 行動妥当性 + 推論品質で報酬。"""
    rewards = []
    for completion, tt in zip(completions, task_type):
        if tt != "env":
            rewards.append(0.0)
            continue

        content = completion[0]["content"] if isinstance(completion, list) else completion
        reward = 0.0

        # (1) ReActフォーマット準拠 (max 0.4)
        has_thought = bool(re.search(r"Thought:", content))
        has_action = bool(re.search(r"Action:", content))
        has_input = bool(re.search(r"Action Input:", content))
        if has_thought and has_action and has_input:
            reward += 0.4
        elif has_thought or has_action:
            reward += 0.1

        # (2) 行動妥当性 (max 0.3)
        valid_actions = [
            "interact", "navigate", "look", "pick", "put", "go",
            "take", "open", "close", "use", "examine", "finish",
            "search", "click", "buy",
        ]
        action_match = re.search(r"Action:\s*(\w+)", content)
        if action_match and any(v in action_match.group(1).lower() for v in valid_actions):
            reward += 0.3

        # (3) 推論品質 (max 0.3)
        thought_match = re.search(r"Thought:\s*(.*?)(?:Action:|$)", content, re.DOTALL)
        if thought_match and len(thought_match.group(1).strip()) > 20:
            reward += 0.3

        rewards.append(reward)

    return rewards
```

**任意拡張: LLM-as-Judge**

ホワイトリストモデル（Qwen3-32B等）をAPI経由で呼び出し、
トラジェクトリの品質を0-1で評価する報酬関数を追加可能。
コスト・レイテンシとの兼ね合いで判断。

### Phase 4: GRPO学習

新規: `training/notebooks/02_grpo_qwen25_7b.py`

#### ハイパーパラメータ

| パラメータ | Qwen2.5-7B (A100) | Qwen3-4B (T4) |
|-----------|-------------------|----------------|
| num_generations | 4 | 2 |
| max_completion_length | 1024 | 512 |
| max_prompt_length | 2048 | 1024 |
| per_device_train_batch_size | 1 | 1 |
| gradient_accumulation_steps | 16 | 32 |
| num_train_epochs | 3 | 3 |
| learning_rate | 5e-6 | 5e-6 |
| lr_scheduler_type | cosine | cosine |
| warmup_ratio | 0.1 | 0.1 |
| beta (KL penalty) | 0.0 | 0.0 |
| epsilon (clipping) | 0.2 | 0.2 |
| temperature (生成時) | 0.8 | 0.8 |
| top_p | 0.95 | 0.95 |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |
| LoRA target modules | q,k,v,o,gate,up,down_proj | 同左 |
| gradient_checkpointing | True | True |
| bf16 | True | True |

**重要な設計判断**:

- **beta=0.0**: KLペナルティなし。リファレンスモデルのロードが不要になり、VRAM節約。
  PEFTを使用する場合、trlはアダプタを無効化してリファレンスlogprobsを計算するため、
  別途モデルをロードする必要がない。
- **LoRA rank=32**: SFT（rank=64）より低い。RL段階では微調整のため。
- **num_generations=4**: プロンプトごとに4応答生成し、グループ内で相対ランキング。
  メモリ制約から A100 で最大4、T4で最大2。

#### VRAM見積もり（Qwen2.5-7B, A100 40GB）

| コンポーネント | VRAM |
|--------------|------|
| 4-bit量子化モデル | ~4 GB |
| LoRAアダプタ (r=32) | ~0.3 GB |
| オプティマイザ (AdamW, LoRAパラメータのみ) | ~1.5 GB |
| アクティベーション (gradient checkpointing, bs=1, 4gen, seq=1024) | ~12-16 GB |
| KVキャッシュ (4シーケンス x 1024トークン) | ~4-6 GB |
| その他 | ~2 GB |
| **合計** | **~24-30 GB** |

#### 監視メトリクス

- `reward/mean`: 平均報酬（上昇が期待される）
- `reward/std`: 報酬の標準偏差（ゼロに近づくと飽和）
- `loss/policy`: ポリシーロス
- `completion_length/mean`: 生成長（爆発していないか）
- wandb で可視化

### Pros / Cons

**Pros**:
- 最もシンプルなパイプライン（SFT→GRPO の2ステップ）
- DB報酬は完全自動化（SQL実行検証）
- beta=0.0 + PEFTでリファレンスモデル不要、メモリ効率
- trl の GRPOTrainer で直接実装可能

**Cons**:
- Env報酬がヒューリスティック（ALFWorld環境にアクセスできない）
- num_generations=4 ではプロンプトあたりの多様性が限定的
- SFT品質が不足するとGRPOが効かないリスク

---

## Option B: ホワイトリストモデル蒸留 → GRPO

### 概要

ホワイトリストの大規模モデル（gpt-oss-120b / Qwen2.5-72B-Instruct）で
高品質なエージェントトラジェクトリを生成し、検証済みのものでSFT。
その後GRPOでさらに最適化する。Option A より強い開始ポリシーが得られるが、
API呼び出しが必要。

### パイプライン

```
02_distill_trajectories_v1.py  →  distilled_trajectories.jsonl (800-1,500件)
            ↓
01_build_training_set_v1.py + 蒸留データ統合  →  enriched_training_set.jsonl
            ↓
01_sft_qwen25_7b.py (1 epoch)  →  sft_adapter/
            ↓
02_build_rl_prompts_v1.py     →  rl_prompts_v1.jsonl
            ↓
02_grpo_qwen25_7b.py          →  grpo_adapter/
```

### Phase 1: トラジェクトリ蒸留

新規: `dataset/notebooks/02_distill_trajectories_v1.py`

**DBトラジェクトリ（500-1,000件）**:
- Spider/SParC/CoSQL の問題をホワイトリストモデルに入力
- ReAct形式で Thought + SQL を生成させる
- **SQL実行で正否を自動検証** → 正解のみ保持
- 蒸留元: `gpt-oss-120b` または `Qwen2.5-72B-Instruct`（APIで呼び出し）

**Envトラジェクトリ（300-500件）**:
- WebShop / ScienceWorld / 合成タスクの記述を入力
- ReAct形式のマルチステップトラジェクトリを生成
- フィルタ: 有効なReAct形式 + 妥当な長さ（5-20ステップ） + 自己評価4/5以上
- ALFWorld / TextWorld 関連キーワードの完全排除

**API消費量見積もり**:
- 1,500プロンプト x ~2Kトークン/応答 = ~3Mトークン
- gpt-oss-120b: 無料公開モデルならローカル実行可能（GGUF版）

### Phase 2-4: SFT → GRPO

Option A と同一。ただし SFT データが蒸留トラジェクトリで強化されているため、
ウォームスタート品質が向上。

### Pros / Cons

**Pros**:
- 最も強い開始ポリシー（大規模モデルの知識を活用）
- DB トラジェクトリはSQL実行検証済みで品質保証
- GRPOの学習効率が向上（報酬分散が大きい）

**Cons**:
- API呼び出しコスト・時間
- 大規模モデルのデプロイ環境が必要な場合あり
- 教師モデルのスタイルに過適合するリスク
- Notebook が1つ増える

---

## Option C: Offline DPO（フォールバック）

### 概要

GRPOでOOMが発生した場合、またはオンライン生成が不安定な場合のフォールバック。
SFTモデルでプロンプトごとに複数応答を生成し、報酬でランキングして
chosen/rejected ペアを作成、DPO で学習する。2-3ラウンド反復。

### パイプライン

```
01_sft_qwen25_7b.py (1 epoch)  →  sft_adapter/
            ↓
03_generate_dpo_pairs_v1.py    →  dpo_pairs_round1.jsonl (600-800ペア)
            ↓
02_dpo_qwen25_7b.py            →  dpo_adapter_round1/
            ↓
(反復) 03_generate_dpo_pairs_v1.py  →  dpo_pairs_round2.jsonl
            ↓
(反復) 02_dpo_qwen25_7b.py         →  dpo_adapter_round2/
```

### Phase 1: DPOペア生成

新規: `dataset/notebooks/03_generate_dpo_pairs_v1.py`

各プロンプトに対してSFTモデルから K=8 の応答を温度0.8で生成。
報酬関数（Option Aと同じ）でスコアリングし、最高/最低をペアに。

```json
{
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "高スコア応答"}],
  "rejected": [{"role": "assistant", "content": "低スコア応答"}]
}
```

フィルタ: `chosen_score == rejected_score` のペアは除外（学習信号なし）。

### Phase 2: DPO学習

新規: `training/notebooks/02_dpo_qwen25_7b.py`

| パラメータ | 値 |
|-----------|-----|
| beta (KL coefficient) | 0.1 |
| loss_type | sigmoid |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 8 |
| num_train_epochs | 1（ラウンドごと） |
| learning_rate | 5e-6 |
| max_length | 4096 |
| max_prompt_length | 2048 |
| LoRA rank / alpha | 32 / 64 |

#### VRAM見積もり（Qwen2.5-7B, A100 40GB）

| コンポーネント | VRAM |
|--------------|------|
| 4-bit量子化モデル | ~4 GB |
| LoRAアダプタ | ~0.3 GB |
| リファレンス（PEFTアダプタ無効化） | ~0 GB 追加 |
| オプティマイザ | ~1.5 GB |
| アクティベーション (bs=2, seq=4096) | ~10-14 GB |
| その他 | ~2 GB |
| **合計** | **~18-22 GB** |

### Phase 3: 反復

2-3ラウンド反復。各ラウンドで:
1. 前ラウンドのDPOモデルから新規応答生成
2. 報酬でランキング → 新ペア作成
3. DPO学習

平均報酬が上昇しなくなったら終了。

### Pros / Cons

**Pros**:
- メモリ効率最高（生成はオフライン）
- DPO は安定した学習アルゴリズム
- ペアを事前に検査・品質確認可能

**Cons**:
- オフライン生成 → 分布シフト（古いポリシーの応答で学習）
- 反復が必要（3ラウンド = 生成3回 + 学習3回）
- エージェントタスクではGRPOより効果が低い傾向（最近の文献）

---

## 推奨実装順序

1. **まず Option A を実装・実行**: 最もシンプルで研究的裏付けが強い
2. **GRPOでOOMまたは不安定なら Option C にフォールバック**
3. **API利用可能かつスコア改善余地があれば Option B の蒸留を追加**

## Notebook一覧（全Option共通）

```
dataset/notebooks/
├── 01_build_training_set_v1.py       # SFTデータ（実装済み）
├── 02_build_rl_prompts_v1.py         # GRPOプロンプトデータ（Option A/B）
├── 02_distill_trajectories_v1.py     # 蒸留トラジェクトリ（Option B のみ）
└── 03_generate_dpo_pairs_v1.py       # DPOペア生成（Option C のみ）

training/notebooks/
├── 01_sft_qwen25_7b.py               # SFTウォームスタート（実装済み）
├── 02_grpo_qwen25_7b.py              # GRPO学習 7B（Option A/B）
├── 02_dpo_qwen25_7b.py               # DPO学習 7B（Option C のみ）
└── 03_grpo_qwen3_4b.py               # GRPO学習 4B（Option A/B）
```

## 参考文献

| 論文 | RL関連の知見 |
|------|-------------|
| SCoRe (arXiv:2509.14257) | 7B が GRPO で 72B 教師に匹敵（+6.3 over baseline） |
| DeepSeek-R1 (2025) | コールドスタートSFT → GRPO の2段階が有効 |
| Embodied Planner-R1 (arXiv:2506.23127) | 純RL で ALFWorld 97.78% |
| RAGEN (arXiv:2504.20073) | マルチターンRL の StarPO フレームワーク |
| AgentTuning (arXiv:2310.12823) | SFT + 汎用データ混合がベースライン |
| Agent-FLAN (arXiv:2403.12881) | ネガティブサンプルで幻覚削減 |
| trl GRPOTrainer | https://huggingface.co/docs/trl/main/en/grpo_trainer |
