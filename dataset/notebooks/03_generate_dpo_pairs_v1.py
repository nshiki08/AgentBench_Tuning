# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.5.0",
#     "transformers>=4.46.0",
#     "peft>=0.13.0",
#     "bitsandbytes>=0.44.0",
#     "accelerate>=1.1.0",
#     "datasets",
#     "numpy",
# ]
# ///
"""
Generate DPO Preference Pairs V1 for AgentBench (Plan C: Offline DPO).

Given an SFT-trained LoRA adapter and a prompt dataset, generates K=8 completions
per prompt, scores each with task-specific reward functions, and creates chosen/rejected
pairs for DPO training.

Input:  dataset/output/rl_prompts_v1.jsonl  (from 02_build_rl_prompts_v1.py)
Output: dataset/output/dpo_pairs_v1.jsonl
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# Cell 1: Setup & Configuration
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Generate DPO Preference Pairs V1

        **Plan C: Offline DPO** -- SFT モデルでプロンプトごとに K=8 個の完了を生成し、
        報酬関数でスコアリングして chosen/rejected ペアを作成する。

        ## Pipeline

        1. SFT LoRA アダプタ + ベースモデルをロード (4-bit 量子化)
        2. プロンプトデータセットを読み込み
        3. 各プロンプトに対して K=8 個の完了を生成
        4. タスク別の報酬関数でスコアリング (DB: SQL 実行一致, Env: ReAct 形式準拠)
        5. chosen (最高スコア) / rejected (最低スコア) ペアを選択
        6. JSONL として出力
        """
    )
    return (mo,)


@app.cell
def _():
    import json
    import logging
    import re
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # ========================= Colab detection =========================
    def _is_colab() -> bool:
        try:
            import google.colab  # type: ignore[import-untyped]  # noqa: F401

            return True
        except ImportError:
            return False

    IS_COLAB: bool = _is_colab()

    if IS_COLAB:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch>=2.5.0",
                "transformers>=4.46.0",
                "peft>=0.13.0",
                "bitsandbytes>=0.44.0",
                "accelerate>=1.1.0",
                "datasets",
                "numpy",
            ]
        )

    # ========================= Google Drive mount =========================
    DRIVE_MOUNTED: bool = False
    if IS_COLAB:
        try:
            from google.colab import drive  # type: ignore[import-untyped]

            drive.mount("/content/drive")
            DRIVE_MOUNTED = True
            print("Google Drive mounted at /content/drive")
        except Exception as e:
            print(f"Drive mount skipped: {e}")

    # ========================= Configuration =========================
    # SFT adapter path: adjust to your environment
    # On Colab with Drive: "/content/drive/MyDrive/agentbench_checkpoints/final_adapter"
    # Local: "./sft_qwen25_7b_agentbench/final_adapter"
    SFT_ADAPTER_PATH: str = "./sft_qwen25_7b_agentbench/final_adapter"
    BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

    PROMPTS_PATH: Path = Path("/workspaces/AgentBench_Tuning/dataset/output/rl_prompts_v1.jsonl")
    OUTPUT_DIR: Path = Path("/workspaces/AgentBench_Tuning/dataset/output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH: Path = OUTPUT_DIR / "dpo_pairs_v1.jsonl"
    METADATA_PATH: Path = OUTPUT_DIR / "dpo_pairs_v1_metadata.json"

    # Generation parameters
    NUM_COMPLETIONS: int = 8  # K completions per prompt
    TEMPERATURE: float = 0.8
    TOP_P: float = 0.95
    MAX_NEW_TOKENS: int = 1024
    BATCH_SIZE: int = 4  # prompts to process at once

    # Pair selection threshold
    MIN_SCORE_DIFF: float = 0.1  # minimum score difference between chosen and rejected

    # Reproducibility
    RANDOM_SEED: int = 42

    logger.info("Configuration loaded.")
    logger.info("  SFT adapter  : %s", SFT_ADAPTER_PATH)
    logger.info("  Base model   : %s", BASE_MODEL)
    logger.info("  Prompts      : %s", PROMPTS_PATH)
    logger.info("  Output       : %s", OUTPUT_PATH)
    logger.info("  K completions: %d", NUM_COMPLETIONS)
    logger.info("  Temperature  : %.2f", TEMPERATURE)
    logger.info("  Min score diff: %.2f", MIN_SCORE_DIFF)
    logger.info("  Colab: %s, Drive: %s", IS_COLAB, DRIVE_MOUNTED)

    return (
        BASE_MODEL,
        BATCH_SIZE,
        DRIVE_MOUNTED,
        IS_COLAB,
        MAX_NEW_TOKENS,
        METADATA_PATH,
        MIN_SCORE_DIFF,
        NUM_COMPLETIONS,
        OUTPUT_DIR,
        OUTPUT_PATH,
        Path,
        PROMPTS_PATH,
        RANDOM_SEED,
        SFT_ADAPTER_PATH,
        TEMPERATURE,
        TOP_P,
        json,
        logger,
        np,
        re,
        sqlite3,
        subprocess,
        sys,
    )


# ---------------------------------------------------------------------------
# Cell 2: Load Model
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 2 -- Load SFT Model

        4-bit NF4 量子化でベースモデルをロードし、SFT LoRA アダプタを適用する。
        """
    )
    return ()


@app.cell
def _(BASE_MODEL, Path, SFT_ADAPTER_PATH, logger):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # ========================= GPU check =========================
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name}  |  VRAM: {gpu_mem_gb:.1f} GB")
    else:
        print("WARNING: No CUDA GPU detected. Generation will be extremely slow.")

    # ========================= Quantization config =========================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ========================= Load base model =========================
    logger.info("Loading base model: %s", BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    # ========================= Load tokenizer =========================
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="left",  # left-padding for batched generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ========================= Load SFT LoRA adapter =========================
    adapter_path = Path(SFT_ADAPTER_PATH)
    if adapter_path.exists():
        from peft import PeftModel

        logger.info("Loading SFT LoRA adapter from: %s", SFT_ADAPTER_PATH)
        model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
        model.eval()
        logger.info("SFT adapter loaded successfully.")
    else:
        logger.warning(
            "SFT adapter not found at '%s'. Using base model only. "
            "Completions will reflect the base model (no fine-tuning).",
            SFT_ADAPTER_PATH,
        )

    print(f"Model loaded: {BASE_MODEL}")
    print(f"  pad_token : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"  eos_token : {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  Adapter   : {'loaded' if adapter_path.exists() else 'NOT found (using base)'}")

    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, bnb_config, model, tokenizer, torch


# ---------------------------------------------------------------------------
# Cell 3: Reward Functions
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 3 -- Reward Functions

        タスク別の報酬関数を定義する:

        - **DB Reward**: SQL 実行結果の一致度を計算 (gold SQL が利用可能な場合)
        - **Env Reward**: ReAct 形式への準拠度 + アクション妥当性 + 推論品質
        """
    )
    return ()


@app.cell
def _(logger, re, sqlite3):
    # ===================================================================
    # SQL extraction helpers
    # ===================================================================
    _SQL_KEYWORDS: tuple[str, ...] = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE")

    def extract_sql(completion: str) -> str | None:
        """Extract SQL query from a completion string.

        Tries (in order):
        1. Fenced ```sql ... ``` blocks
        2. Fenced ``` ... ``` blocks containing SELECT/INSERT/UPDATE/DELETE
        3. Raw lines starting with SELECT/INSERT/UPDATE/DELETE/WITH/CREATE
        """
        # Try fenced SQL block
        match = re.search(r"```sql\s*(.*?)```", completion, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            if sql:
                return sql

        # Try generic fenced block with SQL keywords
        match = re.search(r"```\s*(.*?)```", completion, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate.upper().lstrip().startswith(_SQL_KEYWORDS):
                return candidate

        # Try raw SQL lines
        for line in completion.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith(_SQL_KEYWORDS):
                return stripped

        return None

    def execute_sql(db_path: str, sql: str, timeout: float = 5.0) -> set[tuple]:
        """Execute SQL against a SQLite database and return result rows as a set of tuples.

        Returns an empty set on error or timeout.
        """
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.execute("PRAGMA query_only = ON")  # safety: read-only
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            conn.close()
            return set(rows)
        except Exception:
            return set()

    # ===================================================================
    # DB Reward
    # ===================================================================
    def compute_db_reward(completion: str, gold_sql: str | None, db_path: str | None) -> float:
        """Score DB task completion by SQL execution match.

        Scoring:
          +1.0  : exact result match with gold SQL
          0.0-0.5: partial match (Jaccard similarity * 0.5)
          -0.25 : SQL executed but results differ completely
          -0.5  : SQL extraction failed or execution error
           0.0  : gold_sql or db_path not available (fallback to format-only)
        """
        # --- Format-only scoring when gold data is unavailable ---
        if gold_sql is None or db_path is None:
            sql = extract_sql(completion)
            if sql is not None:
                # Give partial credit for producing valid-looking SQL
                return 0.3
            return -0.3  # no SQL found at all

        # --- Full execution-based scoring ---
        sql = extract_sql(completion)
        if sql is None:
            return -0.5  # format penalty: no SQL could be extracted

        try:
            pred_result = execute_sql(db_path, sql)
            gold_result = execute_sql(db_path, gold_sql)

            if not gold_result:
                # Gold SQL itself failed -- fall back to format-only
                if pred_result:
                    return 0.2  # at least the prediction executed
                return 0.0

            if pred_result == gold_result:
                return 1.0  # exact match

            intersection = pred_result & gold_result
            if intersection:
                union = pred_result | gold_result
                jaccard = len(intersection) / len(union)
                return jaccard * 0.5
            else:
                return -0.25  # executed but completely wrong
        except Exception as e:
            logger.debug("SQL execution error: %s", e)
            return -0.5  # execution error

    # ===================================================================
    # Env Reward
    # ===================================================================
    def compute_env_reward(completion: str) -> float:
        """Score env task completion by ReAct format compliance + quality.

        Scoring components (max 1.0):
          (1) ReAct format adherence  : max 0.4
          (2) Action validity         : max 0.3
          (3) Reasoning quality       : max 0.3
        """
        reward = 0.0

        # (1) ReAct format (max 0.4)
        has_thought = bool(re.search(r"Thought:", completion))
        has_action = bool(re.search(r"Action:", completion))
        has_input = bool(re.search(r"Action Input:", completion))
        if has_thought and has_action and has_input:
            reward += 0.4
        elif has_thought or has_action:
            reward += 0.1

        # (2) Action validity (max 0.3)
        valid_actions = [
            "interact", "navigate", "look", "pick", "put", "go",
            "take", "open", "close", "use", "examine", "finish",
            "search", "click", "buy", "execute", "read", "write",
        ]
        action_match = re.search(r"Action:\s*(\w+)", completion)
        if action_match and any(v in action_match.group(1).lower() for v in valid_actions):
            reward += 0.3

        # (3) Reasoning quality (max 0.3)
        thought_match = re.search(r"Thought:\s*(.*?)(?:Action:|$)", completion, re.DOTALL)
        if thought_match:
            thought_text = thought_match.group(1).strip()
            if len(thought_text) > 50:
                reward += 0.3  # substantive reasoning
            elif len(thought_text) > 20:
                reward += 0.2  # moderate reasoning
            elif len(thought_text) > 5:
                reward += 0.1  # minimal reasoning

        return reward

    logger.info("Reward functions defined: compute_db_reward, compute_env_reward")

    return compute_db_reward, compute_env_reward, execute_sql, extract_sql


# ---------------------------------------------------------------------------
# Cell 4: Generation Loop
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 4 -- Generation Loop

        各プロンプトに対して K=8 個の完了を生成し、報酬関数でスコアリングして
        chosen/rejected ペアを選択する。
        """
    )
    return ()


@app.cell
def _(
    BATCH_SIZE,
    MAX_NEW_TOKENS,
    MIN_SCORE_DIFF,
    NUM_COMPLETIONS,
    OUTPUT_PATH,
    PROMPTS_PATH,
    RANDOM_SEED,
    TEMPERATURE,
    TOP_P,
    compute_db_reward,
    compute_env_reward,
    json,
    logger,
    model,
    np,
    tokenizer,
    torch,
):
    import time

    # ========================= Load prompts =========================
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {PROMPTS_PATH}\n"
            "Please run dataset/notebooks/02_build_rl_prompts_v1.py first."
        )

    prompts: list[dict] = []
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    logger.info("Loaded %d prompts from %s", len(prompts), PROMPTS_PATH)

    # Show task type distribution
    task_dist: dict[str, int] = {}
    for p in prompts:
        tt = p.get("task_type", "unknown")
        task_dist[tt] = task_dist.get(tt, 0) + 1
    logger.info("Prompt task distribution: %s", task_dist)

    # ========================= Set seed =========================
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ========================= Generation helpers =========================
    def generate_completions(
        prompt_messages: list[dict[str, str]],
        k: int,
    ) -> list[str]:
        """Generate k completions for a single prompt using the loaded model.

        Args:
            prompt_messages: List of message dicts (system + user turns).
            k: Number of completions to generate.

        Returns:
            List of k completion strings (assistant response only).
        """
        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        input_length = input_ids.shape[1]

        completions: list[str] = []

        with torch.no_grad():
            # Generate k completions at once using num_return_sequences
            # For memory efficiency, generate in sub-batches if k is large
            sub_batch = min(k, 4)
            for start in range(0, k, sub_batch):
                n = min(sub_batch, k - start)
                output_ids = model.generate(
                    input_ids=input_ids.expand(n, -1),
                    attention_mask=attention_mask.expand(n, -1),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                for seq in output_ids:
                    generated_text = tokenizer.decode(seq[input_length:], skip_special_tokens=True)
                    completions.append(generated_text.strip())

        return completions

    def score_completion(
        completion: str,
        task_type: str,
        gold_sql: str | None = None,
        db_path: str | None = None,
    ) -> float:
        """Score a single completion using the appropriate reward function.

        Args:
            completion: The generated text to score.
            task_type: "db" or "env".
            gold_sql: Gold SQL query (for DB tasks, if available).
            db_path: Path to SQLite database (for DB tasks, if available).

        Returns:
            A float reward score.
        """
        if task_type == "db":
            return compute_db_reward(completion, gold_sql, db_path)
        elif task_type == "env":
            return compute_env_reward(completion)
        else:
            # Fallback: basic format reward
            if len(completion.strip()) > 10:
                return 0.1
            return -0.1

    def select_pair(
        scores: list[float],
        min_diff: float,
    ) -> tuple[int, int] | None:
        """Select best (chosen) and worst (rejected) indices from scored completions.

        Args:
            scores: Reward scores for each completion.
            min_diff: Minimum score difference required.

        Returns:
            Tuple of (chosen_idx, rejected_idx) or None if pair doesn't meet threshold.
        """
        if len(scores) < 2:
            return None

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best_idx = sorted_indices[0]
        worst_idx = sorted_indices[-1]

        if scores[best_idx] - scores[worst_idx] < min_diff:
            return None

        return best_idx, worst_idx

    # ========================= Main generation loop =========================
    logger.info("=" * 60)
    logger.info("Starting DPO pair generation")
    logger.info("  Prompts: %d, K: %d, Batch size: %d", len(prompts), NUM_COMPLETIONS, BATCH_SIZE)
    logger.info("=" * 60)

    dpo_pairs: list[dict] = []
    skipped_low_diff = 0
    skipped_error = 0
    all_chosen_scores: list[float] = []
    all_rejected_scores: list[float] = []
    total_start_time = time.time()

    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[batch_start:batch_start + BATCH_SIZE]

        for i, prompt_data in enumerate(batch_prompts):
            global_idx = batch_start + i
            prompt_messages = prompt_data.get("prompt", prompt_data.get("messages", []))
            task_type = prompt_data.get("task_type", "env")
            source = prompt_data.get("source", "unknown")
            gold_sql = prompt_data.get("gold_sql", None)
            db_path = prompt_data.get("db_path", None)

            # Validate prompt
            if not prompt_messages or len(prompt_messages) < 1:
                logger.warning("Prompt %d: empty messages, skipping", global_idx)
                skipped_error += 1
                continue

            try:
                # Generate K completions
                completions = generate_completions(prompt_messages, NUM_COMPLETIONS)

                # Score each completion
                scores = [
                    score_completion(c, task_type, gold_sql=gold_sql, db_path=db_path)
                    for c in completions
                ]

                # Select best/worst pair
                pair = select_pair(scores, MIN_SCORE_DIFF)
                if pair is None:
                    skipped_low_diff += 1
                    if (global_idx + 1) % 50 == 0:
                        logger.info(
                            "  [%d/%d] skipped (low score diff), scores: %s",
                            global_idx + 1,
                            len(prompts),
                            [f"{s:.2f}" for s in scores],
                        )
                    continue

                chosen_idx, rejected_idx = pair

                # Build DPO pair
                dpo_pair = {
                    "prompt": prompt_messages,
                    "chosen": [{"role": "assistant", "content": completions[chosen_idx]}],
                    "rejected": [{"role": "assistant", "content": completions[rejected_idx]}],
                    "chosen_score": round(scores[chosen_idx], 4),
                    "rejected_score": round(scores[rejected_idx], 4),
                    "task_type": task_type,
                    "source": source,
                }
                dpo_pairs.append(dpo_pair)
                all_chosen_scores.append(scores[chosen_idx])
                all_rejected_scores.append(scores[rejected_idx])

            except Exception as e:
                logger.warning("Prompt %d: generation error: %s", global_idx, e)
                skipped_error += 1
                continue

            # Progress logging
            if (global_idx + 1) % 10 == 0 or global_idx == len(prompts) - 1:
                elapsed = time.time() - total_start_time
                prompts_per_sec = (global_idx + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "  [%d/%d] pairs=%d, skipped_diff=%d, skipped_err=%d  (%.2f prompts/s)",
                    global_idx + 1,
                    len(prompts),
                    len(dpo_pairs),
                    skipped_low_diff,
                    skipped_error,
                    prompts_per_sec,
                )

    total_elapsed = time.time() - total_start_time
    logger.info("=" * 60)
    logger.info("Generation complete in %.1f seconds", total_elapsed)
    logger.info("  Total pairs       : %d", len(dpo_pairs))
    logger.info("  Skipped (low diff): %d", skipped_low_diff)
    logger.info("  Skipped (error)   : %d", skipped_error)
    logger.info("=" * 60)

    # ========================= Write output =========================
    n_written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info("Wrote %d DPO pairs to %s", n_written, OUTPUT_PATH)

    return (
        all_chosen_scores,
        all_rejected_scores,
        dpo_pairs,
        generate_completions,
        n_written,
        score_completion,
        select_pair,
        skipped_error,
        skipped_low_diff,
        time,
        total_elapsed,
    )


# ---------------------------------------------------------------------------
# Cell 5: Statistics & Export
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 5 -- Statistics & Export

        生成された DPO ペアの統計情報を表示し、メタデータを保存する。
        """
    )
    return ()


@app.cell
def _(
    BASE_MODEL,
    METADATA_PATH,
    MIN_SCORE_DIFF,
    NUM_COMPLETIONS,
    OUTPUT_PATH,
    RANDOM_SEED,
    SFT_ADAPTER_PATH,
    TEMPERATURE,
    TOP_P,
    all_chosen_scores,
    all_rejected_scores,
    dpo_pairs,
    json,
    logger,
    n_written,
    np,
    skipped_error,
    skipped_low_diff,
    total_elapsed,
):
    # ========================= Compute statistics =========================
    if not dpo_pairs:
        print("No DPO pairs were generated. Check the generation log above for errors.")
        stats: dict = {"total_pairs": 0, "error": "no pairs generated"}
    else:
        chosen_arr = np.array(all_chosen_scores)
        rejected_arr = np.array(all_rejected_scores)
        diff_arr = chosen_arr - rejected_arr

        # Task type distribution
        task_type_dist: dict[str, int] = {}
        source_dist: dict[str, int] = {}
        chosen_lens: list[int] = []
        rejected_lens: list[int] = []

        for pair in dpo_pairs:
            tt = pair["task_type"]
            src = pair["source"]
            task_type_dist[tt] = task_type_dist.get(tt, 0) + 1
            source_dist[src] = source_dist.get(src, 0) + 1
            chosen_lens.append(len(pair["chosen"][0]["content"]))
            rejected_lens.append(len(pair["rejected"][0]["content"]))

        chosen_len_arr = np.array(chosen_lens)
        rejected_len_arr = np.array(rejected_lens)

        stats = {
            "total_pairs": len(dpo_pairs),
            "skipped_low_score_diff": skipped_low_diff,
            "skipped_error": skipped_error,
            "generation_time_seconds": round(total_elapsed, 1),
            "chosen_score": {
                "mean": round(float(np.mean(chosen_arr)), 4),
                "std": round(float(np.std(chosen_arr)), 4),
                "min": round(float(np.min(chosen_arr)), 4),
                "max": round(float(np.max(chosen_arr)), 4),
                "median": round(float(np.median(chosen_arr)), 4),
            },
            "rejected_score": {
                "mean": round(float(np.mean(rejected_arr)), 4),
                "std": round(float(np.std(rejected_arr)), 4),
                "min": round(float(np.min(rejected_arr)), 4),
                "max": round(float(np.max(rejected_arr)), 4),
                "median": round(float(np.median(rejected_arr)), 4),
            },
            "score_diff": {
                "mean": round(float(np.mean(diff_arr)), 4),
                "std": round(float(np.std(diff_arr)), 4),
                "min": round(float(np.min(diff_arr)), 4),
                "max": round(float(np.max(diff_arr)), 4),
                "median": round(float(np.median(diff_arr)), 4),
            },
            "chosen_length_chars": {
                "mean": round(float(np.mean(chosen_len_arr)), 1),
                "median": round(float(np.median(chosen_len_arr)), 1),
                "min": int(np.min(chosen_len_arr)),
                "max": int(np.max(chosen_len_arr)),
            },
            "rejected_length_chars": {
                "mean": round(float(np.mean(rejected_len_arr)), 1),
                "median": round(float(np.median(rejected_len_arr)), 1),
                "min": int(np.min(rejected_len_arr)),
                "max": int(np.max(rejected_len_arr)),
            },
            "task_type_distribution": dict(sorted(task_type_dist.items())),
            "source_distribution": dict(sorted(source_dist.items())),
        }

        # Pretty-print statistics
        print("=" * 60)
        print("DPO Pair Generation Statistics")
        print("=" * 60)
        print(f"  Total pairs generated  : {stats['total_pairs']}")
        print(f"  Skipped (low diff)     : {stats['skipped_low_score_diff']}")
        print(f"  Skipped (error)        : {stats['skipped_error']}")
        print(f"  Generation time        : {stats['generation_time_seconds']:.1f}s")
        print()
        print("  Chosen scores:")
        cs = stats["chosen_score"]
        print(f"    mean={cs['mean']:.4f}  std={cs['std']:.4f}  median={cs['median']:.4f}")
        print(f"    min={cs['min']:.4f}  max={cs['max']:.4f}")
        print()
        print("  Rejected scores:")
        rs = stats["rejected_score"]
        print(f"    mean={rs['mean']:.4f}  std={rs['std']:.4f}  median={rs['median']:.4f}")
        print(f"    min={rs['min']:.4f}  max={rs['max']:.4f}")
        print()
        print("  Score difference (chosen - rejected):")
        sd = stats["score_diff"]
        print(f"    mean={sd['mean']:.4f}  std={sd['std']:.4f}  median={sd['median']:.4f}")
        print(f"    min={sd['min']:.4f}  max={sd['max']:.4f}")
        print()
        print("  Chosen completion length (chars):")
        cl = stats["chosen_length_chars"]
        print(f"    mean={cl['mean']:.1f}  median={cl['median']:.1f}  min={cl['min']}  max={cl['max']}")
        print()
        print("  Rejected completion length (chars):")
        rl = stats["rejected_length_chars"]
        print(f"    mean={rl['mean']:.1f}  median={rl['median']:.1f}  min={rl['min']}  max={rl['max']}")
        print()
        print("  Task type distribution:")
        for tt, count in sorted(task_type_dist.items()):
            print(f"    {tt}: {count}")
        print()
        print("  Source distribution:")
        for src, count in sorted(source_dist.items()):
            print(f"    {src}: {count}")

    # ========================= Write metadata =========================
    metadata = {
        "dataset_name": "dpo_pairs_v1",
        "version": "1.0.0",
        "description": (
            "DPO preference pairs generated from SFT model completions "
            "for AgentBench DB Bench and environment tasks. "
            "Each pair contains a chosen (highest-scoring) and rejected (lowest-scoring) "
            "completion, scored by task-specific reward functions."
        ),
        "creation_date": "2026-02-14",
        "pipeline": {
            "base_model": BASE_MODEL,
            "sft_adapter": SFT_ADAPTER_PATH,
            "num_completions_per_prompt": NUM_COMPLETIONS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "min_score_diff": MIN_SCORE_DIFF,
            "random_seed": RANDOM_SEED,
        },
        "output_format": {
            "prompt": "list of {role, content} dicts (system + user turns)",
            "chosen": "list with single {role: 'assistant', content: ...} dict",
            "rejected": "list with single {role: 'assistant', content: ...} dict",
            "chosen_score": "float reward score for chosen completion",
            "rejected_score": "float reward score for rejected completion",
            "task_type": "db | env",
            "source": "source dataset identifier",
        },
        "reward_functions": {
            "db": "SQL execution match (exact=1.0, partial=Jaccard*0.5, format-only=0.3/-0.3, error=-0.5)",
            "env": "ReAct format (0.4) + action validity (0.3) + reasoning quality (0.3)",
        },
        "statistics": stats,
        "output_file": str(OUTPUT_PATH),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Metadata written to %s", METADATA_PATH)

    # ========================= Verification =========================
    if n_written > 0:
        verified = 0
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                assert "prompt" in obj, "Missing 'prompt' field"
                assert "chosen" in obj, "Missing 'chosen' field"
                assert "rejected" in obj, "Missing 'rejected' field"
                assert "chosen_score" in obj, "Missing 'chosen_score' field"
                assert "rejected_score" in obj, "Missing 'rejected_score' field"
                assert "task_type" in obj, "Missing 'task_type' field"
                assert isinstance(obj["prompt"], list), "'prompt' must be a list"
                assert isinstance(obj["chosen"], list), "'chosen' must be a list"
                assert isinstance(obj["rejected"], list), "'rejected' must be a list"
                assert obj["chosen_score"] > obj["rejected_score"], "chosen_score must be > rejected_score"
                verified += 1

        print(f"\nVerification: {verified} / {n_written} pairs OK")
    else:
        print("\nNo pairs to verify.")

    print(f"\nOutput : {OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Done.")

    return metadata, stats


if __name__ == "__main__":
    app.run()
