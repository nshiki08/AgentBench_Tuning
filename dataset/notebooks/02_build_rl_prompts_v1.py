# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "datasets>=3.0.0",
#     "pandas>=2.2.0",
#     "numpy>=1.26.0",
#     "tqdm",
# ]
# ///
"""
Build RL Prompts V1 -- Prompt-only dataset for GRPO/DPO training.

Plan C (Offline DPO): generate a set of prompts (system + user) that an SFT
model will later complete multiple times.  Completions are scored by a reward
function (execution accuracy for DB, env reward for Env) and the best/worst
pairs become DPO training data.

Output schema (JSONL):
  {
    "prompt":    [{"role": "system", ...}, {"role": "user", ...}],
    "task_type": "db" | "env",
    "source":    "spider" | "sparc" | "cosql" | "webshop" | "scienceworld" | "synthetic_react",
    "gold_sql":  "SELECT ..." | null,
    "db_path":   "database/{db_id}/{db_id}.sqlite" | null
  }

CRITICAL: No AgentBench evaluation data (ALFWorld, TextWorld) is included.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# Cell 1: Documentation header
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Build RL Prompts V1

        AgentBench DB / Env スコア向上のための RL (GRPO/DPO) 用プロンプトデータセットを構築する。

        **Plan C (Offline DPO)**: SFT 済みモデルで複数 completion を生成し、
        reward 関数でスコアリングして DPO ペアを作る。このノートブックはその入力プロンプト集を作成する。

        | カテゴリ | 目標件数 | ソース |
        |----------|----------|--------|
        | DB       | 500-800  | Spider, SParC, CoSQL |
        | Env      | 300-500  | ETO (WebShop, ScienceWorld), Synthetic ReAct templates |

        **禁止**: ALFWorld / TextWorld のデータを含めない (AgentBench 評価データの混入防止)。
        """
    )
    return (mo,)


# ---------------------------------------------------------------------------
# Cell 2: Configuration & Constants
# ---------------------------------------------------------------------------
@app.cell
def _():
    import hashlib
    import json
    import logging
    import random
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # --------------- reproducibility ---------------
    RANDOM_SEED: int = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --------------- output paths ---------------
    OUTPUT_DIR: Path = Path("/workspaces/AgentBench_Tuning/dataset/output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL: Path = OUTPUT_DIR / "rl_prompts_v1.jsonl"
    METADATA_JSON: Path = OUTPUT_DIR / "rl_prompts_v1_metadata.json"

    # --------------- target counts ---------------
    TARGET_DB_MIN: int = 500
    TARGET_DB_MAX: int = 800
    TARGET_ENV_MIN: int = 300
    TARGET_ENV_MAX: int = 500

    # Sub-source budgets (within DB range)
    DB_BUDGET: dict[str, int] = {
        "spider": 300,
        "sparc": 250,
        "cosql": 250,
    }
    # Sub-source budgets (within Env range)
    ENV_BUDGET: dict[str, int] = {
        "webshop": 150,
        "scienceworld": 150,
        "synthetic_react": 200,
    }

    # --------------- quality thresholds ---------------
    MIN_USER_CONTENT_LENGTH: int = 20  # minimum characters for user message

    # --------------- banned keywords for contamination ---------------
    BANNED_ENV_KEYWORDS: list[str] = [
        "alfworld",
        "alf_world",
        "textworld",
        "text_world",
        "tw_",
        "tw-",
    ]

    # --------------- system prompts ---------------
    DB_SYSTEM_PROMPT: str = (
        "You are an expert database assistant. Given a database schema, write SQL queries "
        "to answer user questions. Think step by step before writing SQL. "
        "Format your SQL in ```sql\n...\n``` blocks."
    )

    ENV_SYSTEM_PROMPT: str = (
        "You are an autonomous agent operating in an interactive environment. "
        "Your goal is to complete the given task by reasoning step by step and taking actions.\n\n"
        "Use the following format:\n"
        "Thought: [your reasoning]\n"
        "Action: [action to take]\n"
        "Action Input: [input for the action]"
    )

    logger.info("Configuration loaded. Output -> %s", OUTPUT_JSONL)

    return (
        Any,
        BANNED_ENV_KEYWORDS,
        DB_BUDGET,
        DB_SYSTEM_PROMPT,
        ENV_BUDGET,
        ENV_SYSTEM_PROMPT,
        METADATA_JSON,
        MIN_USER_CONTENT_LENGTH,
        OUTPUT_DIR,
        OUTPUT_JSONL,
        Path,
        RANDOM_SEED,
        TARGET_DB_MAX,
        TARGET_DB_MIN,
        TARGET_ENV_MAX,
        TARGET_ENV_MIN,
        hashlib,
        json,
        logger,
        np,
        pd,
        random,
    )


# ---------------------------------------------------------------------------
# Cell 3: Shared utilities
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 3 -- Shared Utilities

        Contamination checking, content hashing, and prompt validation functions.
        """
    )
    return ()


@app.cell
def _(Any, BANNED_ENV_KEYWORDS, MIN_USER_CONTENT_LENGTH, hashlib, json, logger):
    def contamination_check_text(text: str) -> bool:
        """Return True if text contains any banned keyword (i.e. is contaminated)."""
        text_lower = text.lower()
        for kw in BANNED_ENV_KEYWORDS:
            if kw in text_lower:
                return True
        return False

    def contamination_check_prompt(prompt_messages: list[dict[str, str]]) -> bool:
        """Return True if any message in the prompt contains banned keywords."""
        blob = json.dumps(prompt_messages, ensure_ascii=False).lower()
        for kw in BANNED_ENV_KEYWORDS:
            if kw in blob:
                logger.warning("Contamination keyword '%s' found in prompt -- skipping", kw)
                return True
        return False

    def content_hash(prompt_messages: list[dict[str, str]]) -> str:
        """Deterministic SHA-256 hash for deduplication."""
        blob = json.dumps(prompt_messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def validate_prompt(
        prompt_messages: list[dict[str, str]],
        task_type: str,
        source: str,
        gold_sql: str | None = None,
        db_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Validate and package a single RL prompt record. Returns None if invalid."""
        # Must have system + user (at least 2 messages)
        if not prompt_messages or len(prompt_messages) < 2:
            return None

        # Check roles: first must be system, last must be user
        if prompt_messages[0]["role"] != "system":
            return None
        if prompt_messages[-1]["role"] != "user":
            return None

        # Check user message length
        user_content = prompt_messages[-1].get("content", "")
        if len(user_content.strip()) < MIN_USER_CONTENT_LENGTH:
            return None

        # Triple-layer contamination check
        if contamination_check_prompt(prompt_messages):
            return None
        if gold_sql and contamination_check_text(gold_sql):
            return None
        if db_path and contamination_check_text(db_path):
            return None

        return {
            "prompt": prompt_messages,
            "task_type": task_type,
            "source": source,
            "gold_sql": gold_sql,
            "db_path": db_path,
        }

    logger.info("Shared utilities loaded.")

    return (
        contamination_check_prompt,
        contamination_check_text,
        content_hash,
        validate_prompt,
    )


# ---------------------------------------------------------------------------
# Cell 4: DB Prompt Extraction
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 4 -- DB Prompt Extraction

        Extract question + schema from Spider, SParC, and CoSQL training sets.
        Each prompt is a system message with DB instructions plus a user message
        containing the database schema and the natural-language question.

        Metadata fields `gold_sql` and `db_path` are preserved for the reward function.
        """
    )
    return ()


@app.cell
def _(Any, DB_BUDGET, DB_SYSTEM_PROMPT, logger, random, validate_prompt):
    from datasets import load_dataset

    # ===================================================================
    # 4-a: Spider prompt extraction
    # ===================================================================
    def _format_spider_schema(row: dict[str, Any]) -> str:
        """Build a human-readable schema string from Spider row metadata."""
        db_id = row.get("db_id", "unknown")
        # Spider datasets may provide schema in different formats.
        # Try structured fields first, fall back to a simple db_id reference.
        table_names = row.get("table_names", row.get("table_names_original", []))
        column_names = row.get("column_names", row.get("column_names_original", []))

        if table_names and column_names:
            # column_names is typically [(table_idx, col_name), ...]
            tables: dict[str, list[str]] = {}
            for entry in column_names:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    tidx, cname = entry
                    if isinstance(tidx, int) and 0 <= tidx < len(table_names):
                        tname = table_names[tidx]
                        tables.setdefault(tname, []).append(str(cname))
                    elif tidx == -1:
                        # Special index for * column; skip
                        continue
            if tables:
                lines = []
                for tname, cols in tables.items():
                    lines.append(f"Table: {tname}")
                    lines.append(f"  Columns: {', '.join(cols)}")
                return "\n".join(lines)

        # Fallback: just use db_id
        return f"Database: {db_id}"

    def load_spider_prompts(budget: int) -> list[dict[str, Any]]:
        """Extract prompts from xlangai/spider training split."""
        try:
            ds = load_dataset("xlangai/spider", split="train", trust_remote_code=True)
            logger.info("Loaded Spider (%d rows)", len(ds))
        except Exception as e:
            logger.error("Could not load Spider: %s", e)
            return []

        samples: list[dict[str, Any]] = []
        indices = list(range(len(ds)))
        random.shuffle(indices)

        for idx in indices:
            if len(samples) >= budget:
                break
            row = ds[idx]
            question = row.get("question", "")
            query = row.get("query", "")
            db_id = row.get("db_id", "unknown")

            if not question or not query:
                continue

            schema_text = _format_spider_schema(row)
            user_msg = f"Database schema:\n{schema_text}\n\nQuestion: {question}"

            prompt_messages: list[dict[str, str]] = [
                {"role": "system", "content": DB_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            db_path = f"database/{db_id}/{db_id}.sqlite"
            sample = validate_prompt(
                prompt_messages,
                task_type="db",
                source="spider",
                gold_sql=query,
                db_path=db_path,
            )
            if sample is not None:
                samples.append(sample)

        logger.info("Spider prompts: collected %d / %d budget", len(samples), budget)
        return samples

    # ===================================================================
    # 4-b: SParC prompt extraction
    # ===================================================================
    def load_sparc_prompts(budget: int) -> list[dict[str, Any]]:
        """Extract prompts from SParC training split.

        SParC is multi-turn; we extract each individual question within an
        interaction as a separate prompt, including prior context in the user
        message to preserve conversational context.
        """
        candidates: list[str] = ["aherntech/sparc", "xlangai/sparc"]
        ds = None
        for name in candidates:
            try:
                ds = load_dataset(name, split="train", trust_remote_code=True)
                logger.info("Loaded SParC from '%s' (%d rows)", name, len(ds))
                break
            except Exception:
                logger.warning("Could not load '%s', trying next...", name)
        if ds is None:
            logger.error("SParC unavailable from all candidates. Returning empty.")
            return []

        samples: list[dict[str, Any]] = []
        indices = list(range(len(ds)))
        random.shuffle(indices)

        for idx in indices:
            if len(samples) >= budget:
                break
            row = ds[idx]
            db_id = row.get("database_id", row.get("db_id", "unknown"))
            utterances = row.get("utterances", row.get("question", []))
            queries = row.get("query", row.get("sql", []))

            if not utterances or not queries:
                continue
            if isinstance(utterances, str):
                utterances = [utterances]
            if isinstance(queries, str):
                queries = [queries]

            # Extract each turn as a separate prompt with prior context
            for turn_idx, (utt, sql) in enumerate(zip(utterances, queries)):
                if len(samples) >= budget:
                    break

                # Build context from prior turns
                context_parts: list[str] = []
                if turn_idx > 0:
                    for prev_idx in range(turn_idx):
                        if prev_idx < len(utterances):
                            context_parts.append(f"Previous question {prev_idx + 1}: {utterances[prev_idx]}")

                user_content = f"Database: {db_id}\n"
                if context_parts:
                    user_content += "\n".join(context_parts) + "\n\n"
                user_content += f"Question: {utt}"

                prompt_messages: list[dict[str, str]] = [
                    {"role": "system", "content": DB_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                db_path = f"database/{db_id}/{db_id}.sqlite"
                sample = validate_prompt(
                    prompt_messages,
                    task_type="db",
                    source="sparc",
                    gold_sql=sql,
                    db_path=db_path,
                )
                if sample is not None:
                    samples.append(sample)

        logger.info("SParC prompts: collected %d / %d budget", len(samples), budget)
        return samples

    # ===================================================================
    # 4-c: CoSQL prompt extraction
    # ===================================================================
    def load_cosql_prompts(budget: int) -> list[dict[str, Any]]:
        """Extract prompts from CoSQL training split.

        Similar to SParC, CoSQL is conversational. We extract each question
        with its conversational context.
        """
        candidates: list[str] = ["aherntech/cosql", "xlangai/cosql", "parkervg/cosql"]
        ds = None
        for name in candidates:
            try:
                ds = load_dataset(name, split="train", trust_remote_code=True)
                logger.info("Loaded CoSQL from '%s' (%d rows)", name, len(ds))
                break
            except Exception:
                logger.warning("Could not load '%s', trying next...", name)
        if ds is None:
            logger.error("CoSQL unavailable. Returning empty.")
            return []

        samples: list[dict[str, Any]] = []
        indices = list(range(len(ds)))
        random.shuffle(indices)

        for idx in indices:
            if len(samples) >= budget:
                break
            row = ds[idx]
            db_id = row.get("database_id", row.get("db_id", "unknown"))
            utterances = row.get("utterances", row.get("question", []))
            queries = row.get("query", row.get("sql", []))

            if not utterances or not queries:
                continue
            if isinstance(utterances, str):
                utterances = [utterances]
            if isinstance(queries, str):
                queries = [queries]

            for turn_idx, (utt, sql) in enumerate(zip(utterances, queries)):
                if len(samples) >= budget:
                    break

                context_parts: list[str] = []
                if turn_idx > 0:
                    for prev_idx in range(turn_idx):
                        if prev_idx < len(utterances):
                            context_parts.append(f"Previous question {prev_idx + 1}: {utterances[prev_idx]}")

                user_content = f"Database: {db_id}\n"
                if context_parts:
                    user_content += "\n".join(context_parts) + "\n\n"
                user_content += f"Question: {utt}"

                prompt_messages: list[dict[str, str]] = [
                    {"role": "system", "content": DB_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                db_path = f"database/{db_id}/{db_id}.sqlite"
                sample = validate_prompt(
                    prompt_messages,
                    task_type="db",
                    source="cosql",
                    gold_sql=sql,
                    db_path=db_path,
                )
                if sample is not None:
                    samples.append(sample)

        logger.info("CoSQL prompts: collected %d / %d budget", len(samples), budget)
        return samples

    # ===================================================================
    # Run all DB loaders
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Starting DB prompt extraction")
    logger.info("=" * 60)

    db_prompts_spider: list[dict[str, Any]] = load_spider_prompts(DB_BUDGET["spider"])
    db_prompts_sparc: list[dict[str, Any]] = load_sparc_prompts(DB_BUDGET["sparc"])
    db_prompts_cosql: list[dict[str, Any]] = load_cosql_prompts(DB_BUDGET["cosql"])

    all_db_prompts: list[dict[str, Any]] = db_prompts_spider + db_prompts_sparc + db_prompts_cosql
    logger.info("Total DB prompts: %d", len(all_db_prompts))

    return (
        all_db_prompts,
        db_prompts_cosql,
        db_prompts_sparc,
        db_prompts_spider,
        load_cosql_prompts,
        load_dataset,
        load_sparc_prompts,
        load_spider_prompts,
    )


# ---------------------------------------------------------------------------
# Cell 5: Env Prompt Extraction
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 5 -- Environment Prompt Extraction

        Extract task descriptions for interactive environments.

        **Sources**:
        - `agent-eto/eto-sft-trajectory`: extract initial prompts from **WebShop** and
          **ScienceWorld** entries ONLY (ALFWorld/TextWorld filtered out).
        - Synthetic templates: diverse household, navigation, and tool-use task prompts
          (30+ unique templates with parameterized variation).
        """
    )
    return ()


@app.cell
def _(Any, BANNED_ENV_KEYWORDS, ENV_BUDGET, ENV_SYSTEM_PROMPT, json, load_dataset, logger, random, validate_prompt):
    # ===================================================================
    # 5-a: ETO trajectory prompt extraction (WebShop + ScienceWorld only)
    # ===================================================================
    def _is_alfworld_or_textworld(row: dict[str, Any]) -> bool:
        """Return True if the row belongs to ALFWorld or TextWorld (must be excluded)."""
        for field in ["task_type", "source", "environment", "env", "dataset", "domain", "task"]:
            val = str(row.get(field, "")).lower()
            for kw in BANNED_ENV_KEYWORDS:
                if kw in val:
                    return True
        # Also check message content
        messages_raw = row.get("messages", row.get("conversations", []))
        if messages_raw:
            blob = json.dumps(messages_raw).lower() if not isinstance(messages_raw, str) else messages_raw.lower()
            for kw in BANNED_ENV_KEYWORDS:
                if kw in blob:
                    return True
        return False

    def _classify_eto_source(row: dict[str, Any]) -> str | None:
        """Classify an ETO row as 'webshop', 'scienceworld', or None (skip)."""
        for field in ["task_type", "source", "environment", "env", "dataset", "domain", "task"]:
            val = str(row.get(field, "")).lower()
            if "webshop" in val or "web_shop" in val:
                return "webshop"
            if "science" in val or "sciworld" in val:
                return "scienceworld"
        # Heuristic: check message content
        messages_raw = row.get("messages", row.get("conversations", []))
        blob = ""
        if messages_raw:
            blob = json.dumps(messages_raw).lower() if not isinstance(messages_raw, str) else messages_raw.lower()
        if "webshop" in blob or "web_shop" in blob or ("buy" in blob and "product" in blob):
            return "webshop"
        if "science" in blob or "experiment" in blob or "hypothesis" in blob:
            return "scienceworld"
        return None

    def _extract_first_user_message(row: dict[str, Any]) -> str | None:
        """Extract the first user message from an ETO trajectory row as the task prompt."""
        raw = row.get("messages", row.get("conversations", []))
        if not raw or not isinstance(raw, list):
            return None

        for m in raw:
            if not isinstance(m, dict):
                continue
            role = m.get("role", m.get("from", "")).lower()
            content = m.get("content", m.get("value", ""))
            if role in ("user", "human", "prompter") and content:
                return str(content).strip()
        return None

    def load_eto_prompts(webshop_budget: int, sciworld_budget: int) -> list[dict[str, Any]]:
        """Load agent-eto/eto-sft-trajectory and extract initial prompts only."""
        try:
            ds = load_dataset("agent-eto/eto-sft-trajectory", split="train", trust_remote_code=True)
            logger.info("Loaded eto-sft-trajectory (%d rows)", len(ds))
        except Exception as e:
            logger.error("Could not load eto-sft-trajectory: %s", e)
            return []

        webshop_samples: list[dict[str, Any]] = []
        sciworld_samples: list[dict[str, Any]] = []

        indices = list(range(len(ds)))
        random.shuffle(indices)

        for idx in indices:
            if len(webshop_samples) >= webshop_budget and len(sciworld_samples) >= sciworld_budget:
                break
            row = ds[idx]

            # CRITICAL: exclude ALFWorld / TextWorld
            if _is_alfworld_or_textworld(row):
                continue

            source_type = _classify_eto_source(row)
            if source_type is None:
                continue

            user_msg = _extract_first_user_message(row)
            if not user_msg:
                continue

            prompt_messages: list[dict[str, str]] = [
                {"role": "system", "content": ENV_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            sample = validate_prompt(
                prompt_messages,
                task_type="env",
                source=source_type,
                gold_sql=None,
                db_path=None,
            )
            if sample is None:
                continue

            if source_type == "webshop" and len(webshop_samples) < webshop_budget:
                webshop_samples.append(sample)
            elif source_type == "scienceworld" and len(sciworld_samples) < sciworld_budget:
                sciworld_samples.append(sample)

        logger.info(
            "ETO prompts: WebShop=%d/%d, ScienceWorld=%d/%d",
            len(webshop_samples), webshop_budget,
            len(sciworld_samples), sciworld_budget,
        )
        return webshop_samples + sciworld_samples

    # ===================================================================
    # 5-b: Synthetic environment prompt templates (30+ diverse templates)
    # ===================================================================

    # --- Household task prompts ---
    _HOUSEHOLD_PROMPTS: list[str] = [
        "Clean the kitchen counter and put all the dirty dishes into the dishwasher.",
        "Find the TV remote control in the living room and place it on the coffee table.",
        "Organize the bookshelf by sorting books alphabetically and removing any magazines.",
        "Set the dining table for four people with plates, glasses, utensils, and napkins.",
        "Water all the plants in the apartment, starting with the ones in the living room.",
        "Take the laundry from the washing machine, fold it, and put it in the bedroom closet.",
        "Empty the trash cans in every room and replace them with fresh bags.",
        "Prepare a sandwich with bread, cheese, and lettuce, then serve it on a clean plate.",
        "Find the red jacket in the bedroom and hang it in the hallway closet.",
        "Vacuum the living room carpet and then mop the kitchen floor.",
        "Collect all the toys from the floor and put them in the toy box in the children's room.",
        "Make the bed with fresh sheets and place the pillows neatly.",
    ]

    # --- Navigation task prompts ---
    _NAVIGATION_PROMPTS: list[str] = [
        "Navigate from the building entrance to the conference room on the third floor.",
        "Find the nearest pharmacy starting from the central park and go there.",
        "Deliver the package to apartment 4B in the residential building on Oak Street.",
        "Walk from the train station to the public library using the shortest route.",
        "Navigate through the office building to find the IT department on the second floor.",
        "Go from the parking garage to the rooftop garden, using the service elevator.",
        "Find the emergency exit on the current floor and navigate to the ground-level assembly point.",
        "Walk from the hotel lobby to room 305, picking up the room key from reception first.",
        "Navigate from the campus entrance to the chemistry lab in Building C.",
        "Find the lost-and-found office in the shopping mall starting from the main entrance.",
    ]

    # --- Tool-use task prompts ---
    _TOOL_USE_PROMPTS: list[str] = [
        "Search for today's weather forecast and send a summary to the team via email.",
        "Read the shopping list from shopping_list.txt, calculate the total cost, and save the result to total.txt.",
        "Look up the current stock price of ACME Corp and write a brief summary report.",
        "Translate the contents of document.txt from English to Japanese and save the translation.",
        "Download the latest sales data from the API, compute monthly averages, and save as a CSV file.",
        "Search the knowledge base for articles about machine learning and create a reading list.",
        "Read the error log file, identify the most frequent error, and draft a bug report.",
        "Check the calendar for tomorrow's meetings and send reminder emails to all participants.",
        "Compress all image files in the uploads folder and move them to the archive directory.",
        "Query the database for all orders placed this week and generate a summary report.",
    ]

    # --- Science experiment task prompts ---
    _SCIENCE_PROMPTS: list[str] = [
        "Determine the boiling point of water at different altitudes by conducting experiments in the lab.",
        "Test whether a plant grows faster under blue light or red light over a one-week period.",
        "Mix sodium bicarbonate with vinegar and measure the volume of gas produced.",
        "Measure the electrical conductivity of salt water at various concentrations.",
        "Observe the effect of temperature on the rate of dissolving sugar in water.",
        "Grow bacteria cultures on agar plates and identify which antibiotic is most effective.",
        "Build a simple circuit with a battery, switch, and light bulb to demonstrate Ohm's law.",
        "Investigate how the angle of incidence affects the angle of reflection using a mirror and laser pointer.",
    ]

    def _parameterize_household(template: str) -> str:
        """Add slight variation to a household task prompt."""
        rooms = ["living room", "bedroom", "kitchen", "bathroom", "study", "hallway"]
        objects = ["a glass of water", "the newspaper", "a blanket", "the keys", "an umbrella"]
        # Append a minor sub-task for diversity
        if random.random() < 0.4:
            extra = random.choice([
                f" Also, check the {random.choice(rooms)} for anything out of place.",
                f" While you are at it, bring {random.choice(objects)} to the {random.choice(rooms)}.",
                " Report what you see along the way.",
            ])
            return template + extra
        return template

    def _parameterize_navigation(template: str) -> str:
        """Add slight variation to a navigation task prompt."""
        if random.random() < 0.35:
            extras = [
                " Avoid the construction area on the second floor.",
                " The main elevator is out of service; use the stairs or freight elevator.",
                " Ask the security guard for directions if you get lost.",
                " Note any closed doors or restricted areas you encounter.",
            ]
            return template + random.choice(extras)
        return template

    def _parameterize_tool_use(template: str) -> str:
        """Add slight variation to a tool-use task prompt."""
        if random.random() < 0.3:
            extras = [
                " Log every step you take for auditing purposes.",
                " If any tool call fails, retry once before reporting the error.",
                " Prioritize accuracy over speed.",
            ]
            return template + random.choice(extras)
        return template

    def _parameterize_science(template: str) -> str:
        """Add slight variation to a science experiment task prompt."""
        if random.random() < 0.35:
            extras = [
                " Record all measurements carefully in a data table.",
                " Repeat the experiment three times for statistical reliability.",
                " Make sure to wear appropriate safety equipment.",
                " Write a brief conclusion summarizing your findings.",
            ]
            return template + random.choice(extras)
        return template

    def generate_synthetic_env_prompts(budget: int) -> list[dict[str, Any]]:
        """Generate synthetic environment prompts from diverse templates with parameterized variation."""
        all_template_groups: list[tuple[list[str], Any]] = [
            (_HOUSEHOLD_PROMPTS, _parameterize_household),
            (_NAVIGATION_PROMPTS, _parameterize_navigation),
            (_TOOL_USE_PROMPTS, _parameterize_tool_use),
            (_SCIENCE_PROMPTS, _parameterize_science),
        ]

        samples: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = budget * 10

        while len(samples) < budget and attempts < max_attempts:
            attempts += 1
            templates, parameterizer = random.choice(all_template_groups)
            base_prompt = random.choice(templates)
            varied_prompt = parameterizer(base_prompt)

            # Wrap in env system prompt
            user_content = f"Task: {varied_prompt}\n\nYou are in an interactive environment. Complete the task."
            prompt_messages: list[dict[str, str]] = [
                {"role": "system", "content": ENV_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            sample = validate_prompt(
                prompt_messages,
                task_type="env",
                source="synthetic_react",
                gold_sql=None,
                db_path=None,
            )
            if sample is not None:
                samples.append(sample)

        logger.info("Synthetic env prompts: generated %d / %d budget", len(samples), budget)
        return samples

    # ===================================================================
    # Run all environment prompt loaders
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Starting Environment prompt extraction")
    logger.info("=" * 60)

    env_prompts_eto: list[dict[str, Any]] = load_eto_prompts(
        webshop_budget=ENV_BUDGET["webshop"],
        sciworld_budget=ENV_BUDGET["scienceworld"],
    )
    env_prompts_synthetic: list[dict[str, Any]] = generate_synthetic_env_prompts(ENV_BUDGET["synthetic_react"])

    all_env_prompts: list[dict[str, Any]] = env_prompts_eto + env_prompts_synthetic
    logger.info("Total Environment prompts: %d", len(all_env_prompts))

    return (
        all_env_prompts,
        env_prompts_eto,
        env_prompts_synthetic,
        generate_synthetic_env_prompts,
        load_eto_prompts,
    )


# ---------------------------------------------------------------------------
# Cell 6: Deduplication, Quality, and Statistics
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 6 -- Deduplication, Contamination Re-check & Statistics

        Combine all prompts, remove duplicates by content hash, run a final
        contamination sweep, and compute distribution statistics.
        """
    )
    return ()


@app.cell
def _(
    Any,
    BANNED_ENV_KEYWORDS,
    TARGET_DB_MAX,
    TARGET_DB_MIN,
    TARGET_ENV_MAX,
    TARGET_ENV_MIN,
    all_db_prompts,
    all_env_prompts,
    content_hash,
    json,
    logger,
    np,
    pd,
    random,
):
    # ===================================================================
    # 6-a: Deduplicate by content hash
    # ===================================================================
    def deduplicate_prompts(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove exact duplicates based on prompt content hash."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for s in samples:
            h = content_hash(s["prompt"])
            if h not in seen:
                seen.add(h)
                unique.append(s)
        return unique

    logger.info("Deduplicating prompts...")
    db_deduped = deduplicate_prompts(all_db_prompts)
    env_deduped = deduplicate_prompts(all_env_prompts)

    logger.info(
        "After dedup -- DB: %d -> %d, Env: %d -> %d",
        len(all_db_prompts), len(db_deduped),
        len(all_env_prompts), len(env_deduped),
    )

    # ===================================================================
    # 6-b: Final contamination check (third layer)
    # ===================================================================
    def final_contamination_check(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Final sweep: reject any sample with banned keywords anywhere."""
        clean: list[dict[str, Any]] = []
        removed = 0
        for s in samples:
            blob = json.dumps(s, ensure_ascii=False).lower()
            contaminated = False
            for kw in BANNED_ENV_KEYWORDS:
                if kw in blob:
                    contaminated = True
                    break
            if contaminated:
                removed += 1
            else:
                clean.append(s)
        if removed > 0:
            logger.warning("Final contamination check removed %d samples", removed)
        return clean

    db_clean = final_contamination_check(db_deduped)
    env_clean = final_contamination_check(env_deduped)

    # ===================================================================
    # 6-c: Enforce target count ranges
    # ===================================================================
    def enforce_range(
        samples: list[dict[str, Any]],
        min_count: int,
        max_count: int,
        label: str,
    ) -> list[dict[str, Any]]:
        """Cap at max_count; warn if below min_count."""
        if len(samples) > max_count:
            selected = random.sample(samples, max_count)
            logger.info("%s: capped %d -> %d (max %d)", label, len(samples), max_count, max_count)
            return selected
        if len(samples) < min_count:
            logger.warning(
                "%s: only %d samples available, below target minimum %d",
                label, len(samples), min_count,
            )
        else:
            logger.info("%s: %d samples (within target %d-%d)", label, len(samples), min_count, max_count)
        return samples

    db_final = enforce_range(db_clean, TARGET_DB_MIN, TARGET_DB_MAX, "DB")
    env_final = enforce_range(env_clean, TARGET_ENV_MIN, TARGET_ENV_MAX, "Env")

    # Combine
    all_prompts_combined: list[dict[str, Any]] = db_final + env_final
    random.shuffle(all_prompts_combined)

    logger.info(
        "Final prompt dataset: DB=%d, Env=%d, Total=%d",
        len(db_final), len(env_final), len(all_prompts_combined),
    )

    # ===================================================================
    # 6-d: Compute statistics
    # ===================================================================
    def compute_statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute distribution statistics for the prompt dataset."""
        if not samples:
            return {"total": 0}

        task_types: dict[str, int] = {}
        sources: dict[str, int] = {}
        user_msg_lengths: list[int] = []
        has_gold_sql_count = 0
        has_db_path_count = 0

        for s in samples:
            tt = s["task_type"]
            src = s["source"]
            task_types[tt] = task_types.get(tt, 0) + 1
            sources[src] = sources.get(src, 0) + 1

            # User message is the last message in the prompt
            user_msg = s["prompt"][-1]["content"]
            user_msg_lengths.append(len(user_msg))

            if s.get("gold_sql"):
                has_gold_sql_count += 1
            if s.get("db_path"):
                has_db_path_count += 1

        return {
            "total": len(samples),
            "task_type_distribution": dict(sorted(task_types.items())),
            "source_distribution": dict(sorted(sources.items())),
            "user_message_length": {
                "mean": float(np.mean(user_msg_lengths)),
                "median": float(np.median(user_msg_lengths)),
                "min": int(np.min(user_msg_lengths)),
                "max": int(np.max(user_msg_lengths)),
                "std": float(np.std(user_msg_lengths)),
            },
            "has_gold_sql": has_gold_sql_count,
            "has_db_path": has_db_path_count,
            "db_to_env_ratio": f"{len(db_final)}:{len(env_final)}",
        }

    dataset_stats = compute_statistics(all_prompts_combined)

    # Pretty-print
    logger.info("=" * 60)
    logger.info("Prompt Dataset Statistics")
    logger.info("=" * 60)
    logger.info("Total prompts: %d", dataset_stats["total"])
    logger.info("Task type distribution: %s", dataset_stats.get("task_type_distribution", {}))
    logger.info("Source distribution: %s", dataset_stats.get("source_distribution", {}))
    if "user_message_length" in dataset_stats:
        ul = dataset_stats["user_message_length"]
        logger.info(
            "User msg length -- mean: %.0f, median: %.0f, min: %d, max: %d, std: %.0f",
            ul["mean"], ul["median"], ul["min"], ul["max"], ul["std"],
        )
    logger.info("Prompts with gold_sql: %d", dataset_stats.get("has_gold_sql", 0))
    logger.info("Prompts with db_path: %d", dataset_stats.get("has_db_path", 0))
    logger.info("DB:Env ratio = %s", dataset_stats.get("db_to_env_ratio", "N/A"))

    # Summary DataFrame
    summary_rows: list[dict[str, Any]] = []
    for src, count in sorted(dataset_stats.get("source_distribution", {}).items()):
        summary_rows.append({"source": src, "count": count})
    stats_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(columns=["source", "count"])

    return (
        all_prompts_combined,
        compute_statistics,
        dataset_stats,
        db_clean,
        db_final,
        deduplicate_prompts,
        enforce_range,
        env_clean,
        env_final,
        final_contamination_check,
        stats_df,
    )


# ---------------------------------------------------------------------------
# Cell 7: Export (JSONL + metadata)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 7 -- Export

        Save the final prompt dataset as JSONL with a companion metadata JSON file,
        then verify the output.
        """
    )
    return ()


@app.cell
def _(
    Any,
    METADATA_JSON,
    OUTPUT_JSONL,
    RANDOM_SEED,
    all_prompts_combined,
    dataset_stats,
    json,
    logger,
    stats_df,
):
    # ===================================================================
    # 7-a: Write JSONL
    # ===================================================================
    def write_jsonl(samples: list[dict[str, Any]], path: Any) -> int:
        """Write samples to JSONL file. Returns number of written lines."""
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
        return count

    n_written = write_jsonl(all_prompts_combined, OUTPUT_JSONL)
    logger.info("Wrote %d prompts to %s", n_written, OUTPUT_JSONL)

    # ===================================================================
    # 7-b: Write metadata
    # ===================================================================
    metadata: dict[str, Any] = {
        "dataset_name": "agentbench_rl_prompts_v1",
        "version": "1.0.0",
        "description": (
            "Prompt-only dataset for RL training (GRPO/DPO). Contains system+user message pairs "
            "that an SFT model will complete multiple times. Completions are ranked by reward "
            "(SQL execution accuracy for DB, env reward for Env) to create DPO training pairs."
        ),
        "creation_date": "2026-02-14",
        "random_seed": RANDOM_SEED,
        "format": "JSONL -- one JSON object per line",
        "schema": {
            "prompt": "list of {role, content} dicts (system + user only, no assistant)",
            "task_type": "db | env",
            "source": "spider | sparc | cosql | webshop | scienceworld | synthetic_react",
            "gold_sql": "gold-standard SQL string (DB prompts) or null (Env prompts)",
            "db_path": "relative path to SQLite database file or null",
        },
        "usage": (
            "1. Load prompts from JSONL.  "
            "2. For each prompt, generate N completions with the SFT model.  "
            "3. Score completions: DB prompts use SQL execution match against gold_sql; "
            "Env prompts use environment reward.  "
            "4. Select best/worst pairs for DPO training."
        ),
        "contamination_exclusions": [
            "ALFWorld (all variants)",
            "TextWorld (all variants)",
            "AgentBench evaluation data",
        ],
        "statistics": dataset_stats,
        "output_file": str(OUTPUT_JSONL),
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Wrote metadata to %s", METADATA_JSON)

    # ===================================================================
    # 7-c: Verification
    # ===================================================================
    VALID_TASK_TYPES = {"db", "env"}
    VALID_SOURCES = {"spider", "sparc", "cosql", "webshop", "scienceworld", "synthetic_react"}

    verified_count = 0
    errors: list[str] = []

    with open(OUTPUT_JSONL, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            obj = json.loads(line.strip())

            # Required fields
            for field in ("prompt", "task_type", "source", "gold_sql", "db_path"):
                if field not in obj:
                    errors.append(f"Line {line_num}: missing field '{field}'")

            # Prompt structure
            prompt = obj.get("prompt", [])
            if len(prompt) < 2:
                errors.append(f"Line {line_num}: prompt has fewer than 2 messages")
            elif prompt[0].get("role") != "system":
                errors.append(f"Line {line_num}: first message is not system role")
            elif prompt[-1].get("role") != "user":
                errors.append(f"Line {line_num}: last message is not user role")

            # Valid enums
            if obj.get("task_type") not in VALID_TASK_TYPES:
                errors.append(f"Line {line_num}: invalid task_type '{obj.get('task_type')}'")
            if obj.get("source") not in VALID_SOURCES:
                errors.append(f"Line {line_num}: invalid source '{obj.get('source')}'")

            # DB prompts should have gold_sql
            if obj.get("task_type") == "db" and not obj.get("gold_sql"):
                errors.append(f"Line {line_num}: DB prompt missing gold_sql")

            verified_count += 1

    if errors:
        for err in errors[:20]:  # Show first 20 errors
            logger.error("Verification error: %s", err)
        logger.error("Total verification errors: %d", len(errors))
    else:
        logger.info("Verification passed: %d / %d prompts OK (0 errors)", verified_count, n_written)

    logger.info("=" * 60)
    logger.info("DONE. Prompt dataset ready at: %s", OUTPUT_JSONL)
    logger.info("=" * 60)

    # Display summary
    print(f"\nPrompt dataset written: {OUTPUT_JSONL}")
    print(f"Metadata written: {METADATA_JSON}")
    print(f"Total prompts: {n_written}")
    print(f"Verified: {verified_count} (errors: {len(errors)})")
    print("\nSource distribution:")
    print(stats_df.to_string(index=False))

    return (
        metadata,
        n_written,
        verified_count,
        write_jsonl,
    )


if __name__ == "__main__":
    app.run()
