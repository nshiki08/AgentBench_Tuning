# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "datasets>=3.0.0",
#     "pandas>=2.2.0",
#     "numpy>=1.26.0",
#     "tqdm",
#     "tiktoken",
# ]
# ///
"""
Build Training Set V1 for AgentBench DB Bench & ALFWorld score improvement.

Constructs a mixed training dataset (~6,000 samples) with:
- DB multi-turn dialogue data (SParC, CoSQL, Spider, synthetic SQL)
- Environment goal-achievement multi-turn data (WebShop, ScienceWorld, synthetic ReAct)
- General chat data (OpenAssistant, etc.)

CRITICAL: No AgentBench evaluation data (ALFWorld, TextWorld) is included.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# Cell 1: Configuration & Constants
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Build Training Set V1

        AgentBench DB Bench / ALFWorld スコア向上のための学習データセット (第1版) を構築する。

        **混合比率**: agent (DB + env) : general = 1 : 2 (約 2,000 : 4,000 = 6,000 件)

        **禁止**: AgentBench 本体のデータ (ALFWorld / TextWorld 環境含む) を学習データに含めない。
        """
    )
    return (mo,)


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
    OUTPUT_JSONL: Path = OUTPUT_DIR / "training_set_v1.jsonl"
    METADATA_JSON: Path = OUTPUT_DIR / "training_set_v1_metadata.json"

    # --------------- target counts ---------------
    TARGET_AGENT_TOTAL: int = 2_000  # DB + env combined
    TARGET_DB_COUNT: int = 1_000
    TARGET_ENV_COUNT: int = 1_000
    TARGET_GENERAL_COUNT: int = 4_000
    TARGET_TOTAL: int = TARGET_AGENT_TOTAL + TARGET_GENERAL_COUNT  # ~6,000

    # --------------- sub-source budgets (approximate) ---------------
    DB_BUDGET: dict[str, int] = {
        "sparc": 300,
        "cosql": 300,
        "spider": 200,
        "synthetic_sql": 200,
    }
    ENV_BUDGET: dict[str, int] = {
        "webshop": 350,
        "scienceworld": 350,
        "synthetic_react": 300,
    }
    GENERAL_BUDGET: dict[str, int] = {
        "oasst2": 2_000,
        "chatbot_arena": 2_000,
    }

    # --------------- quality thresholds ---------------
    MIN_MESSAGE_COUNT: int = 2  # at least system + 1 turn
    MAX_MESSAGE_COUNT: int = 60
    MIN_CONTENT_LENGTH: int = 5  # characters per message content
    MAX_TOTAL_CHARS: int = 50_000  # per conversation

    # --------------- banned keywords for contamination ---------------
    BANNED_ENV_KEYWORDS: list[str] = [
        "alfworld",
        "alf_world",
        "textworld",
        "text_world",
        "tw_",
        "tw-",
    ]

    logger.info("Configuration loaded. Output -> %s", OUTPUT_JSONL)

    return (
        BANNED_ENV_KEYWORDS,
        DB_BUDGET,
        ENV_BUDGET,
        GENERAL_BUDGET,
        MAX_MESSAGE_COUNT,
        MAX_TOTAL_CHARS,
        MIN_CONTENT_LENGTH,
        MIN_MESSAGE_COUNT,
        METADATA_JSON,
        OUTPUT_DIR,
        OUTPUT_JSONL,
        RANDOM_SEED,
        TARGET_AGENT_TOTAL,
        TARGET_DB_COUNT,
        TARGET_ENV_COUNT,
        TARGET_GENERAL_COUNT,
        TARGET_TOTAL,
        Any,
        Path,
        hashlib,
        json,
        logger,
        np,
        pd,
        random,
    )


# ---------------------------------------------------------------------------
# Cell 2: DB Multi-turn Dialogue Data
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 2 -- DB Multi-turn Dialogue Data

        Sources: SParC, CoSQL, Spider (converted to multi-turn), gretelai/synthetic_text_to_sql
        """
    )
    return ()


@app.cell
def _(
    Any,
    BANNED_ENV_KEYWORDS,
    DB_BUDGET,
    MAX_MESSAGE_COUNT,
    MAX_TOTAL_CHARS,
    MIN_CONTENT_LENGTH,
    MIN_MESSAGE_COUNT,
    hashlib,
    json,
    logger,
    random,
):
    from datasets import load_dataset

    # ===================================================================
    # Helper utilities
    # ===================================================================
    def _make_chatml(
        messages: list[dict[str, str]],
        task_type: str,
        source: str,
    ) -> dict[str, Any] | None:
        """Build a single ChatML sample with quality checks. Returns None if invalid."""
        if not messages or len(messages) < MIN_MESSAGE_COUNT:
            return None
        if len(messages) > MAX_MESSAGE_COUNT:
            return None
        total_chars = sum(len(m.get("content", "")) for m in messages)
        if total_chars > MAX_TOTAL_CHARS:
            return None
        for m in messages:
            content = m.get("content", "")
            if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
                return None
        # contamination check
        full_text = json.dumps(messages).lower()
        for kw in BANNED_ENV_KEYWORDS:
            if kw in full_text:
                logger.warning("Contamination keyword '%s' found -- skipping sample (source=%s)", kw, source)
                return None
        return {"messages": messages, "task_type": task_type, "source": source}

    def _content_hash(sample: dict[str, Any]) -> str:
        """Deterministic hash for deduplication."""
        blob = json.dumps(sample["messages"], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    DB_SYSTEM_PROMPT: str = (
        "You are an expert database assistant. The user will ask questions about a database. "
        "Write SQL queries to answer them. Explain your reasoning before writing SQL.\n\n"
        "Respond in the following format:\n"
        "Thought: <your reasoning>\n"
        "Action: execute_sql\n"
        "Action Input: ```sql\n<your SQL query>\n```"
    )

    # ===================================================================
    # 2-a: SParC loader
    # ===================================================================
    def load_sparc(budget: int) -> list[dict[str, Any]]:
        """Load SParC multi-turn text-to-SQL data."""
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
            # SParC has interaction_id, database_id, turns (utterances + queries)
            db_id = row.get("database_id", row.get("db_id", "unknown"))
            utterances = row.get("utterances", row.get("question", []))
            queries = row.get("query", row.get("sql", []))
            if not utterances or not queries:
                continue
            # Normalize to lists
            if isinstance(utterances, str):
                utterances = [utterances]
            if isinstance(queries, str):
                queries = [queries]

            messages: list[dict[str, str]] = [
                {"role": "system", "content": DB_SYSTEM_PROMPT + f"\nDatabase: {db_id}"},
            ]
            for utt, sql in zip(utterances, queries):
                messages.append({"role": "user", "content": utt})
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"Thought: I need to write a SQL query for the user's question about the {db_id} database.\n"
                        f"Action: execute_sql\n"
                        f"Action Input: ```sql\n{sql}\n```"
                    ),
                })
            sample = _make_chatml(messages, task_type="db", source="sparc")
            if sample is not None:
                samples.append(sample)

        logger.info("SParC: collected %d / %d target samples", len(samples), budget)
        return samples

    # ===================================================================
    # 2-b: CoSQL loader
    # ===================================================================
    def load_cosql(budget: int) -> list[dict[str, Any]]:
        """Load CoSQL multi-turn text-to-SQL data."""
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

            messages: list[dict[str, str]] = [
                {"role": "system", "content": DB_SYSTEM_PROMPT + f"\nDatabase: {db_id}"},
            ]
            for utt, sql in zip(utterances, queries):
                messages.append({"role": "user", "content": utt})
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"Thought: Let me analyze this question about the {db_id} database.\n"
                        f"Action: execute_sql\n"
                        f"Action Input: ```sql\n{sql}\n```"
                    ),
                })
            sample = _make_chatml(messages, task_type="db", source="cosql")
            if sample is not None:
                samples.append(sample)

        logger.info("CoSQL: collected %d / %d target samples", len(samples), budget)
        return samples

    # ===================================================================
    # 2-c: Spider -> multi-turn conversion
    # ===================================================================
    def load_spider_multiturn(budget: int) -> list[dict[str, Any]]:
        """Load Spider and convert to multi-turn by grouping queries per database."""
        try:
            ds = load_dataset("xlangai/spider", split="train", trust_remote_code=True)
            logger.info("Loaded Spider (%d rows)", len(ds))
        except Exception as e:
            logger.error("Could not load Spider: %s", e)
            return []

        # Group by db_id
        db_groups: dict[str, list[dict[str, str]]] = {}
        for row in ds:
            db_id = row.get("db_id", "unknown")
            question = row.get("question", "")
            query = row.get("query", "")
            if question and query:
                db_groups.setdefault(db_id, []).append({"question": question, "query": query})

        samples: list[dict[str, Any]] = []
        db_ids = list(db_groups.keys())
        random.shuffle(db_ids)

        for db_id in db_ids:
            if len(samples) >= budget:
                break
            items = db_groups[db_id]
            if len(items) < 2:
                continue
            random.shuffle(items)
            # Take 2-5 queries to form a multi-turn conversation
            n_turns = min(len(items), random.randint(2, 5))
            selected = items[:n_turns]

            messages: list[dict[str, str]] = [
                {"role": "system", "content": DB_SYSTEM_PROMPT + f"\nDatabase: {db_id}"},
            ]
            for it in selected:
                messages.append({"role": "user", "content": it["question"]})
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"Thought: The user asks about the {db_id} database. "
                        f"Let me construct an appropriate SQL query.\n"
                        f"Action: execute_sql\n"
                        f"Action Input: ```sql\n{it['query']}\n```"
                    ),
                })
            sample = _make_chatml(messages, task_type="db", source="spider")
            if sample is not None:
                samples.append(sample)

        logger.info("Spider multi-turn: collected %d / %d target samples", len(samples), budget)
        return samples

    # ===================================================================
    # 2-d: Synthetic Text-to-SQL
    # ===================================================================
    def load_synthetic_sql(budget: int) -> list[dict[str, Any]]:
        """Load gretelai/synthetic_text_to_sql with quality filtering."""
        try:
            ds = load_dataset("gretelai/synthetic_text_to_sql", split="train", trust_remote_code=True)
            logger.info("Loaded synthetic_text_to_sql (%d rows)", len(ds))
        except Exception as e:
            logger.error("Could not load synthetic_text_to_sql: %s", e)
            return []

        # Quality filter: keep rows with reasonable SQL length and non-trivial questions
        filtered_indices: list[int] = []
        for i, row in enumerate(ds):
            sql = row.get("sql", "") or ""
            question = row.get("sql_prompt", "") or row.get("question", "") or ""
            context = row.get("sql_context", "") or row.get("context", "") or ""
            if len(sql) < 10 or len(question) < 10:
                continue
            # Skip trivially short SQL
            if sql.strip().upper().startswith("SELECT") and len(sql) > 15:
                filtered_indices.append(i)

        random.shuffle(filtered_indices)
        logger.info("synthetic_text_to_sql: %d passed quality filter", len(filtered_indices))

        # Group consecutive pairs into multi-turn conversations
        samples: list[dict[str, Any]] = []
        i = 0
        while i < len(filtered_indices) - 1 and len(samples) < budget:
            n_turns = min(random.randint(2, 4), len(filtered_indices) - i)
            messages: list[dict[str, str]] = [
                {"role": "system", "content": DB_SYSTEM_PROMPT + "\nDatabase: synthetic_db"},
            ]
            for j in range(n_turns):
                row = ds[filtered_indices[i + j]]
                question = row.get("sql_prompt", "") or row.get("question", "")
                sql = row.get("sql", "")
                context = row.get("sql_context", "") or row.get("context", "") or ""
                user_msg = question
                if context:
                    user_msg = f"{question}\n\nContext:\n{context}"
                messages.append({"role": "user", "content": user_msg})
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"Thought: Let me analyze this query request.\n"
                        f"Action: execute_sql\n"
                        f"Action Input: ```sql\n{sql}\n```"
                    ),
                })
            sample = _make_chatml(messages, task_type="db", source="synthetic_sql")
            if sample is not None:
                samples.append(sample)
            i += n_turns

        logger.info("synthetic_text_to_sql: collected %d / %d target samples", len(samples), budget)
        return samples

    # ===================================================================
    # Run all DB loaders
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Starting DB data collection")
    logger.info("=" * 60)

    db_samples_sparc: list[dict[str, Any]] = load_sparc(DB_BUDGET["sparc"])
    db_samples_cosql: list[dict[str, Any]] = load_cosql(DB_BUDGET["cosql"])
    db_samples_spider: list[dict[str, Any]] = load_spider_multiturn(DB_BUDGET["spider"])
    db_samples_synthetic: list[dict[str, Any]] = load_synthetic_sql(DB_BUDGET["synthetic_sql"])

    all_db_samples: list[dict[str, Any]] = (
        db_samples_sparc + db_samples_cosql + db_samples_spider + db_samples_synthetic
    )
    logger.info("Total DB samples: %d", len(all_db_samples))

    return (
        DB_SYSTEM_PROMPT,
        _content_hash,
        _make_chatml,
        all_db_samples,
        db_samples_cosql,
        db_samples_sparc,
        db_samples_spider,
        db_samples_synthetic,
        load_cosql,
        load_dataset,
        load_sparc,
        load_spider_multiturn,
        load_synthetic_sql,
    )


# ---------------------------------------------------------------------------
# Cell 3: Environment Goal-Achievement Multi-turn Data
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 3 -- Environment Goal-Achievement Multi-turn Data

        Sources:
        - **agent-eto/eto-sft-trajectory** (WebShop + ScienceWorld only -- ALFWorld excluded!)
        - **Synthetic ReAct trajectories** (template-based household / navigation / tool-use)
        """
    )
    return ()


@app.cell
def _(
    Any,
    BANNED_ENV_KEYWORDS,
    ENV_BUDGET,
    _make_chatml,
    hashlib,
    json,
    load_dataset,
    logger,
    random,
):
    # ===================================================================
    # System prompts for environment tasks
    # ===================================================================
    WEBSHOP_SYSTEM_PROMPT: str = (
        "You are a shopping assistant agent interacting with a web shopping environment. "
        "Your goal is to find and purchase the product that best matches the user's request.\n\n"
        "Respond in ReAct format:\n"
        "Thought: <reasoning about current state and next action>\n"
        "Action: <action_name>\n"
        "Action Input: <action argument>"
    )

    SCIWORLD_SYSTEM_PROMPT: str = (
        "You are a scientific experiment agent operating in a simulated environment. "
        "Your goal is to complete the given scientific task by interacting with objects in the environment.\n\n"
        "Respond in ReAct format:\n"
        "Thought: <reasoning about current state and next action>\n"
        "Action: <action_name>\n"
        "Action Input: <action argument>"
    )

    REACT_SYSTEM_PROMPT: str = (
        "You are an autonomous agent operating in an interactive environment. "
        "Your goal is to complete the given task by reasoning step by step and taking actions.\n\n"
        "Respond in ReAct format:\n"
        "Thought: <reasoning about current state and next action>\n"
        "Action: <action_name>\n"
        "Action Input: <action argument>"
    )

    # ===================================================================
    # 3-a: ETO trajectory loader (WebShop + ScienceWorld only)
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

    def _normalize_eto_messages(row: dict[str, Any], source_type: str) -> list[dict[str, str]] | None:
        """Extract and normalize messages from an ETO trajectory row."""
        raw = row.get("messages", row.get("conversations", []))
        if not raw:
            return None

        # Handle list of dicts with role/content
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], dict):
                messages: list[dict[str, str]] = []
                for m in raw:
                    role = m.get("role", m.get("from", "")).lower()
                    content = m.get("content", m.get("value", ""))
                    if role in ("system", "user", "assistant", "human", "gpt", "observation"):
                        # Normalize roles
                        if role == "human":
                            role = "user"
                        elif role == "gpt":
                            role = "assistant"
                        elif role == "observation":
                            role = "user"
                            content = f"[Observation] {content}"
                        messages.append({"role": role, "content": str(content)})
                if not messages:
                    return None

                # Ensure system prompt exists
                sys_prompt = WEBSHOP_SYSTEM_PROMPT if source_type == "webshop" else SCIWORLD_SYSTEM_PROMPT
                if messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": sys_prompt})
                else:
                    # Augment existing system prompt
                    messages[0]["content"] = sys_prompt + "\n\n" + messages[0]["content"]

                return messages
        return None

    def load_eto_trajectories(webshop_budget: int, sciworld_budget: int) -> list[dict[str, Any]]:
        """Load agent-eto/eto-sft-trajectory, keeping only WebShop & ScienceWorld."""
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

            messages = _normalize_eto_messages(row, source_type)
            if messages is None:
                continue

            sample = _make_chatml(messages, task_type="env", source=source_type)
            if sample is None:
                continue

            if source_type == "webshop" and len(webshop_samples) < webshop_budget:
                webshop_samples.append(sample)
            elif source_type == "scienceworld" and len(sciworld_samples) < sciworld_budget:
                sciworld_samples.append(sample)

        logger.info(
            "ETO: WebShop=%d/%d, ScienceWorld=%d/%d",
            len(webshop_samples),
            webshop_budget,
            len(sciworld_samples),
            sciworld_budget,
        )
        return webshop_samples + sciworld_samples

    # ===================================================================
    # 3-b: Synthetic ReAct trajectory generation (template-based)
    # ===================================================================

    # --- Household task templates ---
    _HOUSEHOLD_TASKS: list[dict[str, Any]] = [
        {
            "goal": "Clean the kitchen counter and put away the dishes.",
            "steps": [
                ("look around", "You see a kitchen counter with dirty dishes, a sponge, and dish soap."),
                ("pick up sponge", "You pick up the sponge."),
                ("use sponge on counter", "You scrub the counter with the sponge. The counter is now clean."),
                ("pick up dishes", "You pick up the dirty dishes."),
                ("go to sink", "You walk to the sink."),
                ("wash dishes", "You wash the dishes with soap and water. The dishes are now clean."),
                ("go to cabinet", "You walk to the cabinet."),
                ("put dishes in cabinet", "You place the clean dishes in the cabinet. Task complete."),
            ],
        },
        {
            "goal": "Find the book on the shelf and place it on the desk.",
            "steps": [
                ("look around", "You see a room with a bookshelf, a desk, and a chair."),
                ("go to bookshelf", "You walk to the bookshelf. You see several books."),
                ("examine bookshelf", "You see: 'Python Programming', 'Data Science Handbook', 'Machine Learning'."),
                ("take Python Programming", "You pick up 'Python Programming' from the shelf."),
                ("go to desk", "You walk to the desk."),
                ("put book on desk", "You place 'Python Programming' on the desk. Task complete."),
            ],
        },
        {
            "goal": "Heat the soup in the microwave and serve it in a bowl.",
            "steps": [
                ("look around", "You see a kitchen with a microwave, soup on the stove, and bowls in the cabinet."),
                ("go to stove", "You walk to the stove. There is a pot of cold soup."),
                ("take pot", "You pick up the pot of soup."),
                ("go to microwave", "You walk to the microwave."),
                ("put pot in microwave", "You place the pot in the microwave."),
                ("turn on microwave", "The microwave heats the soup for 2 minutes. The soup is now hot."),
                ("take pot from microwave", "You take the hot pot out of the microwave."),
                ("go to cabinet", "You walk to the cabinet."),
                ("take bowl", "You take a bowl from the cabinet."),
                ("pour soup into bowl", "You pour the hot soup into the bowl. Task complete."),
            ],
        },
        {
            "goal": "Water all the plants in the living room.",
            "steps": [
                (
                    "look around",
                    "You see a living room with three potted plants, a watering can near the window, and furniture.",
                ),
                ("go to window", "You walk to the window where the watering can is."),
                ("take watering can", "You pick up the watering can. It is full of water."),
                ("go to plant 1", "You walk to the first plant. It looks dry."),
                ("water plant 1", "You water the first plant. The soil absorbs the water."),
                ("go to plant 2", "You walk to the second plant near the couch."),
                ("water plant 2", "You water the second plant."),
                ("go to plant 3", "You walk to the third plant on the bookshelf."),
                ("water plant 3", "You water the third plant. All plants have been watered. Task complete."),
            ],
        },
        {
            "goal": "Set the dining table for dinner for two people.",
            "steps": [
                ("look around", "You see a dining room with a table, a cabinet with tableware, and a utensil drawer."),
                ("go to cabinet", "You walk to the cabinet. You see plates, glasses, and napkins."),
                ("take 2 plates", "You take two dinner plates from the cabinet."),
                ("go to table", "You walk to the dining table."),
                ("place plates on table", "You set two plates across from each other on the table."),
                ("go to drawer", "You walk to the utensil drawer."),
                ("take utensils", "You take two sets of fork, knife, and spoon."),
                ("go to table", "You walk back to the dining table."),
                ("place utensils", "You arrange the utensils beside each plate."),
                ("go to cabinet", "You walk to the cabinet again."),
                ("take 2 glasses", "You take two water glasses."),
                ("go to table", "You walk to the table."),
                ("place glasses", "You place a glass above each plate. The table is set. Task complete."),
            ],
        },
    ]

    # --- Navigation task templates ---
    _NAVIGATION_TASKS: list[dict[str, Any]] = [
        {
            "goal": "Navigate from the entrance to the conference room on the 3rd floor.",
            "steps": [
                ("look around", "You are at the building entrance. You see a reception desk, elevators, and stairs."),
                ("go to elevator", "You walk to the elevator and press the call button."),
                ("press button 3", "The elevator arrives. You enter and press the button for floor 3."),
                ("exit elevator", "The elevator opens on the 3rd floor. You see a hallway with doors."),
                ("look at signs", "Signs: Room 301 (left), Room 302 (right), Conference Room (straight ahead)."),
                ("go straight", "You walk straight ahead to the conference room door."),
                ("open door", "You open the door and enter the conference room. You have arrived. Task complete."),
            ],
        },
        {
            "goal": "Find the nearest pharmacy from the park.",
            "steps": [
                (
                    "look around",
                    "You are at the city park. North is Main Street, east is Oak Avenue, south is the lake.",
                ),
                ("go north to Main Street", "You walk north and arrive at Main Street. You see shops and a bus stop."),
                ("look at shops", "You see: Coffee Shop, Bookstore, and further east a green cross sign (pharmacy)."),
                ("go east", "You walk east along Main Street toward the green cross sign."),
                ("enter pharmacy", "You enter the pharmacy. You have found the nearest pharmacy. Task complete."),
            ],
        },
        {
            "goal": "Deliver the package to apartment 4B in the residential building.",
            "steps": [
                (
                    "look around",
                    "You are outside a residential building holding a package. You see the entrance with an intercom.",
                ),
                ("use intercom", "You press the button for apartment 4B. A voice answers: 'Come in, door is open.'"),
                ("enter building", "You enter the building. You see a lobby with mailboxes, stairs, and an elevator."),
                ("go to elevator", "You walk to the elevator."),
                ("press button 4", "You take the elevator to the 4th floor."),
                ("exit elevator", "You exit on the 4th floor. You see doors: 4A (left), 4B (right), 4C (far end)."),
                ("go to door 4B", "You walk to door 4B."),
                ("knock on door", "Someone opens the door."),
                ("deliver package", "You hand over the package. The resident thanks you. Task complete."),
            ],
        },
    ]

    # --- Tool-use task templates ---
    _TOOL_USE_TASKS: list[dict[str, Any]] = [
        {
            "goal": "Search for the weather forecast and send it to the user via email.",
            "steps": [
                (
                    "search_web",
                    "weather forecast today",
                    "Results: Today's weather: Sunny, high 72F, low 55F, 10% chance of rain.",
                ),
                (
                    "compose_email",
                    "to: user@example.com, subject: Today's Weather",
                    "Email draft created with weather information.",
                ),
                ("send_email", "", "Email sent successfully to user@example.com. Task complete."),
            ],
        },
        {
            "goal": "Calculate the total cost of items in the shopping list and save to a file.",
            "steps": [
                (
                    "read_file",
                    "shopping_list.txt",
                    "Contents: Apples $3.50, Bread $2.00, Milk $4.50, Eggs $5.00, Cheese $6.00",
                ),
                ("calculator", "3.50 + 2.00 + 4.50 + 5.00 + 6.00", "Result: 21.00"),
                ("write_file", "total_cost.txt with 'Total: $21.00'", "File saved successfully. Task complete."),
            ],
        },
        {
            "goal": "Look up the company's stock price and create a brief summary report.",
            "steps": [
                (
                    "search_web",
                    "ACME Corp stock price",
                    "Results: ACME Corp (ACME) price: $142.50, change: +2.3%, volume: 1.2M",
                ),
                (
                    "search_web",
                    "ACME Corp recent news",
                    "Results: ACME Corp announced Q3 earnings beat, revenue up 15% YoY.",
                ),
                (
                    "write_file",
                    "acme_report.txt with summary",
                    "Report saved: 'ACME Corp at $142.50 (+2.3%), Q3 beat.' Task complete.",
                ),
            ],
        },
        {
            "goal": "Translate the document from English to Japanese and save the result.",
            "steps": [
                (
                    "read_file",
                    "document.txt",
                    "Contents: 'The quick brown fox jumps over the lazy dog. A test document.'",
                ),
                ("translate", "English to Japanese", "Translation complete."),
                ("write_file", "document_ja.txt", "File saved with Japanese translation. Task complete."),
            ],
        },
    ]

    def _generate_react_household(task: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a ReAct trajectory from a household task template."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task['goal']}\n\n"
                    "You are in an interactive household environment. Complete the task."
                ),
            },
        ]
        for i, (action, observation) in enumerate(task["steps"]):
            thought_prefix = random.choice([
                "I need to", "Let me", "I should", "The next step is to", "Now I'll",
            ])
            thought = f"{thought_prefix} {action.lower()} to make progress on the task."
            if i == len(task["steps"]) - 1:
                thought = f"This should be the final step. {thought_prefix} {action.lower()} to complete the task."
            messages.append({
                "role": "assistant",
                "content": f"Thought: {thought}\nAction: interact\nAction Input: {action}",
            })
            messages.append({"role": "user", "content": f"[Observation] {observation}"})
        # Final assistant response
        messages.append({
            "role": "assistant",
            "content": "Thought: The task has been completed successfully.\nAction: finish\nAction Input: done",
        })
        return _make_chatml(messages, task_type="env", source="synthetic_react")

    def _generate_react_navigation(task: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a ReAct trajectory from a navigation task template."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task['goal']}\n\n"
                    "You are in an interactive environment. Navigate to the destination."
                ),
            },
        ]
        for i, (action, observation) in enumerate(task["steps"]):
            thought_prefix = random.choice([
                "I should", "Let me", "I need to", "Next, I'll", "To proceed, I'll",
            ])
            thought = f"{thought_prefix} {action.lower()}."
            messages.append({
                "role": "assistant",
                "content": f"Thought: {thought}\nAction: navigate\nAction Input: {action}",
            })
            messages.append({"role": "user", "content": f"[Observation] {observation}"})
        messages.append({
            "role": "assistant",
            "content": "Thought: I have reached the destination.\nAction: finish\nAction Input: done",
        })
        return _make_chatml(messages, task_type="env", source="synthetic_react")

    def _generate_react_tool_use(task: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a ReAct trajectory from a tool-use task template."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Task: {task['goal']}\n\nYou have access to various tools. Complete the task.",
            },
        ]
        for i, step in enumerate(task["steps"]):
            tool_name, tool_input, observation = step[0], step[1], step[2]
            thought_prefix = random.choice([
                "I'll use the", "Let me use", "I need to use the", "Next I should use",
            ])
            thought = f"{thought_prefix} {tool_name} tool."
            messages.append({
                "role": "assistant",
                "content": f"Thought: {thought}\nAction: {tool_name}\nAction Input: {tool_input}",
            })
            messages.append({"role": "user", "content": f"[Observation] {observation}"})
        messages.append({
            "role": "assistant",
            "content": "Thought: All steps are done. Task finished.\nAction: finish\nAction Input: done",
        })
        return _make_chatml(messages, task_type="env", source="synthetic_react")

    def generate_synthetic_react(budget: int) -> list[dict[str, Any]]:
        """Generate synthetic ReAct trajectories from templates with variation."""
        all_generators: list[tuple[Any, list[dict[str, Any]]]] = [
            (_generate_react_household, _HOUSEHOLD_TASKS),
            (_generate_react_navigation, _NAVIGATION_TASKS),
            (_generate_react_tool_use, _TOOL_USE_TASKS),
        ]

        samples: list[dict[str, Any]] = []
        # Repeat templates with variation to reach budget
        attempts = 0
        max_attempts = budget * 5
        while len(samples) < budget and attempts < max_attempts:
            gen_fn, tasks = random.choice(all_generators)
            task = random.choice(tasks)
            sample = gen_fn(task)
            if sample is not None:
                samples.append(sample)
            attempts += 1

        # Deduplicate (templates may produce identical results)
        seen_hashes: set[str] = set()
        unique: list[dict[str, Any]] = []
        for s in samples:
            h = json.dumps(s["messages"], ensure_ascii=False, sort_keys=True)
            hh = hashlib.sha256(h.encode()).hexdigest()
            if hh not in seen_hashes:
                seen_hashes.add(hh)
                unique.append(s)

        logger.info("Synthetic ReAct: generated %d unique / %d budget", len(unique), budget)
        return unique[:budget]

    # ===================================================================
    # Run all environment loaders
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Starting Environment data collection")
    logger.info("=" * 60)

    env_samples_eto: list[dict[str, Any]] = load_eto_trajectories(
        webshop_budget=ENV_BUDGET["webshop"],
        sciworld_budget=ENV_BUDGET["scienceworld"],
    )
    env_samples_synthetic: list[dict[str, Any]] = generate_synthetic_react(ENV_BUDGET["synthetic_react"])

    all_env_samples: list[dict[str, Any]] = env_samples_eto + env_samples_synthetic
    logger.info("Total Environment samples: %d", len(all_env_samples))

    return (
        REACT_SYSTEM_PROMPT,
        SCIWORLD_SYSTEM_PROMPT,
        WEBSHOP_SYSTEM_PROMPT,
        all_env_samples,
        env_samples_eto,
        env_samples_synthetic,
        generate_synthetic_react,
        load_eto_trajectories,
    )


# ---------------------------------------------------------------------------
# Cell 4: General Chat Data
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 4 -- General Chat Data

        Sources: OpenAssistant/oasst2, lmsys/chatbot_arena_conversations
        """
    )
    return ()


@app.cell
def _(
    Any,
    GENERAL_BUDGET,
    _make_chatml,
    load_dataset,
    logger,
    random,
):
    # ===================================================================
    # 4-a: OpenAssistant oasst2
    # ===================================================================
    def load_oasst2(budget: int) -> list[dict[str, Any]]:
        """Load and convert OpenAssistant oasst2 conversations."""
        try:
            ds = load_dataset("OpenAssistant/oasst2", split="train", trust_remote_code=True)
            logger.info("Loaded oasst2 (%d rows)", len(ds))
        except Exception as e:
            logger.error("Could not load oasst2: %s", e)
            return []

        # oasst2 is tree-structured: group by tree_id, then reconstruct conversations
        # Each row has: message_id, parent_id, text, role, tree_id, etc.
        from collections import defaultdict

        trees: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in ds:
            tree_id = row.get("conversation_id", row.get("tree_id", row.get("parent_id", "")))
            trees[tree_id].append(row)

        # If tree structure is not available, treat each row as a single exchange
        # Try simpler approach: group by message_tree_id and build chains
        message_map: dict[str, dict[str, Any]] = {}
        children_map: dict[str, list[str]] = defaultdict(list)

        for row in ds:
            mid = row.get("message_id", "")
            pid = row.get("parent_id", None)
            if mid:
                message_map[mid] = row
                if pid:
                    children_map[pid].append(mid)

        # Find root messages (no parent)
        roots: list[str] = [
            mid for mid, row in message_map.items()
            if row.get("parent_id") is None or row.get("parent_id") == ""
        ]
        random.shuffle(roots)

        def _build_chain(root_id: str) -> list[dict[str, str]]:
            """Build a conversation chain from root, picking one child at each level."""
            chain: list[dict[str, str]] = []
            current_id = root_id
            while current_id and current_id in message_map:
                msg = message_map[current_id]
                role_raw = str(msg.get("role", "")).lower()
                if role_raw in ("prompter", "human", "user"):
                    role = "user"
                elif role_raw in ("assistant", "gpt"):
                    role = "assistant"
                else:
                    role = "user" if len(chain) % 2 == 0 else "assistant"
                text = msg.get("text", msg.get("content", ""))
                chain.append({"role": role, "content": str(text)})
                # Pick the highest-ranked child
                kids = children_map.get(current_id, [])
                if not kids:
                    break
                # Sort by rank if available, otherwise pick first
                kids_with_rank = [(kid, message_map.get(kid, {}).get("rank", 999)) for kid in kids]
                kids_with_rank.sort(key=lambda x: x[1])
                current_id = kids_with_rank[0][0]
            return chain

        samples: list[dict[str, Any]] = []
        for root_id in roots:
            if len(samples) >= budget:
                break
            chain = _build_chain(root_id)
            if len(chain) < 2:
                continue
            # Add system prompt
            messages: list[dict[str, str]] = [
                {"role": "system", "content": "You are a helpful, harmless, and honest AI assistant."},
            ] + chain
            sample = _make_chatml(messages, task_type="general", source="general_chat")
            if sample is not None:
                samples.append(sample)

        logger.info("oasst2: collected %d / %d target samples", len(samples), budget)
        return samples

    # ===================================================================
    # 4-b: Chatbot Arena conversations
    # ===================================================================
    def load_chatbot_arena(budget: int) -> list[dict[str, Any]]:
        """Load lmsys/chatbot_arena_conversations."""
        try:
            ds = load_dataset("lmsys/chatbot_arena_conversations", split="train", trust_remote_code=True)
            logger.info("Loaded chatbot_arena_conversations (%d rows)", len(ds))
        except Exception as e:
            logger.error("Could not load chatbot_arena_conversations: %s", e)
            # Fallback: try lmsys/lmsys-chat-1m
            try:
                ds = load_dataset("lmsys/lmsys-chat-1m", split="train", trust_remote_code=True)
                logger.info("Fallback: loaded lmsys-chat-1m (%d rows)", len(ds))
            except Exception as e2:
                logger.error("Fallback also failed: %s", e2)
                return []

        samples: list[dict[str, Any]] = []
        indices = list(range(len(ds)))
        random.shuffle(indices)

        for idx in indices:
            if len(samples) >= budget:
                break
            row = ds[idx]

            # chatbot_arena has 'conversation_a' and 'conversation_b' - use the winner
            winner = row.get("winner", "model_a")
            conv_key = "conversation_a" if "model_a" in str(winner) else "conversation_b"
            conv = row.get(conv_key, row.get("conversation", []))

            if not conv:
                continue

            messages: list[dict[str, str]] = [
                {"role": "system", "content": "You are a helpful, harmless, and honest AI assistant."},
            ]
            for turn in conv:
                if isinstance(turn, dict):
                    role = turn.get("role", turn.get("from", "")).lower()
                    content = turn.get("content", turn.get("value", ""))
                    if role in ("user", "human", "prompter"):
                        messages.append({"role": "user", "content": str(content)})
                    elif role in ("assistant", "gpt", "model"):
                        messages.append({"role": "assistant", "content": str(content)})

            sample = _make_chatml(messages, task_type="general", source="general_chat")
            if sample is not None:
                samples.append(sample)

        logger.info("chatbot_arena: collected %d / %d target samples", len(samples), budget)
        return samples

    # ===================================================================
    # Run all general loaders
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Starting General Chat data collection")
    logger.info("=" * 60)

    general_samples_oasst: list[dict[str, Any]] = load_oasst2(GENERAL_BUDGET["oasst2"])
    general_samples_arena: list[dict[str, Any]] = load_chatbot_arena(GENERAL_BUDGET["chatbot_arena"])

    all_general_samples: list[dict[str, Any]] = general_samples_oasst + general_samples_arena
    logger.info("Total General Chat samples: %d", len(all_general_samples))

    return (
        all_general_samples,
        general_samples_arena,
        general_samples_oasst,
        load_chatbot_arena,
        load_oasst2,
    )


# ---------------------------------------------------------------------------
# Cell 5: Data Integration, Mixing, Quality Check
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 5 -- Data Integration, Mixing & Quality Check

        Combine all sources, adjust ratios, deduplicate, and compute statistics.
        """
    )
    return ()


@app.cell
def _(
    Any,
    BANNED_ENV_KEYWORDS,
    TARGET_DB_COUNT,
    TARGET_ENV_COUNT,
    TARGET_GENERAL_COUNT,
    _content_hash,
    all_db_samples,
    all_env_samples,
    all_general_samples,
    json,
    logger,
    np,
    pd,
    random,
):
    # ===================================================================
    # 5-a: Deduplicate across all sources
    # ===================================================================
    def deduplicate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove exact duplicates by content hash."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for s in samples:
            h = _content_hash(s)
            if h not in seen:
                seen.add(h)
                unique.append(s)
        return unique

    logger.info("Deduplicating...")
    db_deduped = deduplicate(all_db_samples)
    env_deduped = deduplicate(all_env_samples)
    general_deduped = deduplicate(all_general_samples)

    logger.info(
        "After dedup -- DB: %d -> %d, Env: %d -> %d, General: %d -> %d",
        len(all_db_samples),
        len(db_deduped),
        len(all_env_samples),
        len(env_deduped),
        len(all_general_samples),
        len(general_deduped),
    )

    # ===================================================================
    # 5-b: Final contamination check
    # ===================================================================
    def final_contamination_check(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Double-check that no ALFWorld/TextWorld data slipped through."""
        clean: list[dict[str, Any]] = []
        removed = 0
        for s in samples:
            blob = json.dumps(s["messages"]).lower()
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
    general_clean = final_contamination_check(general_deduped)

    # ===================================================================
    # 5-c: Ratio adjustment (sample down to target counts)
    # ===================================================================
    def sample_to_budget(samples: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        """Sample down to budget if needed, or return all if under budget."""
        if len(samples) <= budget:
            logger.info("  Available %d <= budget %d, using all", len(samples), budget)
            return samples
        selected = random.sample(samples, budget)
        logger.info("  Sampled %d from %d to meet budget", budget, len(samples))
        return selected

    logger.info("Adjusting ratios...")
    db_final = sample_to_budget(db_clean, TARGET_DB_COUNT)
    env_final = sample_to_budget(env_clean, TARGET_ENV_COUNT)
    general_final = sample_to_budget(general_clean, TARGET_GENERAL_COUNT)

    # Combine all
    all_samples_combined: list[dict[str, Any]] = db_final + env_final + general_final
    random.shuffle(all_samples_combined)

    logger.info(
        "Final dataset: DB=%d, Env=%d, General=%d, Total=%d",
        len(db_final),
        len(env_final),
        len(general_final),
        len(all_samples_combined),
    )

    # ===================================================================
    # 5-d: Statistics
    # ===================================================================
    def compute_statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute comprehensive dataset statistics."""
        if not samples:
            return {"total": 0}

        task_types: dict[str, int] = {}
        sources: dict[str, int] = {}
        turn_counts: list[int] = []
        char_counts: list[int] = []
        msg_counts: list[int] = []

        for s in samples:
            tt = s["task_type"]
            src = s["source"]
            task_types[tt] = task_types.get(tt, 0) + 1
            sources[src] = sources.get(src, 0) + 1

            msgs = s["messages"]
            n_msgs = len(msgs)
            msg_counts.append(n_msgs)
            # Count user-assistant turn pairs (excluding system)
            n_turns = sum(1 for m in msgs if m["role"] == "user")
            turn_counts.append(n_turns)
            total_chars = sum(len(m["content"]) for m in msgs)
            char_counts.append(total_chars)

        stats: dict[str, Any] = {
            "total": len(samples),
            "task_type_distribution": dict(sorted(task_types.items())),
            "source_distribution": dict(sorted(sources.items())),
            "turn_count": {
                "mean": float(np.mean(turn_counts)),
                "median": float(np.median(turn_counts)),
                "min": int(np.min(turn_counts)),
                "max": int(np.max(turn_counts)),
                "std": float(np.std(turn_counts)),
            },
            "message_count": {
                "mean": float(np.mean(msg_counts)),
                "median": float(np.median(msg_counts)),
                "min": int(np.min(msg_counts)),
                "max": int(np.max(msg_counts)),
            },
            "total_chars_per_sample": {
                "mean": float(np.mean(char_counts)),
                "median": float(np.median(char_counts)),
                "min": int(np.min(char_counts)),
                "max": int(np.max(char_counts)),
            },
            "agent_to_general_ratio": (
                f"{len(db_final) + len(env_final)}:{len(general_final)}"
            ),
        }
        return stats

    dataset_stats = compute_statistics(all_samples_combined)

    # Pretty-print statistics
    logger.info("=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)
    logger.info("Total samples: %d", dataset_stats["total"])
    logger.info("Task type distribution: %s", dataset_stats.get("task_type_distribution", {}))
    logger.info("Source distribution: %s", dataset_stats.get("source_distribution", {}))
    if "turn_count" in dataset_stats:
        tc = dataset_stats["turn_count"]
        logger.info(
            "Turn count -- mean: %.1f, median: %.1f, min: %d, max: %d",
            tc["mean"],
            tc["median"],
            tc["min"],
            tc["max"],
        )
    if "total_chars_per_sample" in dataset_stats:
        cc = dataset_stats["total_chars_per_sample"]
        logger.info(
            "Chars/sample -- mean: %.0f, median: %.0f, min: %d, max: %d",
            cc["mean"],
            cc["median"],
            cc["min"],
            cc["max"],
        )
    logger.info("Agent:General ratio = %s", dataset_stats.get("agent_to_general_ratio", "N/A"))

    # Create a summary DataFrame for display
    summary_rows: list[dict[str, Any]] = []
    for src, count in sorted(dataset_stats.get("source_distribution", {}).items()):
        summary_rows.append({"source": src, "count": count})
    stats_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(columns=["source", "count"])

    return (
        all_samples_combined,
        compute_statistics,
        dataset_stats,
        db_clean,
        db_final,
        deduplicate,
        env_clean,
        env_final,
        final_contamination_check,
        general_clean,
        general_final,
        sample_to_budget,
        stats_df,
    )


# ---------------------------------------------------------------------------
# Cell 6: Output (JSONL export)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Cell 6 -- Export

        Save the final dataset as JSONL with a companion metadata JSON file.
        """
    )
    return ()


@app.cell
def _(
    Any,
    METADATA_JSON,
    OUTPUT_JSONL,
    RANDOM_SEED,
    all_samples_combined,
    dataset_stats,
    json,
    logger,
    stats_df,
):
    # ===================================================================
    # 6-a: Write JSONL
    # ===================================================================
    def write_jsonl(samples: list[dict[str, Any]], path: Any) -> int:
        """Write samples to JSONL file. Returns number of written lines."""
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
        return count

    n_written = write_jsonl(all_samples_combined, OUTPUT_JSONL)
    logger.info("Wrote %d samples to %s", n_written, OUTPUT_JSONL)

    # ===================================================================
    # 6-b: Write metadata
    # ===================================================================
    metadata: dict[str, Any] = {
        "dataset_name": "agentbench_training_set_v1",
        "version": "1.0.0",
        "description": (
            "Mixed training dataset for AgentBench DB Bench and ALFWorld score improvement. "
            "Contains DB multi-turn dialogue, environment goal-achievement trajectories, and general chat data."
        ),
        "creation_date": "2026-02-14",
        "random_seed": RANDOM_SEED,
        "format": "ChatML (Qwen-compatible)",
        "schema": {
            "messages": "list of {role, content} dicts",
            "task_type": "db | env | general",
            "source": (
                "sparc | cosql | spider | synthetic_sql"
                " | webshop | scienceworld | synthetic_react | general_chat"
            ),
        },
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
    # 6-c: Verification
    # ===================================================================
    # Read back and verify
    verified_count = 0
    with open(OUTPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            assert "messages" in obj, "Missing 'messages' field"
            assert "task_type" in obj, "Missing 'task_type' field"
            assert "source" in obj, "Missing 'source' field"
            assert obj["task_type"] in ("db", "env", "general"), f"Invalid task_type: {obj['task_type']}"
            assert len(obj["messages"]) >= 2, "Too few messages"
            verified_count += 1

    logger.info("Verification passed: %d / %d samples OK", verified_count, n_written)
    logger.info("=" * 60)
    logger.info("DONE. Dataset ready at: %s", OUTPUT_JSONL)
    logger.info("=" * 60)

    # Display summary
    print(f"\nDataset written: {OUTPUT_JSONL}")
    print(f"Metadata written: {METADATA_JSON}")
    print(f"Total samples: {n_written}")
    print(f"Verified: {verified_count}")
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
