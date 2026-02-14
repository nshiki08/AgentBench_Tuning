# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.5.0",
#     "transformers>=4.46.0",
#     "trl>=0.12.0",
#     "peft>=0.13.0",
#     "bitsandbytes>=0.44.0",
#     "accelerate>=1.1.0",
#     "datasets>=3.1.0",
#     "wandb>=0.18.0",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# Cell 1: Colab environment setup
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # DPO: Qwen2.5-7B-Instruct for AgentBench (Plan C)

        Direct Preference Optimization fine-tuning using preference pairs generated
        by `dataset/notebooks/03_generate_dpo_pairs_v1.py`.

        The model starts from an SFT checkpoint produced by
        `training/notebooks/01_sft_qwen25_7b.py`, or falls back to the base
        Qwen2.5-7B-Instruct weights when no SFT adapter is available.

        QLoRA (4-bit NF4) with a fresh LoRA adapter for the DPO stage.
        Designed to run on Google Colab with an A100 40 GB GPU.

        ## Setup

        The cell below installs required packages when running on Colab and
        verifies that a CUDA GPU is available.
        """
    )
    return (mo,)


@app.cell
def _():
    import subprocess
    import sys

    # ---------- Colab package installation ----------
    def _is_colab() -> bool:
        try:
            import google.colab  # type: ignore[import-untyped]  # noqa: F401

            return True
        except ImportError:
            return False

    if _is_colab():
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch>=2.5.0",
                "transformers>=4.46.0",
                "trl>=0.12.0",
                "peft>=0.13.0",
                "bitsandbytes>=0.44.0",
                "datasets>=3.1.0",
                "accelerate>=1.1.0",
                "wandb>=0.18.0",
            ]
        )

    # ---------- Google Drive mount (optional, for checkpoint backup) ----------
    drive_mounted = False
    if _is_colab():
        try:
            from google.colab import drive  # type: ignore[import-untyped]

            drive.mount("/content/drive")
            drive_mounted = True
            print("Google Drive mounted at /content/drive")
        except Exception as e:
            print(f"Drive mount skipped: {e}")

    # ---------- GPU check ----------
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name}  |  VRAM: {gpu_mem_gb:.1f} GB")
    else:
        print("WARNING: No CUDA GPU detected. Training will be extremely slow on CPU.")

    is_colab: bool = _is_colab()
    return drive_mounted, is_colab, subprocess, sys, torch


# ---------------------------------------------------------------------------
# Cell 2: Hyperparameters
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Hyperparameters

        All tuneable values are collected in a single dataclass so they can be
        adjusted in one place before launching a DPO training run.

        Key DPO-specific parameters:
        - **beta**: KL penalty coefficient controlling how far the policy can
          deviate from the reference model. Lower values allow more deviation.
        - **loss_type**: `"sigmoid"` is the standard DPO loss from the paper.
        - **LoRA rank 32** (lower than SFT's 64) for more stable preference
          learning on top of the already-tuned SFT model.
        """
    )
    return


@app.cell
def _(is_colab):
    from dataclasses import dataclass, field

    @dataclass
    class DPOConfig:
        """Central configuration for the DPO run."""

        # ----- Model -----
        base_model: str = "Qwen/Qwen2.5-7B-Instruct"
        sft_adapter_path: str = "./sft_qwen25_7b_agentbench/final_adapter"

        # ----- DPO hyperparameters -----
        beta: float = 0.1  # KL penalty coefficient
        loss_type: str = "sigmoid"  # standard DPO loss

        # ----- Data -----
        data_path: str = "dataset/output/dpo_pairs_v1.jsonl"
        eval_ratio: float = 0.10

        # ----- Training -----
        learning_rate: float = 5e-6
        num_train_epochs: int = 1  # 1 epoch per round
        per_device_train_batch_size: int = 2
        per_device_eval_batch_size: int = 2
        gradient_accumulation_steps: int = 8  # effective batch = 16
        warmup_ratio: float = 0.1
        lr_scheduler_type: str = "cosine"
        weight_decay: float = 0.01
        max_grad_norm: float = 1.0
        bf16: bool = True
        gradient_checkpointing: bool = True

        # ----- Sequence lengths -----
        max_length: int = 4096
        max_prompt_length: int = 2048

        # ----- LoRA (new adapter for DPO) -----
        lora_rank: int = 32  # lower than SFT (64) for stability
        lora_alpha: int = 64
        lora_dropout: float = 0.05
        lora_target_modules: list[str] = field(
            default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        # ----- Quantization -----
        use_4bit: bool = True
        bnb_4bit_compute_dtype: str = "bfloat16"
        bnb_4bit_quant_type: str = "nf4"

        # ----- Checkpointing / Logging -----
        output_dir: str = "./dpo_qwen25_7b_agentbench"
        logging_steps: int = 5
        save_steps: int = 50
        save_total_limit: int = 3
        eval_steps: int = 25

        # ----- wandb (set to "none" to disable) -----
        report_to: str = "wandb"
        wandb_project: str = "agentbench-dpo"
        wandb_run_name: str = "qwen25-7b-dpo-v1"

        # ----- Google Drive backup (Colab) -----
        drive_backup_dir: str = "/content/drive/MyDrive/agentbench_checkpoints/dpo"

        # ----- Debug / dry-run -----
        debug_max_samples: int | None = None  # set to e.g. 50 for a quick smoke test

    cfg = DPOConfig()

    # When NOT on Colab, default to tensorboard to avoid wandb login prompt
    if not is_colab and cfg.report_to == "wandb":
        cfg.report_to = "none"

    print("=== DPO Training Configuration ===")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")

    return DPOConfig, cfg, dataclass, field


# ---------------------------------------------------------------------------
# Cell 3: Dataset loading
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Dataset

        Load the JSONL preference data produced by
        `dataset/notebooks/03_generate_dpo_pairs_v1.py`.

        Each line contains:
        ```json
        {
          "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
          "chosen": [{"role": "assistant", "content": "high-scoring response"}],
          "rejected": [{"role": "assistant", "content": "low-scoring response"}],
          "chosen_score": 0.8,
          "rejected_score": -0.3,
          "task_type": "db" or "env",
          "source": "sparc"
        }
        ```

        For `trl.DPOTrainer`, we convert:
        - `prompt` messages -> a single string via the chat template
        - `chosen` / `rejected` -> the assistant content string

        The dataset is split into 90% train / 10% eval.
        """
    )
    return


@app.cell
def _(cfg):
    import json
    from pathlib import Path

    from datasets import Dataset

    # ---------- Load JSONL ----------
    dataset_path = Path(cfg.data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path.resolve()}. "
            "Please run the DPO pair generation notebook first or update cfg.data_path."
        )

    raw_records: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    print(f"Loaded {len(raw_records)} preference pairs from {dataset_path}")

    # ---------- Optional debug subset ----------
    if cfg.debug_max_samples is not None:
        raw_records = raw_records[: cfg.debug_max_samples]
        print(f"  (debug mode: using first {cfg.debug_max_samples} samples)")

    # ---------- Convert to DPOTrainer format ----------
    # DPOTrainer expects columns: prompt (list[dict]), chosen (list[dict]), rejected (list[dict])
    # When using chat templates, we pass the message lists directly and let DPOTrainer apply the template.
    dpo_records: list[dict] = []
    for record in raw_records:
        dpo_records.append(
            {
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
                "task_type": record.get("task_type", "unknown"),
                "source": record.get("source", "unknown"),
            }
        )

    # ---------- Convert to HuggingFace Dataset ----------
    full_dataset = Dataset.from_list(dpo_records)

    # ---------- Train / Eval split ----------
    split = full_dataset.train_test_split(test_size=cfg.eval_ratio, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # ---------- Statistics ----------
    task_types: dict[str, int] = {}
    sources: dict[str, int] = {}
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    for record in raw_records:
        tt = record.get("task_type", "unknown")
        task_types[tt] = task_types.get(tt, 0) + 1
        src = record.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        if "chosen_score" in record:
            chosen_scores.append(record["chosen_score"])
        if "rejected_score" in record:
            rejected_scores.append(record["rejected_score"])

    print("\n=== Dataset Statistics ===")
    print(f"  Total preference pairs : {len(raw_records)}")
    print(f"  Train samples          : {len(train_dataset)}")
    print(f"  Eval samples           : {len(eval_dataset)}")
    avg_chosen = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0.0
    avg_rejected = sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0.0
    avg_margin = avg_chosen - avg_rejected
    if chosen_scores:
        print(f"  Avg chosen score       : {avg_chosen:.3f}")
        print(f"  Avg rejected score     : {avg_rejected:.3f}")
        print(f"  Avg score margin       : {avg_margin:.3f}")
    print("  Task type distribution:")
    for tt, count in sorted(task_types.items()):
        print(f"    {tt}: {count}")
    print("  Source distribution:")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count}")

    # ---------- Preview a sample ----------
    print("\n=== Sample Preference Pair (first record) ===")
    sample = dpo_records[0]
    print(f"  Prompt roles  : {[m['role'] for m in sample['prompt']]}")
    print(f"  Chosen (trunc): {str(sample['chosen'][0]['content'])[:200]}")
    print(f"  Rejected (trunc): {str(sample['rejected'][0]['content'])[:200]}")

    return (
        Dataset,
        Path,
        avg_chosen,
        avg_margin,
        avg_rejected,
        chosen_scores,
        dataset_path,
        dpo_records,
        eval_dataset,
        full_dataset,
        json,
        raw_records,
        rejected_scores,
        sample,
        sources,
        split,
        task_types,
        train_dataset,
    )


# ---------------------------------------------------------------------------
# Cell 4: Model and tokenizer loading
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Model & Tokenizer

        Two loading strategies:

        1. **Option 1 (preferred):** Load the SFT adapter, merge it into the base
           model, then apply a fresh LoRA adapter for DPO training.
        2. **Option 2 (fallback):** Load the base Qwen2.5-7B-Instruct directly if
           no SFT adapter is available.

        Both paths use 4-bit NF4 quantization (QLoRA).

        **VRAM budget (~18-22 GB on A100 40 GB):**
        - 4-bit model: ~4 GB
        - LoRA adapters: ~0.3 GB
        - Reference model (implicit via PEFT adapter disable): ~0 GB extra
        - Optimizer states: ~1.5 GB
        - Activations (bs=2, seq=4096, grad ckpt): ~10-14 GB
        """
    )
    return


@app.cell
def _(cfg, torch):
    from pathlib import Path as PathModel

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # ---------- Quantization config ----------
    DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = DTYPE_MAP.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # ---------- Try loading from SFT adapter first, fall back to base ----------
    sft_adapter_path = PathModel(cfg.sft_adapter_path)
    loaded_from_sft = False

    if sft_adapter_path.exists() and (sft_adapter_path / "adapter_config.json").exists():
        print(f"Found SFT adapter at: {sft_adapter_path.resolve()}")
        print("Loading SFT adapter and merging into base model...")

        # Load the PEFT model with quantization -- this loads the base model
        # automatically from the adapter config and applies the LoRA weights.
        sft_model = AutoPeftModelForCausalLM.from_pretrained(
            str(sft_adapter_path),
            torch_dtype=compute_dtype,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        # Merge LoRA weights into the base model and unload the adapter
        model = sft_model.merge_and_unload()
        model.config.use_cache = False
        loaded_from_sft = True
        print("SFT adapter merged successfully. Model is now the SFT-tuned base.")
    else:
        print(f"SFT adapter not found at: {sft_adapter_path.resolve()}")
        print(f"Falling back to base model: {cfg.base_model}")
        print("WARNING: DPO without SFT pre-training may produce suboptimal results.")

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False

    # ---------- Tokenizer ----------
    # Always load from base model to ensure consistency
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        padding_side="left",  # DPO typically uses left padding for generation consistency
    )

    # Ensure pad_token is set (Qwen uses eos_token as pad by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"\nModel loaded: {'SFT-merged' if loaded_from_sft else cfg.base_model}")
    print(f"  dtype          : {model.dtype}")
    print(f"  pad_token      : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"  eos_token      : {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  padding_side   : {tokenizer.padding_side}")
    print(f"  vocab_size     : {tokenizer.vocab_size}")
    print(f"  chat_template  : {'set' if tokenizer.chat_template else 'NOT set'}")

    # ---------- Verify chat template ----------
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    formatted = tokenizer.apply_chat_template(sample_messages, tokenize=False, add_generation_prompt=True)
    print(f"\n=== Chat template sample ===\n{formatted}")

    return (
        AutoModelForCausalLM,
        AutoPeftModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PathModel,
        bnb_config,
        compute_dtype,
        formatted,
        loaded_from_sft,
        model,
        sample_messages,
        sft_adapter_path,
        tokenizer,
    )


# ---------------------------------------------------------------------------
# Cell 5: DPO training
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## DPO Training

        Uses `trl.DPOTrainer` with a fresh LoRA adapter applied to the
        (optionally SFT-merged) base model.

        **Key design decisions:**
        - `DPOTrainer` manages the reference model internally via PEFT: it
          disables the adapter to get reference logprobs, avoiding the need
          to load a separate copy of the model.
        - The `peft_config` argument tells `DPOTrainer` to apply a new LoRA
          adapter before training starts.
        - `remove_unused_columns=False` preserves metadata columns (task_type,
          source) through the data pipeline.
        - A lightweight callback generates sample completions at each eval step
          for qualitative monitoring.
        """
    )
    return


@app.cell
def _(cfg, eval_dataset, model, tokenizer, torch, train_dataset):
    import os

    from peft import LoraConfig, TaskType
    from transformers import TrainerCallback
    from trl import DPOConfig as TrlDPOConfig
    from trl import DPOTrainer

    # ---------- wandb setup ----------
    if cfg.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    # ---------- LoRA config (fresh adapter for DPO) ----------
    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ---------- Sample generation callback ----------
    class DPOSampleGenerationCallback(TrainerCallback):
        """Generate a sample completion at each evaluation step for qualitative monitoring."""

        SAMPLE_PROMPTS: list[list[dict[str, str]]] = [
            [
                {"role": "system", "content": "You are a helpful assistant for database operations."},
                {"role": "user", "content": "Show me all tables in the database."},
            ],
            [
                {"role": "system", "content": "You are a household robot assistant."},
                {"role": "user", "content": "Put a clean plate on the countertop."},
            ],
        ]

        def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):  # type: ignore[override]
            if model is None or tokenizer is None:
                return
            model.eval()
            print(f"\n--- Sample generations (eval step {state.global_step}) ---")
            for i, prompt_messages in enumerate(self.SAMPLE_PROMPTS):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                print(f"\n[Sample {i + 1}]\n  Prompt: {prompt_messages[-1]['content']}\n  Response: {generated[:300]}")
            model.train()

    # ---------- DPO Training Arguments ----------
    training_args = TrlDPOConfig(
        output_dir=cfg.output_dir,
        # DPO-specific
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        # Batch / accumulation
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        # Optimizer / schedule
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        # Sequence lengths
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        # Precision / memory
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging / checkpointing
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=cfg.report_to,
        run_name=cfg.wandb_run_name if cfg.report_to == "wandb" else None,
        # Misc
        seed=42,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # ---------- DPOTrainer ----------
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[DPOSampleGenerationCallback()],
    )

    # ---------- Print summary ----------
    trainable_params = 0
    all_params = 0
    for _, param in trainer.model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = 100 * trainable_params / all_params if all_params > 0 else 0

    print("=== DPOTrainer ready ===")
    print(f"  Train samples         : {len(train_dataset)}")
    print(f"  Eval samples          : {len(eval_dataset)}")
    print(f"  Effective batch       : {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    print(f"  Max length            : {cfg.max_length}")
    print(f"  Max prompt length     : {cfg.max_prompt_length}")
    print(f"  Beta (KL penalty)     : {cfg.beta}")
    print(f"  Loss type             : {cfg.loss_type}")
    print(f"  LoRA rank             : {cfg.lora_rank}")
    print(f"  Trainable parameters  : {trainable_params:,}")
    print(f"  Total parameters      : {all_params:,}")
    print(f"  Trainable %           : {trainable_pct:.2f}%")

    # ---------- Launch training ----------
    train_result = trainer.train()
    print("\n=== DPO Training complete ===")
    print(f"  Global steps  : {trainer.state.global_step}")
    print(f"  Train loss    : {train_result.training_loss:.4f}")

    return (
        DPOSampleGenerationCallback,
        DPOTrainer,
        LoraConfig,
        TaskType,
        TrainerCallback,
        TrlDPOConfig,
        all_params,
        os,
        peft_config,
        train_result,
        trainable_params,
        trainable_pct,
        trainer,
        training_args,
    )


# ---------------------------------------------------------------------------
# Cell 6: Save and evaluate
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Save & Evaluate

        Save the final DPO LoRA adapter, run evaluation metrics, test inference
        with sample prompts from both task types, and optionally back up to
        Google Drive.
        """
    )
    return


@app.cell
def _(cfg, drive_mounted, is_colab, tokenizer, torch, trainer):
    import shutil
    from pathlib import Path as PathFinal

    # ---------- Save final adapter ----------
    final_adapter_dir = PathFinal(cfg.output_dir) / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"DPO LoRA adapter saved to: {final_adapter_dir.resolve()}")

    # ---------- Training metrics ----------
    metrics = trainer.evaluate()
    print("\n=== Final Eval Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # ---------- Inference test ----------
    print("\n=== Inference Test (DPO-trained model) ===")
    test_prompts: list[list[dict[str, str]]] = [
        [
            {"role": "system", "content": "You are a helpful assistant for database operations."},
            {
                "role": "user",
                "content": (
                    "I have a database with tables: customers(id, name, email), "
                    "orders(id, customer_id, total, created_at). "
                    "List the top 5 customers by total order amount."
                ),
            },
        ],
        [
            {
                "role": "system",
                "content": (
                    "You are a household robot assistant operating in a simulated environment. "
                    "Interact with the environment by outputting actions."
                ),
            },
            {"role": "user", "content": "Find and pick up a mug, then place it on the dining table."},
        ],
    ]

    trained_model = trainer.model
    trained_model.eval()

    for i, messages in enumerate(test_prompts):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(trained_model.device)
        with torch.no_grad():
            output_ids = trained_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        task_label = "DB Bench" if i == 0 else "ALFWorld"
        print(f"\n[{task_label} sample]")
        print(f"  User   : {messages[-1]['content'][:150]}")
        print(f"  Model  : {generated[:500]}")

    # ---------- Google Drive backup ----------
    drive_backup_dest = None
    if is_colab and drive_mounted:
        drive_dest = PathFinal(cfg.drive_backup_dir)
        drive_dest.mkdir(parents=True, exist_ok=True)
        drive_backup_dest = drive_dest / "final_adapter"
        if drive_backup_dest.exists():
            shutil.rmtree(drive_backup_dest)
        shutil.copytree(final_adapter_dir, drive_backup_dest)
        print(f"\nAdapter backed up to Google Drive: {drive_backup_dest}")
    else:
        print("\nGoogle Drive backup skipped (not on Colab or Drive not mounted).")

    print("\nDone. The DPO LoRA adapter can be loaded with:")
    print("  from peft import AutoPeftModelForCausalLM")
    print(f'  model = AutoPeftModelForCausalLM.from_pretrained("{final_adapter_dir}")')

    return (
        PathFinal,
        drive_backup_dest,
        final_adapter_dir,
        generated,
        i,
        messages,
        metrics,
        shutil,
        test_prompts,
        trained_model,
    )


if __name__ == "__main__":
    app.run()
