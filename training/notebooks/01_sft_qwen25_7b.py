# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.5.0",
#     "transformers>=4.46.0",
#     "trl>=0.12.0",
#     "peft>=0.13.0",
#     "bitsandbytes>=0.44.0",
#     "datasets>=3.1.0",
#     "accelerate>=1.1.0",
#     "wandb>=0.18.0",
#     "scipy>=1.14.0",
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
        # SFT: Qwen2.5-7B-Instruct for AgentBench (DB Bench + ALFWorld)

        QLoRA (4-bit NF4) fine-tuning with response-only loss masking.
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
                "scipy>=1.14.0",
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

        All tuneable values are collected in a single dictionary so they can be
        adjusted in one place before launching a training run.
        """
    )
    return


@app.cell
def _(is_colab):
    from dataclasses import dataclass, field

    @dataclass
    class TrainingConfig:
        """Central configuration for the SFT run."""

        # ----- Model -----
        model_name: str = "Qwen/Qwen2.5-7B-Instruct"

        # ----- Data -----
        dataset_path: str = "dataset/notebooks/train_data_v1.jsonl"  # path relative to project root or absolute
        max_seq_length: int = 4096
        eval_ratio: float = 0.10

        # ----- QLoRA -----
        lora_rank: int = 64
        lora_alpha: int = 128
        lora_dropout: float = 0.05
        lora_target_modules: list[str] = field(
            default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        # ----- Training -----
        num_train_epochs: int = 3
        per_device_train_batch_size: int = 2
        per_device_eval_batch_size: int = 2
        gradient_accumulation_steps: int = 8  # effective batch size = 2 * 8 = 16
        learning_rate: float = 2e-5
        lr_scheduler_type: str = "cosine"
        warmup_ratio: float = 0.05
        weight_decay: float = 0.01
        max_grad_norm: float = 1.0
        bf16: bool = True
        gradient_checkpointing: bool = True
        optim: str = "adamw_torch_fused"

        # ----- Checkpointing / Logging -----
        output_dir: str = "./sft_qwen25_7b_agentbench"
        save_steps: int = 100
        eval_steps: int = 50
        logging_steps: int = 10
        save_total_limit: int = 3

        # ----- wandb (set to "none" to disable) -----
        report_to: str = "wandb"
        wandb_project: str = "agentbench-sft"
        wandb_run_name: str = "qwen25-7b-sft-v1"

        # ----- Google Drive backup (Colab) -----
        drive_backup_dir: str = "/content/drive/MyDrive/agentbench_checkpoints"

        # ----- Response template for completion-only loss -----
        # Qwen ChatML: <|im_start|>assistant\n  marks where model response begins
        response_template: str = "<|im_start|>assistant\n"

        # ----- Debug / dry-run -----
        debug_max_samples: int | None = None  # set to e.g. 50 for a quick smoke test

    cfg = TrainingConfig()

    # When NOT on Colab, default to tensorboard to avoid wandb login prompt
    if not is_colab and cfg.report_to == "wandb":
        cfg.report_to = "none"

    print("=== Training Configuration ===")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")

    return TrainingConfig, cfg, dataclass, field


# ---------------------------------------------------------------------------
# Cell 3: Dataset loading
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Dataset

        Load the JSONL training data produced by
        `dataset/notebooks/01_build_training_set_v1.py`.

        Each line contains:
        ```json
        {"messages": [...], "task_type": "...", "source": "..."}
        ```

        The dataset is split into 90 % train / 10 % eval.
        """
    )
    return


@app.cell
def _(cfg):
    import json
    from pathlib import Path

    from datasets import Dataset

    # ---------- Load JSONL ----------
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path.resolve()}. "
            "Please run the dataset creation notebook first or update cfg.dataset_path."
        )

    raw_records: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    print(f"Loaded {len(raw_records)} records from {dataset_path}")

    # ---------- Optional debug subset ----------
    if cfg.debug_max_samples is not None:
        raw_records = raw_records[: cfg.debug_max_samples]
        print(f"  (debug mode: using first {cfg.debug_max_samples} samples)")

    # ---------- Convert to HuggingFace Dataset ----------
    full_dataset = Dataset.from_list(raw_records)

    # ---------- Train / Eval split ----------
    split = full_dataset.train_test_split(test_size=cfg.eval_ratio, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # ---------- Statistics ----------
    task_types: dict[str, int] = {}
    total_turns = 0
    for record in raw_records:
        tt = record.get("task_type", "unknown")
        task_types[tt] = task_types.get(tt, 0) + 1
        total_turns += len(record.get("messages", []))

    avg_turns = total_turns / len(raw_records) if raw_records else 0

    print("\n=== Dataset Statistics ===")
    print(f"  Total samples : {len(raw_records)}")
    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Eval samples  : {len(eval_dataset)}")
    print(f"  Avg turns/sample: {avg_turns:.1f}")
    print("  Task type distribution:")
    for tt, count in sorted(task_types.items()):
        print(f"    {tt}: {count}")

    return (
        Dataset,
        Path,
        avg_turns,
        dataset_path,
        eval_dataset,
        full_dataset,
        json,
        raw_records,
        split,
        task_types,
        total_turns,
        train_dataset,
    )


# ---------------------------------------------------------------------------
# Cell 4: Model and tokenizer
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Model & Tokenizer

        Load Qwen2.5-7B-Instruct with 4-bit NF4 quantization (QLoRA).
        The tokenizer is configured with the correct padding token.
        """
    )
    return


@app.cell
def _(cfg, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # ---------- Quantization config ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ---------- Model ----------
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Ensure pad_token is set (Qwen uses eos_token as pad by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded: {cfg.model_name}")
    print(f"  dtype          : {model.dtype}")
    print(f"  pad_token      : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"  eos_token      : {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
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
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        bnb_config, formatted, model, sample_messages, tokenizer,
    )


# ---------------------------------------------------------------------------
# Cell 5: LoRA configuration
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## LoRA Adapter

        Apply a LoRA adapter to the quantised model. Only the adapter weights
        are trained, keeping the base model frozen.
        """
    )
    return


@app.cell
def _(cfg, model):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    # Prepare the quantised model for training (freeze base, enable grad for adapters)
    model_prepared = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model_prepared, lora_config)

    # ---------- Parameter summary ----------
    trainable_params = 0
    all_params = 0
    for _, param in peft_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = 100 * trainable_params / all_params if all_params > 0 else 0
    print("=== LoRA Adapter Applied ===")
    print(f"  Trainable parameters : {trainable_params:,}")
    print(f"  Total parameters     : {all_params:,}")
    print(f"  Trainable %          : {trainable_pct:.2f}%")
    print(f"  LoRA rank            : {cfg.lora_rank}")
    print(f"  LoRA alpha           : {cfg.lora_alpha}")
    print(f"  Target modules       : {cfg.lora_target_modules}")

    return (
        LoraConfig,
        TaskType,
        all_params,
        get_peft_model,
        lora_config,
        model_prepared,
        peft_model,
        prepare_model_for_kbit_training,
        trainable_params,
        trainable_pct,
    )


# ---------------------------------------------------------------------------
# Cell 6: SFTTrainer setup and training
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Training

        Uses `trl.SFTTrainer` with `DataCollatorForCompletionOnlyLM` so that
        the loss is computed only on assistant responses (not on system/user
        turns).

        A lightweight callback generates a sample completion every
        `eval_steps` to allow qualitative monitoring during training.
        """
    )
    return


@app.cell
def _(cfg, eval_dataset, peft_model, tokenizer, torch, train_dataset):
    import os

    from transformers import TrainerCallback, TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    # ---------- wandb setup ----------
    if cfg.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    # ---------- Response-only data collator ----------
    # The response_template marks where the assistant response starts.
    # Tokenize it to get the token-id sequence that acts as the boundary.
    response_template_ids: list[int] = tokenizer.encode(cfg.response_template, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # ---------- Formatting function ----------
    # SFTTrainer expects a function that converts each sample dict into a
    # single formatted string using the chat template.
    def formatting_func(examples: dict) -> list[str]:
        """Apply the Qwen ChatML template to each messages list."""
        output_texts: list[str] = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    # ---------- Sample generation callback ----------
    class SampleGenerationCallback(TrainerCallback):
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

    # ---------- TrainingArguments ----------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.optim,
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
        seed=42,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # required because we use a formatting_func
    )

    # ---------- SFTTrainer ----------
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        formatting_func=formatting_func,
        max_seq_length=cfg.max_seq_length,
        packing=False,
        callbacks=[SampleGenerationCallback()],
    )

    print("=== SFTTrainer ready ===")
    print(f"  Train samples     : {len(train_dataset)}")
    print(f"  Eval samples      : {len(eval_dataset)}")
    print(f"  Effective batch   : {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    print(f"  Max seq length    : {cfg.max_seq_length}")
    print(f"  Response template : {cfg.response_template!r}")
    print(f"  Response template token ids: {response_template_ids}")

    # ---------- Launch training ----------
    train_result = trainer.train()
    print("\n=== Training complete ===")
    print(f"  Global steps  : {trainer.state.global_step}")
    print(f"  Train loss    : {train_result.training_loss:.4f}")

    return (
        DataCollatorForCompletionOnlyLM,
        SFTTrainer,
        SampleGenerationCallback,
        TrainerCallback,
        TrainingArguments,
        data_collator,
        formatting_func,
        os,
        response_template_ids,
        train_result,
        trainer,
        training_args,
    )


# ---------------------------------------------------------------------------
# Cell 7: Checkpoint saving and evaluation
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        """
        ## Save & Evaluate

        Save the final LoRA adapter, run a quick inference test, and optionally
        back up to Google Drive.
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
    print(f"LoRA adapter saved to: {final_adapter_dir.resolve()}")

    # ---------- Training metrics ----------
    metrics = trainer.evaluate()
    print("\n=== Final Eval Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # ---------- Inference test ----------
    print("\n=== Inference Test ===")
    test_prompts: list[list[dict[str, str]]] = [
        [
            {"role": "system", "content": "You are a helpful assistant for database operations."},
            {"role": "user", "content": "List the top 5 customers by total order amount."},
        ],
        [
            {"role": "system", "content": "You are a household robot assistant operating in a simulated environment."},
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
        print(f"  User   : {messages[-1]['content']}")
        print(f"  Model  : {generated[:500]}")

    # ---------- Google Drive backup ----------
    if is_colab and drive_mounted:
        drive_dest = PathFinal(cfg.drive_backup_dir)
        drive_dest.mkdir(parents=True, exist_ok=True)
        dest = drive_dest / "final_adapter"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(final_adapter_dir, dest)
        print(f"\nAdapter backed up to Google Drive: {dest}")
    else:
        print("\nGoogle Drive backup skipped (not on Colab or Drive not mounted).")

    print("\nDone. The LoRA adapter can be loaded with:")
    print("  from peft import AutoPeftModelForCausalLM")
    print(f'  model = AutoPeftModelForCausalLM.from_pretrained("{final_adapter_dir}")')

    return PathFinal, final_adapter_dir, generated, i, messages, metrics, shutil, test_prompts, trained_model


if __name__ == "__main__":
    app.run()
