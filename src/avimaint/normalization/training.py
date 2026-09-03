"""ByT5 fine-tuning with development-only model selection."""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from avimaint.normalization.io import read_table, require_columns, sha256_file, write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset(frame: pd.DataFrame, tokenizer: Any, model_config: dict[str, Any]) -> Dataset:
    require_columns(frame, ["input_text", "target_text"], "training data")
    dataset = Dataset.from_pandas(frame[["input_text", "target_text"]], preserve_index=False)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["input_text"],
            text_target=batch["target_text"],
            max_length=int(model_config["max_source_length"]),
            truncation=True,
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def _compute_metrics(tokenizer: Any):
    from avimaint.normalization.metrics import sanitize_generated_token_ids, score_corpus

    def compute(result: Any) -> dict[str, float]:
        predictions = result.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        # Seq2SeqTrainer may use -100 to pad generated sequences while
        # concatenating evaluation batches. ByT5 decodes token ids as Unicode
        # code points and therefore raises ValueError when -100 (or any other
        # out-of-vocabulary id) reaches batch_decode.
        predictions = sanitize_generated_token_ids(
            predictions, tokenizer.pad_token_id, len(tokenizer)
        )
        labels = np.where(result.label_ids == -100, tokenizer.pad_token_id, result.label_ids)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Source-dependent metrics are computed in the separate evaluator.
        scores = score_corpus(decoded_references, decoded_predictions, decoded_references)
        return {"wer": scores["wer"], "cer": scores["cer"], "exact_match": scores["exact_match"]}

    return compute


def _train_stage(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    base_model: str,
    output_dir: Path,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    epochs: float,
    seed: int,
) -> Path:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if training_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    train_dataset = _dataset(train_frame, tokenizer, model_config)
    eval_dataset = (
        _dataset(validation_frame, tokenizer, model_config)
        if validation_frame is not None and not validation_frame.empty
        else None
    )
    use_eval = eval_dataset is not None
    fp16 = bool(training_config.get("fp16", True) and torch.cuda.is_available())
    bf16_supported = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    bf16 = bool(training_config.get("bf16", False) and bf16_supported)
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(epochs),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.0)),
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
        optim=str(training_config.get("optim", "adafactor")),
        fp16=fp16,
        bf16=bf16,
        eval_strategy="epoch" if use_eval else "no",
        save_strategy="epoch" if use_eval else "no",
        logging_strategy="steps",
        logging_steps=25,
        predict_with_generate=use_eval,
        generation_max_length=int(model_config["max_target_length"]),
        generation_num_beams=int(model_config.get("num_beams", 1)),
        load_best_model_at_end=use_eval,
        metric_for_best_model=str(training_config.get("metric_for_best_model", "eval_wer")),
        greater_is_better=bool(training_config.get("greater_is_better", False)),
        save_total_limit=int(training_config.get("save_total_limit", 2)),
        seed=seed,
        data_seed=seed,
        report_to=[],
        save_safetensors=True,
    )
    callbacks = []
    if use_eval:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(training_config.get("early_stopping_patience", 3))
            )
        )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=_compute_metrics(tokenizer) if use_eval else None,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def train_from_config(config: dict[str, Any]) -> Path:
    run = config["run"]
    model_config = config["model"]
    training_config = config["training"]
    output_dir = Path(run["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(run["seed"])
    stage = run["stage"]

    if stage == "gold_only":
        source_path = run["input_path"]
        frame = read_table(source_path)
        require_columns(frame, ["split"], "gold split")
        train = frame[frame["split"] == "train"].copy()
        validation = frame[frame["split"] == "validation"].copy()
        if train.empty or validation.empty:
            raise ValueError("Gold split must contain non-empty train and validation partitions")
        final_dir = _train_stage(
            train,
            validation,
            model_config["name_or_path"],
            output_dir,
            model_config,
            training_config,
            float(training_config["max_epochs"]),
            seed,
        )
        input_hashes = {source_path: sha256_file(source_path)}
    elif stage == "silver_then_gold":
        silver_path = run["silver_input_path"]
        gold_path = run["gold_input_path"]
        silver = read_table(silver_path)
        gold = read_table(gold_path)
        require_columns(gold, ["split"], "gold split")
        pretrain_dir = output_dir / "silver_stage"
        _train_stage(
            silver,
            None,
            model_config["name_or_path"],
            pretrain_dir,
            model_config,
            training_config,
            float(training_config["silver_epochs"]),
            seed,
        )
        train = gold[gold["split"] == "train"].copy()
        validation = gold[gold["split"] == "validation"].copy()
        final_dir = _train_stage(
            train,
            validation,
            str(pretrain_dir),
            output_dir / "gold_stage",
            model_config,
            training_config,
            float(training_config["gold_max_epochs"]),
            seed,
        )
        input_hashes = {silver_path: sha256_file(silver_path), gold_path: sha256_file(gold_path)}
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    manifest = {
        "run_id": run["id"],
        "stage": stage,
        "seed": seed,
        "base_model": model_config["name_or_path"],
        "final_model_dir": str(final_dir),
        "input_hashes": input_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": config,
    }
    write_json(manifest, output_dir / "run_manifest.json")
    return final_dir
