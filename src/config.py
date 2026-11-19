"""Load the project's YAML config into simple, typed objects.

Usage:
    from src.config import load_config
    cfg = load_config("configs/distilbert.yaml")
    cfg.preprocessing.max_length  # 384

    # The same settings with the smoke-test changes applied on top:
    smoke = load_config("configs/distilbert.yaml", overrides="configs/smoke.yaml")
    smoke.training.max_steps      # 50
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    checkpoint: str = "distilbert-base-uncased"
    hub_model_id: Optional[str] = None


@dataclass
class DataConfig:
    dataset_name: str = "rajpurkar/squad_v2"
    train_fraction: float = 1.0
    eval_fraction: float = 1.0


@dataclass
class PreprocessingConfig:
    max_length: int = 384
    doc_stride: int = 128


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/distilbert-squad2"
    learning_rate: float = 3e-5
    batch_size: int = 16
    # Windows per weight update = batch_size x gradient_accumulation_steps. A bigger model can run out of GPU
    # memory at batch_size 16; batch_size 8 with 2 accumulation steps gives the same update from two halves.
    gradient_accumulation_steps: int = 1
    # Put windows of similar length in the same batch (shuffled in large groups first), so batches need
    # less padding. Saves about half the work on SQuAD; the model sees the same data.
    group_by_length: bool = False
    num_epochs: int = 2
    max_steps: int = -1        # -1 = train for num_epochs; a positive number stops after that many steps
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    logging_steps: int = 100   # how often the training loss is recorded
    save_steps: int = 2000
    save_total_limit: int = 2
    seed: int = 42


@dataclass
class PostprocessingConfig:
    n_best: int = 20
    max_answer_length: int = 30
    null_threshold: float = 0.0


@dataclass
class DistillationConfig:
    """Settings for learning from a teacher model's scores (src/distill.py). No teacher = normal training."""
    teacher: Optional[str] = None      # Hub id or folder of a trained QA model, e.g. Ingsyling/deberta-v3-squad2
    # Softens both models' scores before comparing them, so the student also learns which runner-up
    # answers the teacher found plausible, not only its top pick. 1 = unchanged; 2 is the usual choice.
    temperature: float = 2.0
    # Weight of the teacher's scores in the loss; the rest (1 - alpha) goes to the SQuAD labels.
    alpha: float = 0.5
    teacher_batch_size: int = 32       # windows per teacher forward pass (no training, so it can be large)
    # The teacher reads the same text as each student window. Its word pieces differ, so the same text can
    # take a few more of them; 512 leaves room. DeBERTa-v3 handles 512.
    teacher_max_length: int = 512


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)


def _build(cls, values: dict[str, Any] | None):
    """Create a dataclass from a dict, rejecting unknown keys so typos fail loudly."""
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown setting(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dict of sections, rejecting section names we don't know."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config section(s) in {path}: {sorted(unknown)}")
    return raw


def load_config(
    path: str | Path = "configs/distilbert.yaml",
    overrides: str | Path | None = None,
) -> Config:
    """Read a YAML config file. Missing settings fall back to the defaults above.

    `overrides` is an optional second YAML file that lists only the settings to change,
    such as configs/smoke.yaml for the 50-step smoke test. Everything it doesn't mention
    keeps its value from the main file, so the two runs can't drift apart by accident.
    """
    raw = _read_yaml(path)
    if overrides is not None:
        for section, values in _read_yaml(overrides).items():
            raw[section] = {**(raw.get(section) or {}), **(values or {})}
    return Config(
        model=_build(ModelConfig, raw.get("model")),
        data=_build(DataConfig, raw.get("data")),
        preprocessing=_build(PreprocessingConfig, raw.get("preprocessing")),
        training=_build(TrainingConfig, raw.get("training")),
        postprocessing=_build(PostprocessingConfig, raw.get("postprocessing")),
        distillation=_build(DistillationConfig, raw.get("distillation")),
    )
