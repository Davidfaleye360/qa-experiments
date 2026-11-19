"""Tests for src/distill.py: lining up the teacher's scores, the loss, and a tiny distillation run.

The teacher in these tests is a tiny random DistilBERT with the local test tokenizer, so they run offline.
"""
from __future__ import annotations

import pytest
import torch

from conftest import CONTEXT_CURIE, CONTEXT_FAME, ROOT, make_example
from src.config import Config, TrainingConfig, load_config
from src.distill import (
    IMPOSSIBLE,
    DistillationCollator,
    add_teacher_scores,
    align_teacher_scores,
    distillation_loss,
    make_distill_features,
    teacher_agreement,
)
from src.train import run_training


def test_scores_move_to_the_student_piece_at_the_same_character():
    # Passage "unhappy cats": the teacher has one piece "unhappy" (0-7) and "cats" (8-12);
    # the student splits "un" (0-2) + "##happy" (2-7) and "cat" (8-11) + "##s" (11-12).
    teacher_spans = [None, None, (0, 7), (8, 12), None]          # [CLS], question, passage..., [SEP]
    teacher_start = [5.0, 9.0, 3.0, 1.0, 7.0]
    teacher_end = [6.0, 9.0, 2.0, 4.0, 7.0]
    student_spans = [None, None, (0, 2), (2, 7), (8, 11), (11, 12), None]
    starts, ends = align_teacher_scores(teacher_start, teacher_end, teacher_spans, student_spans)

    assert starts[0] == 5.0 and ends[0] == 6.0, "[CLS] keeps the teacher's no-answer score"
    assert starts[2] == 3.0 and starts[4] == 1.0, "word starts get the teacher's start scores"
    assert starts[3] == IMPOSSIBLE and starts[5] == IMPOSSIBLE, "an answer can't start mid-word"
    assert ends[3] == 2.0 and ends[5] == 4.0, "word ends get the teacher's end scores"
    assert ends[2] == IMPOSSIBLE and ends[4] == IMPOSSIBLE
    assert starts[1] == starts[6] == IMPOSSIBLE, "the question and [SEP] are never an answer"


def test_empty_teacher_pieces_are_ignored():
    # A lone space piece (5, 5) must not take over position 5.
    starts, ends = align_teacher_scores([0.0, 8.0, 1.0], [0.0, 8.0, 1.0], [None, (5, 5), (5, 9)], [None, (5, 9)])
    assert starts == [0.0, 1.0] and ends == [0.0, 1.0]


def test_loss_is_the_label_loss_without_a_teacher_weight_and_zero_teacher_part_for_a_copy():
    torch.manual_seed(0)
    student_start, student_end = torch.randn(3, 10), torch.randn(3, 10)
    positions = torch.tensor([0, 4, 7])
    total, hard, teacher_part = distillation_loss(student_start, student_end, student_start.clone(),
                                                  student_end.clone(), positions, positions, 2.0, 0.0)
    assert total == pytest.approx(hard.item())
    assert teacher_part.item() == pytest.approx(0.0, abs=1e-6), "copying the teacher exactly costs nothing"

    other = torch.randn(3, 10)
    total, hard, teacher_part = distillation_loss(student_start, student_end, other, other, positions, positions, 2.0, 0.5)
    assert teacher_part > 0 and total == pytest.approx(0.5 * hard.item() + 0.5 * teacher_part.item(), rel=1e-5)


@pytest.fixture
def tiny(tokenizer, tmp_path):
    datasets = pytest.importorskip("datasets")
    from transformers import DistilBertConfig, DistilBertForQuestionAnswering

    rows = [
        make_example("a", "When did Curie win the Chemistry prize?", CONTEXT_CURIE, "1911"),
        make_example("b", "Where was Curie born?", CONTEXT_CURIE, None),
        make_example("c", "When did she rise to fame?", CONTEXT_FAME, "in the late 1990s"),
        make_example("d", "What did Curie win in 1903?", CONTEXT_CURIE, "the Nobel Prize in Physics"),
    ] * 2
    cfg = Config()
    cfg.preprocessing.max_length, cfg.preprocessing.doc_stride = 64, 16
    cfg.training = TrainingConfig(output_dir=str(tmp_path / "run"), batch_size=4, max_steps=4,
                                  logging_steps=1, save_steps=2, fp16=False)
    cfg.distillation.teacher = "tiny-test-teacher"
    cfg.distillation.teacher_batch_size = 3

    def model():
        return DistilBertForQuestionAnswering(DistilBertConfig(
            vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2, max_position_embeddings=128))

    torch.manual_seed(0)
    return datasets.Dataset.from_list(rows), cfg, model


def test_teacher_with_the_same_word_pieces_is_copied_exactly(tokenizer, tiny):
    raw, cfg, model = tiny
    teacher = model().eval()
    batch = raw[:4]
    out = add_teacher_scores(batch, tokenizer, teacher, tokenizer, cfg, torch.device("cpu"))

    # The same tokenizer on the same text: every passage piece lines up, so the scores must equal the
    # teacher's own scores on the student's window (passage pieces and [CLS]).
    for i, ids in enumerate(out["input_ids"]):
        with torch.no_grad():
            own = teacher(input_ids=torch.tensor([ids]))
        seq = out.sequence_ids(i)
        for k, seq_id in enumerate(seq):
            if seq_id == 1 or k == 0:
                assert out["teacher_start"][i][k] == pytest.approx(own.start_logits[0, k].item(), abs=1e-4)
                assert out["teacher_end"][i][k] == pytest.approx(own.end_logits[0, k].item(), abs=1e-4)
            else:
                assert out["teacher_start"][i][k] == IMPOSSIBLE
        assert len(out["teacher_start"][i]) == len(ids)


def test_tiny_distillation_run_trains_saves_and_reuses_the_teacher_scores(tokenizer, tiny, capsys):
    raw, cfg, model = tiny
    features = make_distill_features(raw, tokenizer, cfg, device=torch.device("cpu"),
                                     teacher=model(), teacher_tokenizer=tokenizer)
    assert set(features.column_names) >= {"input_ids", "start_positions", "teacher_start", "teacher_end"}
    assert teacher_agreement(features)["windows"] == len(features)

    # The collator pads the teacher's scores like the word pieces.
    batch = DistillationCollator(tokenizer)([features[0], features[2]])
    assert batch["teacher_start"].shape == batch["input_ids"].shape
    assert batch["teacher_start"][batch["attention_mask"] == 0].eq(IMPOSSIBLE).all()

    summary = run_training(model(), tokenizer, features, cfg, use_cpu=True)
    assert summary["steps"] == 4 and summary["final_model_saved"] and summary["teacher"] == "tiny-test-teacher"

    # A second call finds the saved scores and does not need the teacher.
    again = make_distill_features(raw, tokenizer, cfg, device=torch.device("cpu"), teacher=None)
    assert "Using the teacher scores saved" in capsys.readouterr().out
    assert again["teacher_start"] == features["teacher_start"]


def test_the_distillation_config_differs_from_the_baseline_only_where_it_must():
    base = load_config(ROOT / "configs" / "distilbert.yaml")
    distill = load_config(ROOT / "configs" / "distill.yaml")
    assert distill.model.checkpoint == base.model.checkpoint
    assert distill.preprocessing == base.preprocessing and distill.postprocessing == base.postprocessing
    for name in ("learning_rate", "batch_size", "num_epochs", "warmup_ratio", "weight_decay", "seed"):
        assert getattr(distill.training, name) == getattr(base.training, name), name
    assert distill.model.hub_model_id != base.model.hub_model_id
    assert distill.distillation.teacher == "Ingsyling/deberta-v3-squad2" and base.distillation.teacher is None
