"""Distillation: train DistilBERT to copy a stronger teacher model's scores (experiment 2).

    python -m src.train --config configs/distill.yaml --overrides configs/smoke.yaml --fresh   # trial run
    python -m src.train --config configs/distill.yaml --output-dir <Drive folder> --push-to-hub

The idea in one paragraph. Normal training shows the student one right answer per window: "the
answer starts at word piece 57" and nothing else. A trained teacher (our DeBERTa-v3, 87.7 F1) gives
a score to EVERY word piece: "57 is most likely, 55 is a close second, and the date at 80 is not
absurd either". That tells the student far more per window, including how sure to be about "no
answer". The student learns from both: half its loss compares it with the SQuAD labels as before,
half compares it with the teacher's scores (see `distillation_loss`).

The catch: the two models cut text into different word pieces. DeBERTa-v3's tokenizer is not
DistilBERT's, so the teacher's score for its word piece 57 says nothing about the student's word
piece 57. We line them up by CHARACTER position instead (`align_teacher_scores`):
  1. For each student training window, find the stretch of the passage it covers (in characters).
  2. Give the teacher the same question and exactly that stretch of passage.
  3. A student word piece that starts at character 1,204 gets the teacher's START score of the
     teacher word piece that starts at character 1,204, and likewise for END scores and end
     characters. Position 0 ([CLS], "no answer") gets the teacher's [CLS] score.
  4. Student word pieces with no teacher piece starting (or ending) at the same character, such as
     "##ing" in the middle of a word, get a very low score: the teacher would never start an
     answer there either. So do the question and special tokens.

The teacher runs once, before training, and its scores are stored next to the training windows
(about 200 MB for all of SQuAD 2.0), so the two epochs don't pay for it twice.
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, Trainer

from src.config import Config
from src.preprocess import CONTEXT_SEQ_ID, prepare_train_features

# The score for "the teacher would never pick this word piece". Finite (unlike -infinity) so softmax
# and half-precision numbers stay well-behaved; e^-10000 is 0 for all practical purposes.
IMPOSSIBLE = -1e4
TEACHER_COLUMNS = ("teacher_start", "teacher_end")
CACHE_FOLDER = "teacher_scores"   # inside the training output folder, so a rerun skips the teacher
TIMING_FILE = "teacher_timing.json"


# ---------------------------------------------------------------------------------------------
# 1. Lining up the teacher's word pieces with the student's
# ---------------------------------------------------------------------------------------------

def align_teacher_scores(teacher_start, teacher_end, teacher_spans, student_spans):
    """Move the teacher's start/end scores onto the student's word pieces by character position.

    teacher_start, teacher_end: the teacher's scores, one per teacher word piece.
    teacher_spans, student_spans: (start_char, end_char) of each word piece in the passage, or None
    for a word piece that is not passage text (question, [CLS], [SEP], padding). Both use the same
    character positions (in the original passage).
    Index 0 must be [CLS] for both models: it carries the "no answer" score.

    Returns (start scores, end scores), one per STUDENT word piece.
    """
    by_start, by_end = {}, {}
    for k, span in enumerate(teacher_spans):
        if span is None or span[0] >= span[1]:   # not passage text, or an empty piece (a lone space)
            continue
        by_start[span[0]] = max(by_start.get(span[0], IMPOSSIBLE), float(teacher_start[k]))
        by_end[span[1]] = max(by_end.get(span[1], IMPOSSIBLE), float(teacher_end[k]))

    starts = [IMPOSSIBLE] * len(student_spans)
    ends = [IMPOSSIBLE] * len(student_spans)
    starts[0], ends[0] = float(teacher_start[0]), float(teacher_end[0])   # [CLS]: "no answer"
    for k, span in enumerate(student_spans):
        if span is None or k == 0:
            continue
        starts[k] = by_start.get(span[0], IMPOSSIBLE)
        ends[k] = by_end.get(span[1], IMPOSSIBLE)
    return starts, ends


def _passage_spans(offsets, sequence_ids, text: str, shift: int = 0):
    """Character spans of the passage word pieces (None elsewhere), shifted by `shift` characters.

    Like preprocess.trim_leading_spaces, a span that starts on a space is moved to the first letter,
    because DeBERTa-v3's tokenizer counts the space before a word as part of the word.
    """
    spans = []
    for (start, end), seq_id in zip(offsets, sequence_ids):
        if seq_id != CONTEXT_SEQ_ID:
            spans.append(None)
            continue
        while start < end and text[start].isspace():
            start += 1
        spans.append((start + shift, end + shift))
    return spans


# ---------------------------------------------------------------------------------------------
# 2. Running the teacher on the student's training windows
# ---------------------------------------------------------------------------------------------

def _model_inputs(model, inputs: dict) -> dict:
    """Only the inputs the model's forward() takes. DistilBERT, for example, has no token_type_ids,
    although some tokenizers (and the tests' BERT tokenizer) return them."""
    accepted = inspect.signature(model.forward).parameters
    return {k: v for k, v in inputs.items() if k in accepted}


@torch.no_grad()
def _teacher_scores(teacher, teacher_tokenizer, questions, passages, batch_size, max_length, device):
    """The teacher's start/end scores and passage spans for each (question, passage) pair.

    Pairs are run shortest first, so each batch is padded as little as possible.
    """
    order = sorted(range(len(questions)), key=lambda i: len(questions[i]) + len(passages[i]))
    results = [None] * len(questions)
    use_fp16 = device.type == "cuda"   # half precision is about twice as fast on a T4, and fine for scoring
    for first in range(0, len(order), batch_size):
        batch = order[first:first + batch_size]
        enc = teacher_tokenizer([questions[i] for i in batch], [passages[i] for i in batch],
                                truncation="only_second", max_length=max_length, padding=True,
                                return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        inputs = {k: v.to(device) for k, v in _model_inputs(teacher, enc).items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            out = teacher(**inputs)
        start, end = out.start_logits.float().cpu().tolist(), out.end_logits.float().cpu().tolist()
        for row, i in enumerate(batch):
            spans = _passage_spans(offsets[row], enc.sequence_ids(row), passages[i])
            results[i] = (start[row], end[row], spans)
    return results


def add_teacher_scores(examples, tokenizer, teacher, teacher_tokenizer, cfg: Config, device):
    """Batched function for `Dataset.map`: training windows (as in src/train.py) + the teacher's scores.

    The student's windows and labels are exactly the ones normal training uses; two columns are added,
    `teacher_start` and `teacher_end`, with one teacher score per student word piece.
    """
    pre, dist = cfg.preprocessing, cfg.distillation
    enc = prepare_train_features(examples, tokenizer, pre.max_length, pre.doc_stride, keep_mapping=True, pad=False)
    sample_map = enc.pop("overflow_to_sample_mapping")
    offset_mapping = enc.pop("offset_mapping")

    questions, passages, shifts, student_spans = [], [], [], []
    for i, offsets in enumerate(offset_mapping):
        context = examples["context"][sample_map[i]]
        spans = _passage_spans(offsets, enc.sequence_ids(i), context)
        inside = [s for s in spans if s is not None]
        low, high = inside[0][0], inside[-1][1]   # the stretch of passage this window covers
        questions.append(examples["question"][sample_map[i]].strip())
        passages.append(context[low:high])
        shifts.append(low)
        student_spans.append(spans)

    scored = _teacher_scores(teacher, teacher_tokenizer, questions, passages,
                             dist.teacher_batch_size, dist.teacher_max_length, device)
    enc["teacher_start"], enc["teacher_end"] = [], []
    for i, (t_start, t_end, t_spans) in enumerate(scored):
        t_spans = [None if s is None else (s[0] + shifts[i], s[1] + shifts[i]) for s in t_spans]
        starts, ends = align_teacher_scores(t_start, t_end, t_spans, student_spans[i])
        # 32-bit numbers: half the disk space of Python's 64-bit floats, and plenty precise for scores.
        enc["teacher_start"].append(np.asarray(starts, dtype=np.float32))
        enc["teacher_end"].append(np.asarray(ends, dtype=np.float32))
    return enc


def teacher_agreement(features, limit: int = 2000) -> dict:
    """How often the teacher's top start/end word piece (after lining up) is the SQuAD label.

    A check that the lining up worked: if it were broken, the teacher would almost never agree.
    The teacher was trained on these questions, so we expect it to agree most of the time.
    Looks at the first `limit` windows (a Python loop over all 131,000 would take minutes).
    """
    features = features.select(range(min(limit, len(features)))) if hasattr(features, "select") else features[:limit]
    starts = ends = both = 0
    for row in features:
        s = max(range(len(row["teacher_start"])), key=row["teacher_start"].__getitem__)
        e = max(range(len(row["teacher_end"])), key=row["teacher_end"].__getitem__)
        starts += s == row["start_positions"]
        ends += e == row["end_positions"]
        both += s == row["start_positions"] and e == row["end_positions"]
    n = max(1, len(features))
    return {"windows": len(features), "start_pct": round(100 * starts / n, 1),
            "end_pct": round(100 * ends / n, 1), "both_pct": round(100 * both / n, 1)}


def make_distill_features(raw_train, tokenizer, cfg: Config, device=None, teacher=None, teacher_tokenizer=None):
    """Training windows with the teacher's scores. Saved in <output_dir>/teacher_scores and reused.

    The teacher model is loaded here (unless passed in, as the tests do) and freed afterwards, so the
    GPU memory is all the student's during training.
    """
    from datasets import load_from_disk

    from src.evaluate import pick_device

    # The question count is in the name, so a trial run's scores are never reused for the full run.
    cache = Path(cfg.training.output_dir) / f"{CACHE_FOLDER}_{len(raw_train)}"
    if (cache / "dataset_info.json").exists():
        print(f"Using the teacher scores saved in {cache}")
        return load_from_disk(str(cache))

    device = device or pick_device()
    if teacher is None:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.distillation.teacher)
        teacher = AutoModelForQuestionAnswering.from_pretrained(cfg.distillation.teacher)
    teacher.to(device).eval()

    started = time.time()
    features = raw_train.map(
        add_teacher_scores,
        batched=True,
        batch_size=256,
        remove_columns=raw_train.column_names,
        fn_kwargs={"tokenizer": tokenizer, "teacher": teacher, "teacher_tokenizer": teacher_tokenizer,
                   "cfg": cfg, "device": device},
        new_fingerprint=f"teacher-scores-{len(raw_train)}",   # don't try to hash the teacher model
        desc="Teacher scores",
    )
    seconds = time.time() - started
    print(f"Teacher scored {len(features):,} windows in {seconds:.0f}s ({len(features) / seconds:.1f} windows/s)")

    features.save_to_disk(str(cache))
    # Kept for the time estimate in notebook 06 (the training summary only times the training itself).
    timing = {"windows": len(features), "seconds": round(seconds, 1), "device": str(device)}
    (cache / TIMING_FILE).write_text(json.dumps(timing) + "\n", encoding="utf-8")
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features


# ---------------------------------------------------------------------------------------------
# 3. Training on both the labels and the teacher's scores
# ---------------------------------------------------------------------------------------------

def distillation_loss(student_start, student_end, teacher_start, teacher_end, start_positions, end_positions,
                      temperature: float, alpha: float):
    """alpha x (difference from the teacher) + (1 - alpha) x (difference from the SQuAD labels).

    Returns (total, label part, teacher part).

    * Label part: the usual QA loss (cross-entropy on the start and end positions), as in normal training.
    * Teacher part: KL divergence between the teacher's and the student's probabilities over word
      pieces, both softened by `temperature`. It is 0 when the student scores exactly like the
      teacher. Multiplying by temperature^2 keeps its size comparable to the label part (Hinton et
      al., 2015, "Distilling the Knowledge in a Neural Network").
    """
    student_start, student_end = student_start.float(), student_end.float()
    hard = (F.cross_entropy(student_start, start_positions) + F.cross_entropy(student_end, end_positions)) / 2

    def soft(student, teacher):
        return F.kl_div(F.log_softmax(student / temperature, dim=-1), F.log_softmax(teacher.float() / temperature, dim=-1),
                        log_target=True, reduction="batchmean") * temperature ** 2

    teacher_part = (soft(student_start, teacher_start) + soft(student_end, teacher_end)) / 2
    return alpha * teacher_part + (1 - alpha) * hard, hard, teacher_part


class DistillationCollator:
    """Pads a batch like DataCollatorWithPadding, and pads the teacher's scores to the same length."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.pad = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    def __call__(self, features):
        batch = self.pad([{k: v for k, v in f.items() if k not in TEACHER_COLUMNS} for f in features])
        length = batch["input_ids"].shape[1]
        for column in TEACHER_COLUMNS:
            # Padding is not text: the teacher would never pick it.
            batch[column] = torch.tensor([list(f[column]) + [IMPOSSIBLE] * (length - len(f[column])) for f in features])
        return batch


class DistillationTrainer(Trainer):
    """The Hugging Face Trainer with `distillation_loss` in place of the model's own loss."""

    def __init__(self, *args, temperature: float, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature, self.alpha = temperature, alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_start, teacher_end = inputs.pop("teacher_start"), inputs.pop("teacher_end")
        start_positions, end_positions = inputs.pop("start_positions"), inputs.pop("end_positions")
        outputs = model(**_model_inputs(model, inputs))
        loss, _, _ = distillation_loss(outputs.start_logits, outputs.end_logits, teacher_start, teacher_end,
                                       start_positions, end_positions, self.temperature, self.alpha)
        return (loss, outputs) if return_outputs else loss
