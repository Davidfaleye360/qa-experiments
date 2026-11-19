---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: question-answering
base_model: distilbert-base-uncased
datasets:
  - rajpurkar/squad_v2
metrics:
  - exact_match
  - f1
tags:
  - question-answering
  - extractive-qa
  - squad_v2
  - distilbert
  - knowledge-distillation
model-index:
  - name: distilbert-squad2-distilled
    results:
      - task:
          type: question-answering
          name: Extractive question answering
        dataset:
          name: SQuAD 2.0
          type: rajpurkar/squad_v2
          split: validation
        metrics:
          - type: exact_match
            value: 66.64
            name: Exact Match
          - type: f1
            value: 69.76
            name: F1
---

# DistilBERT distilled from DeBERTa-v3 on SQuAD 2.0

`distilbert-base-uncased` fine-tuned for **extractive question answering** on SQuAD 2.0, learning from a
stronger "teacher" model as well as from the correct answers. Given a passage and a question, it marks the
words in the passage that answer the question, or says that the passage does not contain an answer.

A personal experiment by David Faleye, continuing a CSCI 390 course project. It is the course project's
model, [Ingsyling/distilbert-squad2](https://huggingface.co/Ingsyling/distilbert-squad2), trained again with
the same settings, except that half of what it learns from is the scores of
[Ingsyling/deberta-v3-squad2](https://huggingface.co/Ingsyling/deberta-v3-squad2) (87.7 F1). The goal was a more
accurate model at the same size and speed. Code, configuration and training notebook:
https://github.com/Davidfaleye360/qa-experiments

## How to use it

```python
from transformers import pipeline

qa = pipeline("question-answering", model="Ingsyling/distilbert-squad2-distilled")
result = qa(
    question="In what country is Normandy located?",
    context="The Normans gave their name to Normandy, a region in France.",
    handle_impossible_answer=True,   # allow "no answer"; the answer is then an empty string
)
print(result["answer"])   # "France"
```

This needs `transformers` 4.x (version 5 removed the `question-answering` pipeline).

## Results

SQuAD 2.0 dev set, 11,873 questions, at the default no-answer threshold. The project's scoring code and the
official `squad_v2` metric give the same numbers, on a Colab T4 and on an Apple M2 laptop.

| | Exact Match | F1 | Questions |
| --- | --- | --- | --- |
| All | 66.64 | 69.76 | 11,873 |
| Answerable | 66.87 | 73.12 | 5,928 |
| Unanswerable | 66.41 | 66.41 | 5,945 |

For comparison, measured with the same code:

| Model | Exact Match | F1 | Answerable F1 | Unanswerable F1 | Parameters |
| --- | --- | --- | --- | --- | --- |
| This model | 66.64 | 69.76 | 73.12 | 66.41 | 66 M |
| [Ingsyling/distilbert-squad2](https://huggingface.co/Ingsyling/distilbert-squad2) (same model, no teacher) | 64.14 | 67.38 | 72.17 | 62.59 | 66 M |
| `deepset/roberta-base-squad2` | 79.90 | 82.91 | 84.01 | 81.82 | 124 M |
| [Ingsyling/deberta-v3-squad2](https://huggingface.co/Ingsyling/deberta-v3-squad2) (the teacher) | 84.86 | 87.69 | 89.02 | 86.36 | 184 M |

Learning from the teacher adds 2.5 Exact Match and 2.4 F1 points to the same model trained without it. Most
of the gain is on unanswerable questions (+3.8), against +1.0 F1 on answerable ones: the teacher's score for
"no answer" shows the student how sure to be. It closes 15% of the gap to RoBERTa (15.5 → 13.2 F1 points).

## Training

| Setting | Value |
| --- | --- |
| Data | SQuAD 2.0 training set, 130,319 questions (131,754 windows) |
| Window | 384 tokens, 128 tokens of overlap |
| Epochs | 2 (16,470 steps) |
| Batch size | 16 |
| Learning rate | 3e-5, linear warm-up over the first 10% of steps, then linear decay |
| Weight decay | 0.01 |
| Precision | mixed (fp16) |
| Seed | 42 |
| Teacher | [Ingsyling/deberta-v3-squad2](https://huggingface.co/Ingsyling/deberta-v3-squad2) |
| Loss | 0.5 × difference from the teacher (KL divergence, temperature 2) + 0.5 × difference from the correct answer (cross-entropy) |
| Hardware and time | one Tesla T4 (Google Colab): 17 minutes for the teacher's scores, then 38 minutes of training, 2.4 GB GPU memory at most |

**How the teacher's scores reach the student.** The teacher gives every word piece a score for "the answer
starts here" and "the answer ends here". DeBERTa-v3 and DistilBERT cut text into different word pieces, so the
scores are matched by character position: the teacher reads exactly the stretch of passage each student
window covers, and a student word piece that starts (or ends) at the same character as a teacher word piece
gets that piece's start (or end) score. Student word pieces with no match, such as the middle of a word, get
a very low score. The `[CLS]` position carries the "no answer" score for both. The teacher's top choice after
this matching agreed with the training label in 78.5% of 2,000 checked windows; most disagreements were the
teacher picking a slightly longer or shorter span. The teacher runs once, before training, and its scores are
reused in both epochs.

All settings are in `configs/distill.yaml` and the method in `src/distill.py` in the repository. The training
loss (teacher part plus label part) fell from 10.9 to about 1.2; it is not comparable with the loss of a model
trained on the labels alone.

## Limitations

- **It copies, it does not reason.** The answer is always one continuous piece of the passage.
- **Still far behind its teacher and RoBERTa:** 18 F1 points below the teacher and 13 below RoBERTa. Distillation
  narrows the gap a little; it does not make a 66 M parameter model as good as a 184 M one.
- **The default threshold is too eager.** Requiring the best answer to beat "no answer" by about 1.95 (a
  no-answer threshold of −1.95) raises F1 from 69.76 to 70.43, chosen and scored on different halves of the dev
  set. The pipeline above does not apply a threshold.
- **The teacher was trained on the same questions.** Its scores on the training set are more confident than on
  new text, so they carry less extra information than a teacher's scores usually do.
- **Not tested beyond the dev set:** the error analysis, long-document test and speed measurement done for the
  other models were not repeated. The network is the same as the course project's DistilBERT (about 27 ms per
  question on an Apple M2 laptop CPU), so its speed should be the same.
- **English Wikipedia only.** It was trained on Wikipedia paragraphs and has not been tested on other kinds of
  text or other languages.
- Do not use it where a wrong answer has consequences (medical, legal, financial) without a person checking.

## Licences

The base model `distilbert-base-uncased` is released under the Apache 2.0 licence, and the teacher's base model
`microsoft/deberta-v3-base` under the MIT licence. SQuAD 2.0 is released under CC BY-SA 4.0.
