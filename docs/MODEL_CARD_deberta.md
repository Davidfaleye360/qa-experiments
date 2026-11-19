---
language: en
license: mit
library_name: transformers
pipeline_tag: question-answering
base_model: microsoft/deberta-v3-base
datasets:
  - rajpurkar/squad_v2
metrics:
  - exact_match
  - f1
tags:
  - question-answering
  - extractive-qa
  - squad_v2
  - deberta-v3
model-index:
  - name: deberta-v3-squad2
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
            value: 84.86
            name: Exact Match
          - type: f1
            value: 87.69
            name: F1
---

# DeBERTa-v3-base fine-tuned on SQuAD 2.0

`microsoft/deberta-v3-base` fine-tuned for **extractive question answering** on SQuAD 2.0. Given a passage
and a question, it marks the words in the passage that answer the question, or says that the passage does
not contain an answer.

A personal experiment by David Faleye, continuing a CSCI 390 course project
([Ingsyling/distilbert-squad2](https://huggingface.co/Ingsyling/distilbert-squad2)). The goal was to beat
`deepset/roberta-base-squad2` with a model trained on a free GPU. Code, configuration and training notebook:
https://github.com/Davidfaleye360/qa-experiments

## How to use it

```python
from transformers import pipeline

qa = pipeline("question-answering", model="Ingsyling/deberta-v3-squad2")
result = qa(
    question="In what country is Normandy located?",
    context="The Normans gave their name to Normandy, a region in France.",
    handle_impossible_answer=True,   # allow "no answer"; the answer is then an empty string
)
print(result["answer"])   # " France." (see the note below)
```

This needs `transformers` 4.x (version 5 removed the `question-answering` pipeline) and the `sentencepiece`
and `protobuf` packages for the DeBERTa-v3 tokenizer.

The pipeline can return the answer with a leading space and a neighbouring punctuation mark (`" France."`),
because DeBERTa-v3's tokenizer counts the space before a word as part of that word. The scores below were
measured with the project's own code (`src/predict.py` and `src/evaluate.py` in the repository), which
trims that space and returns `"France"` for this example.

## Results

SQuAD 2.0 dev set, 11,873 questions, at the default no-answer threshold. The project's scoring code and the
official `squad_v2` metric give the same numbers, on a Colab T4 and on an Apple M2 laptop.

| | Exact Match | F1 | Questions |
| --- | --- | --- | --- |
| All | 84.86 | 87.69 | 11,873 |
| Answerable | 83.37 | 89.02 | 5,928 |
| Unanswerable | 86.36 | 86.36 | 5,945 |

For comparison, measured with the same code:

| Model | Exact Match | F1 | Answerable F1 | Unanswerable F1 | Parameters |
| --- | --- | --- | --- | --- | --- |
| This model | **84.86** | **87.69** | **89.02** | **86.36** | 184 M |
| `deepset/roberta-base-squad2` | 79.90 | 82.91 | 84.01 | 81.82 | 124 M |
| [Ingsyling/distilbert-squad2](https://huggingface.co/Ingsyling/distilbert-squad2) | 64.14 | 67.38 | 72.17 | 62.59 | 66 M |

This model is 4.8 F1 points ahead of RoBERTa, on both answerable and unanswerable questions. The DeBERTa-v3
authors report about 88 F1 for the base model on SQuAD 2.0.

## Training

| Setting | Value |
| --- | --- |
| Data | SQuAD 2.0 training set, 130,319 questions (131,467 windows) |
| Window | 384 tokens, 128 tokens of overlap |
| Epochs | 2 (16,434 steps) |
| Batch size | 16 (8 per step, gradients accumulated over 2 steps) |
| Learning rate | 3e-5, linear warm-up over the first 10% of steps, then linear decay |
| Weight decay | 0.01 |
| Precision | mixed (fp16) |
| Seed | 42 |
| Hardware and time | one Tesla T4 (Google Colab), 1 hour 39 minutes, 7.0 GB GPU memory at most |

Training batches are padded only to their longest window, and windows of similar length are grouped
together, so most steps process far fewer than 384 tokens per window (the average window is about 164
tokens). Unanswerable
questions, and windows that do not contain the whole answer, are labelled with the `[CLS]` position. The
training loss fell from 5.22 to about 0.5. All settings are in `configs/deberta.yaml` in the repository.

## Limitations

- **It copies, it does not reason.** The answer is always one continuous piece of the passage. Questions
  whose answer has to be put together from several sentences are beyond it.
- **Unanswerable questions are still the main weakness.** 45% of its 1,797 errors on the dev set are answers
  to unanswerable questions (811, against 1,081 for RoBERTa), and "why" questions are the hardest kind
  (76 F1 on answerable "why" questions, against 94 for "when").
- **The default threshold is slightly too eager.** Requiring the best answer to beat "no answer" by about 1.9
  (a no-answer threshold of −1.9) raises F1 from 87.69 to 88.33, chosen and scored on different halves of the
  dev set. The pipeline above does not apply a threshold.
- **Long documents need care.** It reads at most 384 tokens at a time; longer texts are split into
  overlapping windows. On three whole dev articles (12 to 26 windows per question), F1 fell from 91.6 / 85.8 /
  80.9 with the right paragraph to 86.4 / 73.5 / 73.5 when picking the window whose answer beats its own
  "no answer" score by the most, still above our DistilBERT given the right paragraph (70.6 / 60.1 / 59.3).
  The no-answer threshold needs re-tuning for long texts.
- **English Wikipedia only.** It was trained on Wikipedia paragraphs and has not been tested on other kinds
  of text or other languages.
- **Bigger and slower than the alternatives above:** 735 MB, and a median of 85 ms per question on an Apple
  M2 laptop CPU, against 53 ms for `deepset/roberta-base-squad2` and 27 ms for the DistilBERT model.
- Do not use it where a wrong answer has consequences (medical, legal, financial) without a person checking.

## Licences

The base model `microsoft/deberta-v3-base` is released under the MIT licence. SQuAD 2.0 is released under
CC BY-SA 4.0.
