import os
from huggingface_hub import login, HfApi

# Use HF_TOKEN from Kaggle Secrets (already loaded in env by the notebook)
login(token=os.environ.get("HF_TOKEN"))

repo_id = "Nikhil-iitj/distilbert-goodreads-genres"

# Model card content (Markdown format)
model_card = """---
language: en
license: mit
library_name: transformers
tags:
- text-classification
- distilbert
- goodreads
- books
- genres
datasets:
- ucsd_goodreads
metrics:
- accuracy
- f1
model-index:
- name: distilbert-goodreads-genres
  results:
  - task: text-classification
    dataset:
      name: UCSD Goodreads Book Graph
      type: reviews_by_genre
    metrics:
      - type: accuracy
        value: 0.5744
      - type: f1_weighted
        value: 0.5736
---

# DistilBERT Goodreads Genre Classification

Fine-tuned DistilBERT model for classifying Goodreads book reviews into 8 genres.

## Model Description

This model is a fine-tuned version of [distilbert-base-cased](https://huggingface.co/distilbert-base-cased) on the UCSD Goodreads Book Graph review dataset. It classifies book reviews into one of eight genres using the Hugging Face Transformers Trainer API.

## Intended Uses

- **Direct use:** Predict the genre of a book given a user-written review text.
- **Downstream use:** Content tagging, recommendation systems, book cataloguing pipelines.
- **Out-of-scope:** Any use beyond educational/text-classification purposes; not suitable for high-stakes decisions without human review.

## Training Details

- **Base model:** `distilbert-base-cased`
- **Dataset:** UCSD Goodreads Book Graph (8 genres, 800 train / 200 test per genre)
- **Max length:** 512 tokens
- **Epochs:** 3
- **Batch size:** 10 (train), 16 (eval)
- **Learning rate:** 5e-5
- **Warmup steps:** 100
- **Weight decay:** 0.01
- **Optimizer:** AdamW
- **Platform:** Kaggle GPU T4 x2

## Evaluation Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.5744 |
| F1 (weighted) | 0.5736 |
| Eval Loss | 2.4490 |

*Evaluated on a held-out test set of 1,600 reviews (200 per genre).*

## How to Use

### With Transformers pipeline

```python
from transformers import pipeline
pipe = pipeline("text-classification", model="Nikhil-iitj/distilbert-goodreads-genres")
result = pipe("A magical world full of wonder and delight...")
```

### With Transformers AutoModel

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("Nikhil-iitj/distilbert-goodreads-genres")
model = AutoModelForSequenceClassification.from_pretrained("Nikhil-iitj/distilbert-goodreads-genres")
```

## Limitations and Bias

- The dataset is sampled and may not represent all books globally.
- Genres overlap (e.g., fantasy vs. young adult), so misclassifications occur.
- Review length is truncated to 512 tokens; longer reviews lose tail content.
- Not intended for critical applications without human-in-the-loop validation.

## Citation

```bibtex
@misc{distilbert_goodreads_genres,
  title={DistilBERT Fine-Tuned for Goodreads Genre Classification},
  author={Nikhil Saini},
  year={2026},
  publisher={Hugging Face},
  howpublished={\\url{https://huggingface.co/Nikhil-iitj/distilbert-goodreads-genres}}
}
```
"""

# Upload README.md to the model repo (this is the model card on HF)
api = HfApi()
api.upload_file(
    path_or_fileobj=model_card.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model card uploaded to https://huggingface.co/{repo_id}")
