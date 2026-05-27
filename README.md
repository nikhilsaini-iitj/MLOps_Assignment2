# MLOps Assignment 2: Goodreads Genre Classification with DistilBERT

This project fine-tunes **DistilBERT** (`distilbert-base-cased`) on **UCSD Goodreads reviews** to classify books by **genre**. It follows the professor's reference notebook with added MLOps integrations: **Weights & Biases** experiment tracking, **Kaggle GPU** training, and **Hugging Face Hub** deployment.

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) |
| **Genres** | poetry, children, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, young adult |
| **Features** | Review text |
| **Size** | 8 genres × 2,000 reviews = ~16,000 reviews |
| **Preprocessing** | Truncated/padded to 512 tokens |
| **Split** | 800 train / 200 test per genre |

---

## Setup Instructions

1. Import `kaggle_notebook.ipynb` into Kaggle.
2. Enable GPU: Settings → Accelerator → GPU T4 x2.
3. Enable Internet: Settings → Internet ON.
4. Add Kaggle Secrets: `WANDB_API_KEY` and `HF_TOKEN`.
5. Run all cells.

---

## Training Platform

- **Platform:** Kaggle Notebooks
- **Accelerator:** GPU T4 x2
- **Framework:** Hugging Face Transformers + PyTorch
- **Tracking:** Weights & Biases

---

## Results (Update After Training)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.5744 |
| F1 Score  | 0.5736 |
| Eval Loss | 2.4490 |

- **Hugging Face model:** `https://huggingface.co/Nikhil-iitj/distilbert-goodreads-genres`
- **W&B dashboard:** `https://wandb.ai/g25ait2067-prom-iit-rajasthan/mlops-assignment2`
- **Kaggle Notebook:** `https://www.kaggle.com/code/nikhilg25ait2067/ml-ops`


---

## Project Structure

```
├── kaggle_notebook.ipynb   # Kaggle notebook (main deliverable)
├── data.py                 # Data loading & preprocessing
├── train.py                # Training script
├── eval.py                 # Evaluation & artifact logging
├── utils.py                # Dataset class & metrics
├── push_to_hf.py           # Hugging Face deployment
├── README.md               # Documentation
├── report.md               # Academic report
├── requirements.txt        # Dependencies
```

---

## Inference Demo

The notebook includes a live inference cell that loads the deployed model from Hugging Face Hub and predicts the genre of a user-entered book review.

**Sample predictions:**
- *"A magical world full of wonder and delight..."* → **fantasy & paranormal**
- *"The serial killer thriller kept me on the edge of my seat..."* → **mystery, thriller & crime**


---

## Notes

- The dataset is balanced across genres, so accuracy and weighted F1 are both meaningful.
- `distilbert-base-cased` is used as specified in the professor's reference.
- Target for this assignment is workflow mastery (tracking, deployment, reproducibility), not state-of-the-art accuracy.
