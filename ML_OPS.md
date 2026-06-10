# MLOps Assignment 2 — Consolidated Reference
## Goodreads Book-Genre Classification with DistilBERT

---

## Student Details

| Field | Value |
|-------|-------|
| Name | Nikhil Saini |
| Roll Number | G25AIT2067 |
| Program | PGD AI, IIT Jodhpur |
| Date | 27 May 2026 |

---

## Submission Links (Required)

| Resource | Public Link |
|----------|-------------|
| **GitHub Repository** | https://github.com/nikhilsaini-iitj/MLOps_Assignment2 |
| **Kaggle Notebook** | https://www.kaggle.com/code/nikhilg25ait2067/ml-ops |
| **Hugging Face Model** | https://huggingface.co/Nikhil-iitj/distilbert-goodreads-genres |
| **W&B Dashboard** | https://wandb.ai/g25ait2067-prom-iit-rajasthan/mlops-assignment2 |

---

## Files in Repository

| File | Purpose |
|------|---------|
| `kaggle_notebook.ipynb` | Main Kaggle notebook — training + deployment |
| `data.py` | Data loading, sampling, train/test split |
| `train.py` | Model loading, Trainer setup, training loop |
| `eval.py` | Evaluation, metrics, artifact logging |
| `utils.py` | Dataset class, label maps, compute_metrics |
| `push_to_hf.py` | Hugging Face Hub deployment |
| `requirements.txt` | Dependencies |
| `README.md` | Project documentation |
| `report.md` | Academic report (Markdown source) |
| `G25AIT2067_MLOps_Assignment2_Report.docx` | Final report with W&B screenshot |

---

## Model & Dataset Details

| Attribute | Value |
|-----------|-------|
| **Model** | `distilbert-base-cased` |
| **Tokenizer** | `DistilBertTokenizerFast` |
| **Parameters** | 66M |
| **Classes** | 8 genres |
| **Max length** | 512 tokens |
| **Dataset** | UCSD Goodreads Book Graph |
| **Genres** | poetry, children, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, romance, young_adult |
| **Total samples** | ~16,000 (2,000 per genre) |
| **Split** | 800 train / 200 test per genre |

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size (train) | 10 |
| Batch size (eval) | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Evaluation strategy | steps |
| Logging | every 100 steps |

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.5744 |
| **F1 Score (Weighted)** | 0.5736 |
| **Evaluation Loss** | 2.4490 |

---

## Kaggle Secrets (Add via Add-ons → Secrets)

| Secret Name | Source |
|-------------|--------|
| `WANDB_API_KEY` | https://wandb.ai/authorize |
| `HF_TOKEN` | https://huggingface.co/settings/tokens (Write access) |

---

## Key Code Snippets

### Kaggle Secrets Cell
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
WANDB_API_KEY = secrets.get_secret('WANDB_API_KEY')
HF_TOKEN = secrets.get_secret('HF_TOKEN').strip()  # .strip() is critical
import os
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ['HF_TOKEN'] = HF_TOKEN
```

### W&B Init
```python
import wandb
wandb.init(
    project='mlops-assignment2',
    name='distilbert-goodreads-run-1',
    config={'model': model_name, 'epochs': 3, 'batch_size': 10, 'learning_rate': 5e-5}
)
```

### compute_metrics (Working Version)
```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}
```

### Hugging Face Push
```python
from huggingface_hub import login
login(token=HF_TOKEN)
model.push_to_hub('Nikhil-iitj/distilbert-goodreads-genres')
tokenizer.push_to_hub('Nikhil-iitj/distilbert-goodreads-genres')
wandb.run.summary['huggingface_model'] = 'https://huggingface.co/Nikhil-iitj/distilbert-goodreads-genres'
```

---

## Troubleshooting Log

| Issue | Cause | Fix |
|-------|-------|-----|
| `ConnectionError` loading Secrets | Internet OFF in Kaggle | Settings → Internet toggle ON → Restart session |
| `LocalProtocolError: Illegal header value b'Bearer '` | HF token had trailing whitespace | Add `.strip()` to `secrets.get_secret('HF_TOKEN')` |
| `403 Forbidden` on `push_to_hub` | Wrong HF username or read-only token | Verify exact username `Nikhil-iitj`; use Write-scoped token |
| `NameError: name 'wandb' not defined` | wandb not imported in cell | Add `import wandb` inside the same cell |
| `AttributeError: float object has no attribute 'mean'` | Old broken `precision_recall_fscore_support(...)[2].mean()` code | Replace with `f1_score(..., average='weighted')` |
| `@claude` appears in GitHub Contributors | `Co-Authored-By:` line in commit message | Amend commit to remove trailer; force-push |

---

## Mark Summary (Rubric)

| Task | Marks |
|------|-------|
| Load Pre-trained Model from HF | 10 |
| Train on Kaggle & Track with W&B | 25 |
| Evaluate & Save Results | 15 |
| Push Model to HF Hub | 10 |
| **TOTAL** | **60** |

---

## Submission Checklist

- [x] GitHub repo is public
- [x] HF model is public
- [x] W&B project is public
- [x] Kaggle notebook is public
- [x] W&B screenshot embedded in report
- [x] All 4 links included in report
- [ ] Convert docx to PDF
- [ ] Upload PDF to portal before deadline (27 May 2026, 11:59 PM)

---

*Note: The primary evaluation criterion is the successful implementation of the MLOps workflow, not achieving state-of-the-art accuracy.*
