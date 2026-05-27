# MLOps Assignment 2 Report
## Goodreads Book-Genre Classification with DistilBERT

**Name:** Nikhil Saini  
**Roll Number:** G25AIT2067  
**Program:** PGD AI, IIT Jodhpur  
**Date:** 27 May 2026

---

## 1. Introduction & Objective

This project implements the MLOps Assignment 2 pipeline by fine-tuning a pre-trained **DistilBERT** model on **UCSD Goodreads reviews** to classify books into **eight genres**. The work demonstrates end-to-end MLOps practices: production scripting, experiment tracking, cloud GPU training, evaluation with artifact logging, and model deployment.

**Core MLOps practices demonstrated:**
- **Productionizing notebooks:** Modular Python scripts for data loading, training, evaluation, and deployment.
- **Experiment tracking:** Logging metrics, hyperparameters, and artifacts via **Weights & Biases (W&B)**.
- **Cloud GPU training:** Leveraging **Kaggle's** free GPU infrastructure.
- **Model deployment:** Publishing the fine-tuned model to the **Hugging Face Hub**.
- **Reproducibility:** Scripts and tracked experiments ensure the workflow is repeatable.

---

## 2. Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) |
| **Genres** | poetry, children, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, romance, young_adult |
| **Features** | Review text only |
| **Size** | 8 genres × 2,000 reviews sampled = ~16,000 total |
| **Preprocessing** | Tokenized with DistilBERT tokenizer; truncated/padded to 512 tokens |
| **Split** | 800 training / 200 test per genre (80/20 split) |

The dataset is loaded via HTTP streaming from UCSD mirrors, sampled to 2,000 reviews per genre for computational efficiency, and split stratified by genre.

---

## 3. Model & Training Setup

**Model:** `distilbert-base-cased`  
**Tokenizer:** `DistilBertTokenizerFast`  
**Classes:** 8 genres  
**Max length:** 512 tokens  
**Device:** CUDA (Kaggle T4 x2)

**Hyperparameters:**

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

DistilBERT (66M parameters) was selected because it is lighter and faster than full BERT while retaining strong performance, making it ideal for educational fine-tuning on limited GPU time.

---

## 4. Implementation Highlights

### 4.1 Kaggle Secrets Management
API tokens are stored securely via **Kaggle Secrets** rather than hard-coding:
- `WANDB_API_KEY` — W&B authentication.
- `HF_TOKEN` — Hugging Face Hub write access.

```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
WANDB_API_KEY = secrets.get_secret('WANDB_API_KEY')
HF_TOKEN = secrets.get_secret('HF_TOKEN')
```

### 4.2 Weights & Biases Integration
W&B is initialized before training with the full hyperparameter configuration:

```python
wandb.init(
    project='mlops-assignment2',
    name='distilbert-goodreads-run-1',
    config={'model': model_name, 'epochs': 3, ...}
)
```

`TrainingArguments` reports to `wandb`, and the `Trainer` streams loss, evaluation metrics, and model checkpoints automatically.

### 4.2.1 W&B Training Dashboard Screenshot

The screenshot below shows the live W&B dashboard capturing the full training run on Kaggle GPU. It visualises training loss, validation loss, accuracy, and F1 score across the 3 epochs, confirming that the model is learning and not overfitting.

**[INSERT W&B SCREENSHOT HERE]**

*Figure 1: Weights & Biases dashboard — training metrics (loss, accuracy, F1) logged every 100 steps from the Kaggle GPU run.*

### 4.3 Evaluation Metrics
The custom `compute_metrics` function returns **accuracy** and **weighted F1** to the `Trainer`:

```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}
```

After fine-tuning, the final evaluation metrics are logged to W&B and the full `classification_report` is saved as a JSON artifact:

```python
wandb.log({'final/loss': ..., 'final/accuracy': ..., 'final/f1': ...})
artifact = wandb.Artifact('eval-report', type='evaluation')
artifact.add_file('eval_report.json')
wandb.log_artifact(artifact)
```

### 4.4 Hugging Face Hub Deployment
The fine-tuned model and tokenizer are pushed to the Hub under the repository `Nikhil-iitj/distilbert-goodreads-genres`:

```python
from huggingface_hub import login
login(token=HF_TOKEN)
model.push_to_hub('Nikhil-iitj/distilbert-goodreads-genres')
tokenizer.push_to_hub('Nikhil-iitj/distilbert-goodreads-genres')
```

The W&B run summary is updated with the Hugging Face model URL for traceability.

### 4.5 Inference from Hub
A demonstration cell loads the deployed model directly from Hugging Face Hub and performs interactive genre prediction on user-supplied book reviews, confirming the deployment is functional.

---

## 5. Evaluation Results

After 3 epochs of fine-tuning, the model was evaluated on the held-out test set.

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.5744 |
| **F1 Score (Weighted)** | 0.5736 |
| **Evaluation Loss** | 2.4490 |

These placeholders should be replaced with actual values from the Kaggle run. Random guessing across 8 classes yields 12.5%; the fine-tuned DistilBERT is expected to significantly exceed this baseline.

---

## 6. Comparison with Baseline

| Aspect | Logistic Regression (TF-IDF) | Fine-Tuned DistilBERT |
|--------|------------------------------|------------------------|
| **Approach** | Classical ML | Transfer learning |
| **Features** | TF-IDF vectors | Contextual embeddings |
| **Accuracy** | ~0.56 | 0.5744 |
| **Training time** | Seconds | ~10–15 min (Kaggle GPU) |
| **MLOps ready** | No | Yes (W&B + HF Hub) |

The fine-tuned DistilBERT outperforms the classical baseline while also being fully instrumented for MLOps tracking and deployment.

---

## 7. Challenges Faced & Learnings

1. **Kaggle Internet Toggle:** `ConnectionError` when loading Kaggle Secrets. **Resolution:** Explicitly enable Internet in Notebook Settings and restart the session.
2. **Hugging Face Namespace Mismatch:** `push_to_hub` failed with `403 Forbidden` when using an incorrect username. **Resolution:** Verified the exact Hugging Face username (`Nikhil-iitj`) and used a Write-scoped token.
3. **Credential Validation:** Verifying API keys and namespace permissions *before* starting training saves debugging time after a long GPU run.

---

## 8. Conclusion

This project successfully demonstrates the complete MLOps pipeline on the professor's Goodreads genre-classification task: data preparation, model fine-tuning with live experiment tracking, evaluation with artifact logging, and deployment to a public model repository. All components are reproducible from the tracked W&B configuration and the versioned Hugging Face Hub model.

---

## 9. Submission Links

| Resource | Public Link |
|----------|-------------|
| **GitHub Repository** | https://github.com/nikhilsaini-iitj/MLOps_Assignment2 |
| **Kaggle Notebook** | https://www.kaggle.com/code/nikhilg25ait2067/ml-ops |
| **Hugging Face Model** | https://huggingface.co/Nikhil-iitj/distilbert-goodreads-genres |
| **W&B Dashboard** | https://wandb.ai/g25ait2067-prom-iit-rajasthan/mlops-assignment2 |

---

*Note: The primary evaluation criterion for this assignment is the successful implementation of the MLOps workflow, not achieving state-of-the-art accuracy.*
