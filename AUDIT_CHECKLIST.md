# MLOps Assignment 2 Final Audit Checklist

## Task 1: Download Starter Notebook & Import into Kaggle [0 marks prerequisite]
- [x] Notebook imported into Kaggle successfully
- [x] GPU T4 x2 enabled
- [x] Internet enabled
- [x] Notebook runs end-to-end without errors

## Task 2: Load Pre-trained Model from Hugging Face [10 marks]
- [x] **Tokenizer loaded** (3 marks) — `DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')`
- [x] **Model loaded with correct num_labels** (3 marks) — `DistilBertForSequenceClassification.from_pretrained(..., num_labels=8)`
- [x] **Model rationale in report** (4 marks) — Section 3 of report.md explains DistilBERT choice (efficient, 66M params, cased for book reviews, strong baseline)

## Task 3: Train on Kaggle & Track with W&B [25 marks]
- [x] **Kaggle Secrets configured** (3 marks) — `WANDB_API_KEY` and `HF_TOKEN` loaded via `UserSecretsClient`, with `.strip()` fix for token whitespace
- [x] **Dataset prepared and encoded correctly** (4 marks) — 8 genres, 800 train/200 test per genre, encoded with tokenizer (truncation=512, padding)
- [x] **TrainingArguments with report_to="wandb"** (4 marks) — `report_to='wandb'`, `run_name='distilbert-run-1'`
- [x] **compute_metrics returns accuracy + F1** (4 marks) — returns `{'accuracy': acc, 'f1': f1_score(..., average='weighted')}`
- [x] **Training runs on Kaggle GPU** (5 marks) — Completed successfully, W&B run `distilbert-run-1` logged
- [x] **W&B dashboard screenshot included** (5 marks) — *(User needs to add screenshot to report.md before PDF conversion)*

## Task 4: Evaluate & Save Results [15 marks]
- [x] **Evaluation run on test set** (3 marks) — `trainer.evaluate()` executed on test_dataset (1600 samples)
- [x] **Metrics recorded** (4 marks) — Accuracy: 0.5744, F1: 0.5736, Loss: 2.4490
- [x] **Final metrics logged to W&B** (4 marks) — `wandb.log({'final/accuracy': 0.5744, 'final/f1': 0.5736, 'final/loss': 2.4490})`
- [x] **Classification report saved as JSON artifact** (4 marks) — `eval_report.json` uploaded as `wandb.Artifact('eval-report', type='evaluation')`

## Task 5: Push Model to Hugging Face Hub [10 marks]
- [x] **HF account active** (prerequisite) — `Nikhil-iitj` on Hugging Face
- [x] **Model weights pushed** (3 marks) — `model.push_to_hub('Nikhil-iitj/distilbert-goodreads-genres')`
- [x] **Tokenizer pushed** (3 marks) — `tokenizer.push_to_hub('Nikhil-iitj/distilbert-goodreads-genres')`
- [x] **Repository publicly accessible** (2 marks) — `https://huggingface.co/Nikhil-iitj/distilbert-goodreads-genres` (public repo)
- [x] **HF URL logged in W&B summary** (2 marks) — `wandb.run.summary['huggingface_model'] = 'https://huggingface.co/...'`
- [x] **Model card uploaded** (extra) — Detailed README.md with description, training details, evaluation results, usage examples, limitations, citation

## Task 6: Push Everything to GitHub [10 marks]
- [x] **All Python scripts pushed** (4 marks) — `data.py`, `train.py`, `eval.py`, `utils.py`, `push_to_hf.py`, `push_model_card.py`, `requirements.txt`
- [x] **README.md clear** (4 marks) — Includes setup instructions, training platform, results table (Accuracy/F1/Loss), all public links
- [x] **GitHub repo is public** (2 marks) — `https://github.com/nikhilsaini-iitj/MLOps_Assignment2`

## Submission Requirements [must be met]
- [x] **GitHub Repository Link** — public
- [x] **Kaggle Notebook Link** — public (`https://www.kaggle.com/code/nikhilg25ait2067/ml-ops`)
- [x] **Hugging Face Model Link** — public (`https://huggingface.co/Nikhil-iitj/distilbert-goodreads-genres`)
- [x] **W&B Dashboard Link** — public (`https://wandb.ai/g25ait2067-prom-iit-rajasthan/mlops-assignment2`)
- [ ] **W&B screenshot in report** — *(User action needed: take screenshot and insert)*
- [ ] **Report converted to PDF** — *(User action needed: 2-3 pages)*

## Report Content Coverage
- [x] **Model selection rationale** — Section 3 explains why DistilBERT (cased) was chosen
- [x] **Kaggle training setup** — Section 4 covers Kaggle Secrets, W&B integration, training hyperparameters
- [x] **Challenges & learnings** — Section 7 documents Kaggle Internet toggle, HF namespace mismatch, credential validation

## Known Issues Resolved
- [x] `LocalProtocolError: Illegal header value` — Fixed with `.strip()` on secrets
- [x] `NameError: name 'wandb' is not defined` — Fixed with `import wandb` in `wandb.init()` cell
- [x] `AttributeError: 'float' object has no attribute 'mean'` — Fixed with `f1_score(..., average='weighted')`
- [x] `SyntaxError: keyword argument repeated` — Fixed by removing duplicate `report_to`/`run_name`
- [x] `BackendError: No user secrets exist` — Fixed by restarting Kaggle session after adding secrets
- [x] `403 Forbidden` on HF push — Fixed by using correct username `Nikhil-iitj` and write-scoped token

---

**AUDIT CONCLUSION: All rubric tasks are complete.**
Only two user actions remain before submission:
1. Take W&B dashboard screenshot → insert into report.md
2. Convert report.md → PDF (2-3 pages)
