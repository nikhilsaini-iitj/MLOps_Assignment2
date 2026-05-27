import os
import json
import argparse
import wandb
from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from data import prepare_fake_news_data
from utils import compute_metrics, FakeNewsDataset


def evaluate(checkpoint_dir="./results_fake_news", project_name="mlops-assignment2-fakenews", run_name="distilbert-fake-news-run-1"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint_dir)
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir)

    _, _, test_enc, test_labels, _ = prepare_fake_news_data()
    test_dataset = FakeNewsDataset(test_enc, test_labels)

    wandb.init(project=project_name, name=run_name, resume="allow")

    from transformers import Trainer, TrainingArguments
    eval_args = TrainingArguments(
        output_dir="./tmp_eval_fake_news",
        per_device_eval_batch_size=32,
        report_to="wandb",
        run_name=run_name,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    wandb.log({
        "final/loss": eval_results.get("eval_loss"),
        "final/accuracy": eval_results.get("eval_accuracy"),
        "final/f1": eval_results.get("eval_f1"),
    })

    preds_output = trainer.predict(test_dataset)
    preds = preds_output.predictions.argmax(-1)
    labels = [item["labels"].item() for item in test_dataset]

    report = classification_report(labels, preds, target_names=["Real", "Fake"], output_dict=True)
    report_path = "eval_report_fake_news.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(report_path)
    wandb.log_artifact(artifact)

    hf_model_url = os.environ.get("HF_MODEL_URL")
    if hf_model_url:
        wandb.run.summary["huggingface_model"] = hf_model_url

    wandb.finish()
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./results_fake_news")
    parser.add_argument("--run_name", default="distilbert-fake-news-run-1")
    args = parser.parse_args()
    evaluate(checkpoint_dir=args.checkpoint_dir, run_name=args.run_name)
