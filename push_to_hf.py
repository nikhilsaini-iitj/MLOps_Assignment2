import os
import argparse
import wandb
from huggingface_hub import login
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def push_to_hub(
    checkpoint_dir="./results_fake_news",
    repo_id="Nikhil-iitj/distilbert-fake-news",
    project_name="mlops-assignment2-fakenews",
    run_name="distilbert-fake-news-run-1",
):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set.")
    login(token=hf_token)

    model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint_dir)

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    hf_url = f"https://huggingface.co/{repo_id}"
    print(f"Model pushed to {hf_url}")

    wandb.init(project=project_name, name=run_name, resume="allow")
    wandb.run.summary["huggingface_model"] = hf_url
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./results_fake_news")
    parser.add_argument("--repo_id", default="Nikhil-iitj/distilbert-fake-news")
    parser.add_argument("--run_name", default="distilbert-fake-news-run-1")
    args = parser.parse_args()
    push_to_hub(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        run_name=args.run_name,
    )
