import os
import torch
import wandb
import argparse
from transformers import (
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from data import prepare_fake_news_data
from utils import FakeNewsDataset, compute_metrics


def train(
    model_name="distilbert-base-uncased",
    max_length=512,
    output_dir="./results_fake_news",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    run_name="distilbert-fake-news-run-1",
    project_name="mlops-assignment2-fakenews",
):
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model": model_name,
            "epochs": num_train_epochs,
            "batch_size": per_device_train_batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "dataset": "Kaggle Fake News Classification",
            "platform": "Kaggle",
            "task": "Fake News Detection",
        },
    )

    train_enc, train_labels, test_enc, test_labels, tokenizer = prepare_fake_news_data(
        model_name=model_name, max_length=max_length
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1},
    ).to(device)

    train_dataset = FakeNewsDataset(train_enc, train_labels)
    test_dataset = FakeNewsDataset(test_enc, test_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()
    return trainer, model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--run_name", default="distilbert-fake-news-run-1")
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        run_name=args.run_name,
    )
