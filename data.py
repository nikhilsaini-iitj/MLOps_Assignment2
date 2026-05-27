import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

# Label mapping (global)
id2label = {0: 'Real', 1: 'Fake'}
label2id = {'Real': 0, 'Fake': 1}


def prepare_fake_news_data(
    csv_path="/kaggle/input/fake-news-classification/train.csv",
    samples_per_class=4000,
    model_name="distilbert-base-uncased",
    max_length=512,
    test_size=0.2,
):
    """Load and prepare the Kaggle Fake News dataset."""

    # Fallback for local runs
    if not os.path.exists(csv_path):
        # Try common local paths
        alt_paths = [
            "./train.csv",
            "../input/fake-news-classification/train.csv",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                csv_path = p
                break

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["title", "text", "label"])
    df["article"] = df["title"].astype(str) + ". " + df["text"].astype(str)
    df["article"] = df["article"].str[: max_length * 4]

    # Balance classes by sampling
    df = (
        df.groupby("label")
        .apply(lambda x: x.sample(min(samples_per_class, len(x)), random_state=42))
        .reset_index(drop=True)
    )

    print(f"Dataset shape after sampling: {df.shape}")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_enc = tokenizer(
        train_df["article"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    test_enc = tokenizer(
        test_df["article"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    return train_enc, train_df["label"].tolist(), test_enc, test_df["label"].tolist(), tokenizer
