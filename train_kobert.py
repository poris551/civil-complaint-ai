import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "skt/kobert-base-v1"
DATA_PATH = "data/test_complaints.csv"
OUT_DIR = "models/kobert_category"

df = pd.read_csv(DATA_PATH)
df = df[["complaint_text", "category"]].dropna()

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["category"])

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["complaint_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True)).map(tokenize, batched=True)

id2label = {i: label for i, label in enumerate(encoder.classes_)}
label2id = {label: i for i, label in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
with open(Path(OUT_DIR) / "labels.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, ensure_ascii=False, indent=2)

print("saved:", OUT_DIR)
