"""GPT-2 model class and functions."""
import os

import pandas as pd
import torch
from fastapi import Response
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)

from utils import TEST_PATH, TRAIN_PATH

GPT_MODEL_NAME = "gpt2"
GPT_NUM_CLASSES = 2
GPT_MAX_LENGTH = 60
GPT_BATCH_SIZE = 32
GPT_EPOCHS = 4
GPT_LEARNING_RATE = 2e-5
GPT_EPSILON = 1e-8
GPT_MODEL_PATH = r"../gpt_classifier.pth"


class GPTDataset(Dataset):
    """Dataset to hold texts and labels."""

    def __init__(self, texts, labels):
        """Init with respective texts and labels."""
        self.texts = texts
        self.labels = labels

    def __len__(self):
        """Length of dataset."""
        return len(self.texts)

    def __getitem__(self, item):
        """Get dataset entry as dict."""
        return {"text": self.texts[item], "label": self.labels[item]}


class GPTCollator(object):
    """Convert texts and labels to GPT2 tensor inputs."""

    def __init__(self, tokenizer):
        """Transform type tokenizer and max sequence length."""
        self.tokenizer = tokenizer
        self.max_legth = GPT_MAX_LENGTH

    def __call__(self, sequences):
        """Tokenize inputs."""
        texts = [sequence["text"] for sequence in sequences]
        labels = [sequence["label"] for sequence in sequences]
        inputs = self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_legth,
        )
        inputs.update({"labels": torch.tensor(labels)})

        return inputs


class GPTClassifier(torch.nn.Module):
    """GPT2 Classifier class."""

    def __init__(self, device: str, load_from_path: bool = False):
        """Init model from path or download it."""
        super(GPTClassifier, self).__init__()
        self.config = GPT2Config.from_pretrained(
            pretrained_model_name_or_path=GPT_MODEL_NAME,
            num_labels=GPT_NUM_CLASSES,
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path=GPT_MODEL_NAME
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=GPT_MODEL_NAME, config=self.config
        )
        self.gpt.resize_token_embeddings(len(self.tokenizer))
        self.gpt.config.pad_token_id = self.gpt.config.eos_token_id
        self.gpt.to(device)

        if load_from_path and os.path.exists(GPT_MODEL_PATH):
            self.gpt.load_state_dict(
                torch.load(
                    f=GPT_MODEL_PATH,
                    map_location=torch.device(device),
                )
            )

    def train(self, data_loader, optimizer, scheduler, device):
        """Train GPT-2."""
        predictions_labels = []
        true_labels = []
        total_loss = 0
        self.gpt.train()
        for batch in tqdm(data_loader, desc="Training", unit="batch"):
            self.gpt.zero_grad()
            true_labels += batch["labels"].numpy().flatten().tolist()
            batch = {
                k: v.type(torch.long).to(device) for k, v in batch.items()
            }
            outputs = self.gpt(**batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gpt.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits = logits.detach().cpu().numpy()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        avg_epoch_loss = total_loss / len(data_loader)

        return true_labels, predictions_labels, avg_epoch_loss

    def evaluate(self, data_loader, device):
        """Evaluate GPT-2."""
        predictions_labels = []
        true_labels = []
        total_loss = 0
        self.gpt.eval()
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            true_labels += batch["labels"].numpy().flatten().tolist()
            batch = {
                k: v.type(torch.long).to(device) for k, v in batch.items()
            }

            with torch.no_grad():
                outputs = self.gpt(**batch)
                loss, logits = outputs[:2]
                logits = logits.detach().cpu().numpy()
                total_loss += loss.item()
                predict_content = logits.argmax(axis=-1).flatten().tolist()
                predictions_labels += predict_content

        avg_epoch_loss = total_loss / len(data_loader)

        return true_labels, predictions_labels, avg_epoch_loss

    def prediction(self, input, max_length=128):
        """Precict with GPT-2."""
        self.gpt.eval()
        encoding = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            outputs = self.gpt(**encoding)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            return "positive" if predicted_class == 1 else "negative"


def train_gpt(device: str, model: GPTClassifier):
    """Train and evaluation pipeline."""
    logger.info("Preparing datasets...")
    train_dataframe = pd.read_csv(TRAIN_PATH).sort_values(
        by="ID", ascending=True
    )
    train_dataset = GPTDataset(
        train_dataframe["Text"], train_dataframe["Label"]
    )
    val_dataframe = pd.read_csv(TEST_PATH).sort_values(by="ID", ascending=True)
    val_dataset = GPTDataset(val_dataframe["Text"], val_dataframe["Label"])
    gpt_collator = GPTCollator(model.tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=GPT_BATCH_SIZE,
        shuffle=True,
        collate_fn=gpt_collator,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=GPT_BATCH_SIZE, collate_fn=gpt_collator
    )

    logger.info("Loading optimizer and scheduler...")
    optimizer = AdamW(
        model.parameters(), lr=GPT_LEARNING_RATE, eps=GPT_EPSILON
    )
    total_steps = len(train_dataloader) * GPT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    logger.info("Train/Evaluate...")
    for epoch in range(GPT_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{GPT_EPOCHS}")
        train_labels, train_predict, train_loss = model.train(
            train_dataloader, optimizer, scheduler, device
        )
        train_acc = accuracy_score(train_labels, train_predict)
        valid_labels, valid_predict, val_loss = model.evaluate(
            val_dataloader, device
        )
        val_acc = accuracy_score(valid_labels, valid_predict)
        report = classification_report(valid_labels, valid_predict)
        logger.info(
            "  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"  # noqa: E501
            % (train_loss, val_loss, train_acc, val_acc)
        )

    logger.info("Saving model")
    torch.save(model.gpt.state_dict(), GPT_MODEL_PATH)
    logger.info("Done!")

    return Response(content=report, media_type="text")
