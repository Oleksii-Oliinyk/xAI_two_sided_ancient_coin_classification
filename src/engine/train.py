from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from src.constants import EXPERIMENT_RUNS_DIR, WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
from src.dataset.custom_dataset import CoinDataset
from src.utils.metrics import compute_metrics, format_metrics, save_confusion_matrix, to_wandb


class ClassifierTrain():

    def __init__(self, data_root, side, model_name, model_config, num_epochs=10):
        """
        data_root:    folder with train/val/test splits
        side:         'Observe' or 'Reverse', a separate model is trained for each side
        model_name:   used for the run folder name
        model_config: {"model": nn.Module class, "lr": float, "batch_size": int}
        num_epochs:   max number of epochs (early stopping may finish earlier)
        """
        self.data_root = Path(data_root)
        self.side = side
        self.num_epochs = num_epochs

        self.patience = 10
        self.batch_size = model_config["batch_size"]
        self.lr = model_config["lr"]
        self.augment_factor = 3
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data
        train_dataset = CoinDataset(self.data_root / "train", side, augment=True, augment_factor=self.augment_factor)
        val_dataset = CoinDataset(self.data_root / "val", side)
        test_dataset = CoinDataset(self.data_root / "test", side)
        self.class_names = train_dataset.classes

        loader_kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        print(f"Side: {side} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} | "
              f"Classes: {len(self.class_names)}")

        # model
        self.model = model_config["model"]().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # outputs
        run_name = f"{model_name}_{side}_{datetime.now():%Y%m%d_%H%M%S}"
        self.output_dir = Path(EXPERIMENT_RUNS_DIR) / run_name
        self.cm_dir = self.output_dir / "confusion_matrices"
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.output_dir / "best_model.pth"
        self.log_path = self.output_dir / "train_log.txt"

        # wandb
        if WANDB_API_KEY:
            wandb.login(key=WANDB_API_KEY)
        self.wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=side,
            tags=[side, model_name],
            dir=str(self.output_dir),
            config={
                "model": model_name,
                "side": side,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.num_epochs,
                "patience": self.patience,
                "augment_factor": self.augment_factor,
                "optimizer": type(self.optimizer).__name__,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
            },
        )

    def _log(self, message, log_file):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    def train_epoch(self, epoch):
        self.model.train()
        actual, predicted = [], []
        total_loss = 0.0

        for images, labels in tqdm(self.train_loader, desc=f"Train {epoch}", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            actual.append(labels.cpu().numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())

        actual = np.concatenate(actual)
        predicted = np.concatenate(predicted)
        return actual, predicted, total_loss / len(actual)

    @torch.no_grad()
    def evaluate(self, dataloader, desc="Eval"):
        self.model.eval()
        actual, predicted = [], []
        total_loss = 0.0

        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            total_loss += self.criterion(logits, labels).item() * images.size(0)

            actual.append(labels.cpu().numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())

        actual = np.concatenate(actual)
        predicted = np.concatenate(predicted)
        return actual, predicted, total_loss / len(actual)

    def fit(self):
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history = []

        with open(self.log_path, "w") as log_file:
            self._log(f"Device: {self.device}", log_file)

            for epoch in range(1, self.num_epochs + 1):
                train_actual, train_predicted, train_loss = self.train_epoch(epoch)
                val_actual, val_predicted, val_loss = self.evaluate(self.val_loader, desc=f"Eval {epoch}")

                train_metrics = compute_metrics(train_actual, train_predicted, train_loss)
                val_metrics = compute_metrics(val_actual, val_predicted, val_loss)
                history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

                self._log(f"Epoch {epoch}/{self.num_epochs}", log_file)
                self._log(format_metrics("Train", train_metrics), log_file)
                self._log(format_metrics("Eval", val_metrics), log_file)

                self.wandb_run.log({
                    **to_wandb("Train", train_metrics),
                    **to_wandb("Eval", val_metrics),
                }, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self._log(f"New best model saved (val loss {val_loss:.4f})", log_file)
                    self.wandb_run.summary["best_epoch"] = epoch
                    self.wandb_run.summary["best_val_loss"] = val_loss
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        self._log(f"Early stopping: no improvement for {self.patience} epochs", log_file)
                        break

        return history

    def test(self):
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        actual, predicted, loss = self.evaluate(self.test_loader, desc="Test")
        metrics = compute_metrics(actual, predicted, loss)

        with open(self.log_path, "a") as log_file:
            self._log(format_metrics("Test", metrics), log_file)

        cm_path = self.cm_dir / "test_best_model.png"
        save_confusion_matrix(
            actual,
            predicted,
            self.class_names,
            title="Test Confusion Matrix - Best Model",
            save_path=cm_path,
        )

        for key, value in to_wandb("Test", metrics).items():
            self.wandb_run.summary[key] = value
        self.wandb_run.log({"Test/Confusion Matrix": wandb.Image(str(cm_path))})
        self.wandb_run.finish()
        return metrics
