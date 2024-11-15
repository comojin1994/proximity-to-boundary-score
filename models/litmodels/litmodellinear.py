from lightning import LightningModule
from easydict import EasyDict
from models.utils import torch2np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class LitModelLinear(LightningModule):
    def __init__(self, model: nn.Module, args: EasyDict):
        super().__init__()

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self._adjust_learning_rate()

        label = batch[1].reshape(-1)
        logit = self(batch[0])
        pred = logit.argmax(axis=1)

        loss = self.criterion(logit, label)
        label, pred = torch2np(label), torch2np(pred)
        acc = accuracy_score(label, pred)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def evaluate(self, batch, stage=None):
        label = batch[1].reshape(-1)

        logit = self(batch[0])
        pred = logit.argmax(axis=1)

        label, pred = torch2np(label), torch2np(pred)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="eval")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logit = self(batch[0])
        pred = logit.argmax(axis=1)

        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )

        return {"optimizer": self.optimizer}

    def _adjust_learning_rate(self):
        """Decay the learning rate based on schedule"""
        warmup_epoch = self.args.EPOCHS // 10 if self.args.EPOCHS <= 100 else 40

        if self.current_epoch < warmup_epoch:
            cur_lr = self.args.lr * self.current_epoch / warmup_epoch + 1e-9
        else:
            cur_lr = (
                self.args.lr
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (self.current_epoch - warmup_epoch)
                        / (self.args.EPOCHS - warmup_epoch)
                    )
                )
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = cur_lr


class LitModelWeightedLinear(LitModelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        self._adjust_learning_rate()

        label = batch[1].reshape(-1)
        logit = self(batch[0])
        score = batch[2].reshape(-1)

        loss = F.cross_entropy(logit, label, reduction="none") * score
        loss = loss.mean()

        return {"loss": loss}
