
import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from .model import TiDE

class Model(L.LightningModule):
    def __init__(
        self,
        model_config,
        lr,
        min_lr, 
        weight_decay,
        warmup_iters,
        max_iters,
        lr_strategy = 'constant',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()
        self._create_model()

    def _create_model(self):
        self.model = TiDE(**self.hparams.model_config)

    def forward(self, batch,) -> torch.Tensor:
        prediction = self.model(batch['lookback'], batch['dynamic'])
        return prediction

    @torch.no_grad()
    def infer(self, batch, *args, **kwargs):
        return self.forward(batch, *args, **kwargs)
    
    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.optimizers().param_groups[0]['lr'] < self.hparams.min_lr and self.lr_schedulers().last_epoch > self.hparams.warmup_iters:
            self.optimizers().param_groups[0]['lr'] = self.hparams.min_lr
            self.lr_schedulers().last_epoch -= 1

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        if self.hparams.lr_strategy == 'constant':
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_iters,
            )
        elif self.hparams.lr_strategy == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_iters,
                num_training_steps=self.hparams.max_iters,
            )
        elif self.hparams.lr_strategy == 'linear':
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_iters,
                num_training_steps=self.hparams.max_iters,
            )
        else:
            raise NotImplementedError
            
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        loss = self.criterion(self.forward(batch), batch['label'])
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.criterion(self.forward(batch), batch['label'])
        
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

