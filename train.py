import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader
from TiDE.data import SimpleDataset, collate_fn
from lightning.pytorch.callbacks import ModelCheckpoint

from TiDE.pl_model import Model

torch.set_float32_matmul_precision('medium')

L_n = 50
H_n = 20
train_dataset = SimpleDataset(torch.randn((10000,128)),torch.randn((10000)), L_n, H_n)
train_loader = DataLoader(dataset=train_dataset, collate_fn = collate_fn, num_workers = 32, shuffle=True, batch_size=6)
val_loader = DataLoader(dataset=train_dataset, collate_fn = collate_fn, num_workers = 32, shuffle=False, batch_size=6)

model_config = dict(
    L=L_n, 
    H=H_n, 
    feature_dim=128, 
    feature_encode_dim=4, 
    decode_dim=16, 
    hidden_dim=256, 
    dropout=0.1, 
    bias=False
)

max_iters = 100_000

model = Model(
    model_config = model_config,
    lr = 1e-4,
    min_lr = 2e-5, 
    weight_decay = 0.1,
    warmup_iters = 1000,
    max_iters = max_iters,
    lr_strategy = 'cosine',
)

# define trainer
trainer = L.Trainer(
    default_root_dir='runs/',
    callbacks = ModelCheckpoint(
        mode="min", 
        monitor="val_loss",
        save_top_k=1,
        every_n_epochs=1,
        save_weights_only=False, 
    ),
    accelerator="gpu",
    strategy='auto',
    precision='32',
    max_steps=max_iters,
    gradient_clip_val=1.,
)
trainer.logger._default_hp_metric = None 

trainer.fit(model, train_loader, val_loader)