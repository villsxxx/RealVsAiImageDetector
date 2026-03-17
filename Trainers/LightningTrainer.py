import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import warnings
from Tools import LightningModelWrapper

warnings.filterwarnings("ignore", message="Checkpoint directory.*exists and is not empty")

class LightningTrainer:
    def __init__(
            self,
            model,
            train_dataset,
            val_dataset,
            criterion,
            model_dir,
            optimizer_config,
            seed,
            batch_size,
            epochs,
            save_every_n_epochs,
            limit_train_batches,
            limit_val_batches,
            num_workers,
            persistent_workers,
            log_every_n_steps,
            check_val_every_n_epoch,
            accelerator,
            devices,
            precision,
    ):
        torch.set_float32_matmul_precision('medium')
        if not isinstance(model, pl.LightningModule):
            model = LightningModelWrapper(
                model=model,
                criterion=criterion,
                optimizer_config=optimizer_config,
            )
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            generator=torch.Generator().manual_seed(seed)
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            generator=torch.Generator().manual_seed(seed)
        ) if val_dataset else None
        callbacks = [
            ModelCheckpoint(dirpath=model_dir,
                            filename="epoch-{epoch:02d}-{val_loss:.4f}",
                            every_n_epochs=save_every_n_epochs,
                            save_top_k=-1),
            ModelCheckpoint(dirpath=model_dir,
                            filename="best-{epoch:02d}-{val_loss:.4f}",
                            monitor="val_loss",
                            mode="min",
                            save_top_k=3),
            ModelCheckpoint(dirpath=model_dir,
                            filename="last-{epoch:02d}",
                            save_last=True),
        ]
        logger = TensorBoardLogger(save_dir=model_dir, name="tensorboard", version=0)
        self.pl_trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            default_root_dir=model_dir,
            logger=logger,
            callbacks=callbacks,
            enable_checkpointing=True,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
        )

    def run(self):
        self.pl_trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )