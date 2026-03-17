import torch
import pytorch_lightning as pl

class LightningModelWrapper(pl.LightningModule):
    def __init__(self, model, criterion, optimizer_config):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_config = optimizer_config
        self.save_hyperparameters(ignore=['model', 'criterion'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_class = getattr(torch.optim, self.optimizer_config["type"])
        optimizer = opt_class(self.parameters(), **self.optimizer_config["args"])
        return optimizer