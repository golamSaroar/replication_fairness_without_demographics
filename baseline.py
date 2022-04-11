import torch
import torch.nn as nn
import pytorch_lightning as pl


class Baseline(pl.LightningModule):
    def __init__(self,
                 config,
                 num_features,
                 hidden_units=[64, 32],
                 optimizer=torch.optim.Adagrad,
                 opt_kwargs={}):
        super().__init__()

        self.save_hyperparameters()

        self.optimizer = optimizer

        net_list = []
        num_units = [self.hparams.num_features] + self.hparams.hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net = nn.Sequential(*net_list)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input):
        return self.net(input).squeeze(dim=-1)

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        logits = self(x)
        return self.loss_fct(logits, y)

    def validation_step(self, batch, batch_idx):
        x, y, s = batch
        logits = self(x)
        loss = self.loss_fct(logits, y)

    def test_step(self, batch, batch_idx):
        x, y, s = batch
        logits = self(x)
        loss = self.loss_fct(logits, y)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.config['lr'], **self.hparams.opt_kwargs)
