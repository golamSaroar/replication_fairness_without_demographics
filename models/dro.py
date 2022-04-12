import torch
import torch.nn as nn
import pytorch_lightning as pl


class DROLoss(nn.Module):
    def __init__(self,
                 eta,
                 k):
        super(DROLoss, self).__init__()
        self.eta = eta
        self.k = k
        self.log_sigmoid = nn.LogSigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, y):
        bce = -1 * y * self.log_sigmoid(x) - (1 - y) * self.log_sigmoid(-x)

        if self.k > 0:
            bce = self.relu(bce - self.eta)
            bce = bce ** self.k
            return bce.mean()
        else:
            return bce.mean()


class DRO(pl.LightningModule):
    def __init__(self,
                 config,
                 num_features,
                 pretrain_steps,
                 hidden_units=[64, 32],
                 k=2.0,
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

        self.loss_fct = DROLoss(self.hparams.config['eta'], self.hparams.k)

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input):
        out = self.net(input).squeeze(dim=-1)
        return out

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        logits = self(x)

        if self.global_step > self.hparams.pretrain_steps:
            loss = self.loss_fct(logits, y)
        else:
            loss = self.bce(logits, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, s = batch
        logits = self(x)

        if self.global_step > self.hparams.pretrain_steps:
            loss = self.loss_fct(logits, y)
        else:
            loss = self.bce(logits, y)

    def test_step(self, batch, batch_idx):
        x, y, s = batch
        logits = self(x)

        if self.global_step > self.hparams.pretrain_steps:
            loss = self.loss_fct(logits, y)
        else:
            loss = self.bce(logits, y)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.config['lr'], **self.hparams.opt_kwargs)
