import torch
import torch.nn as nn
import pytorch_lightning as pl
from arl import Learner


class IPW(pl.LightningModule):
    def __init__(self,
                 config,
                 num_features,
                 group_probs,
                 hidden_units=[64, 32],
                 optimizer=torch.optim.Adagrad,
                 sensitive_label=False,
                 opt_kwargs={}):

        super().__init__()

        self.save_hyperparameters('config', 'num_features', 'hidden_units', 'optimizer', 'sensitive_label',
                                  'opt_kwargs')
        self.hparams.group_probs = group_probs
        self.group_probs = group_probs

        self.learner = Learner(input_shape=num_features, hidden_units=hidden_units)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        loss = self.learner_step(x, y, s)
        return loss

    def learner_step(self, x, y, s=None):
        logits = self.learner(x)
        bce = self.loss_fct(logits, y)

        if s is not None:
            if self.hparams.sensitive_label:
                sample_weights = torch.index_select(torch.index_select(self.group_probs.to(self.device), 0, s), 1,
                                                    y.long())
            else:
                sample_weights = torch.index_select(self.group_probs.to(self.device), 0, s)

            loss = torch.mean(bce / sample_weights)

        else:
            loss = torch.mean(bce)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, s = batch
        loss = self.learner_step(x, y)

    def test_step(self, batch, batch_idx):
        x, y, s = batch
        loss = self.learner_step(x, y)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.learner.parameters(), lr=self.hparams.config['lr'],
                                           **self.hparams.opt_kwargs)
        return optimizer

    def forward(self, x):
        return self.learner(x)
