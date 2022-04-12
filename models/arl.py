import pytorch_lightning as pl
import torch
import torch.nn as nn


class Learner(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_units=[64, 32]):
        super().__init__()

        net_list = []
        num_units = [input_shape] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net = nn.Sequential(*net_list)

    def forward(self, x):
        out = self.net(x)
        return torch.squeeze(out, dim=-1)


class Adversary(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_units=[],
                 adv_input={'X', 'Y'},
                 num_groups=None):
        super().__init__()

        net_list = []
        num_inputs = 0
        if 'X' in adv_input:
            num_inputs += input_shape
        if 'Y' in adv_input:
            num_inputs += 1
        if 'S' in adv_input:
            num_inputs += num_groups
        self.adv_input = adv_input
        self.num_groups = num_groups

        num_units = [num_inputs] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))
        net_list.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_list)

    def forward(self, x, y, s):
        inputs = []

        if 'X' in self.adv_input:
            inputs.append(x)
        if 'Y' in self.adv_input:
            inputs.append(y.unsqueeze(1).float())
        if 'S' in self.adv_input:
            inputs.append(nn.functional.one_hot(s.long(), num_classes=self.num_groups).float())

        input = torch.cat(inputs, dim=1).float()
        adv = self.net(input)
        adv_norm = adv / torch.sum(adv)

        out = x.shape[0] * adv_norm + torch.ones_like(adv_norm)

        return torch.squeeze(out, dim=-1)


class ARL(pl.LightningModule):
    def __init__(self,
                 config,
                 input_shape,
                 pretrain_steps,
                 prim_hidden=[64, 32],
                 adv_hidden=[],
                 adv_input={'X', 'Y'},
                 num_groups=None,
                 optimizer=torch.optim.Adagrad,
                 opt_kwargs={}):
        super().__init__()

        self.save_hyperparameters()

        self.learner = Learner(input_shape=input_shape, hidden_units=prim_hidden)
        self.adversary = Adversary(input_shape=input_shape, hidden_units=adv_hidden,
                                   adv_input=adv_input,
                                   num_groups=num_groups)

        # init loss function
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def training_step(self,
                      batch,
                      batch_idx,
                      optimizer_idx):
        x, y, s = batch

        if optimizer_idx == 0:
            loss = self.learner_step(x, y, s)
            return loss

        elif optimizer_idx == 1 and self.global_step > self.hparams.pretrain_steps:
            loss = self.adversary_step(x, y, s)
            return loss
        else:
            return None

    def learner_step(self, x, y, s):
        logits = self.learner(x)
        bce = self.loss_fct(logits, y)

        lambdas = self.adversary(x, y, s)
        loss = torch.mean(lambdas * bce)

        return loss

    def adversary_step(self, x, y, s):
        logits = self.learner(x)
        bce = self.loss_fct(logits, y)

        lambdas = self.adversary(x, y, s)
        loss = -torch.mean(lambdas * bce)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, s = batch
        loss = self.learner_step(x, y, s)

    def validation_step(self, batch, batch_idx):
        x, y, s = batch
        loss = self.learner_step(x, y, s)

    def forward(self, x):
        return self.learner(x)

    def save_scatter(self, x, y, s, name):
        pass


    def configure_optimizers(self):
        optimizer_learn = self.hparams.optimizer(self.learner.parameters(), lr=self.hparams.config["lr"],
                                                 **self.hparams.opt_kwargs)
        optimizer_adv = self.hparams.optimizer(self.adversary.parameters(), lr=self.hparams.config["lr"],
                                               **self.hparams.opt_kwargs)

        return [optimizer_learn, optimizer_adv], []