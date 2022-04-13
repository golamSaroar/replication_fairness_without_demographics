from argparse import Namespace
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from datasets import FullDataset, CustomSubset
import numpy as np
from time import time
import json
from collections import defaultdict
import itertools

datasets = ['uci_adult', 'compas', 'law_school']
targets = ['race', 'sex']

with open('hyper_parameters.json') as f:
    optimal_hyper_parameters = json.load(f)

optimizer_dict = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam
}


class Linear(pl.LightningModule):
    def __init__(self,
                 num_features,
                 lr,
                 train_index2value,
                 test_index2value,
                 target_grp,
                 optimizer):
        super().__init__()

        self.lr = lr
        self.train_index2value = train_index2value
        self.test_index2value = test_index2value
        self.target_grp = target_grp
        self.optimizer = optimizer
        self.net = nn.Linear(num_features, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        input = torch.cat([x, y.unsqueeze(1)], dim=1).float()
        return self.net(input).squeeze(dim=-1)

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.get_loss(batch)

        self.log('training/loss', loss)
        self.log('training/accuracy', accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.get_loss(batch)

        self.log('validation/loss', loss)
        self.log('validation/accuracy', accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, s = batch

        pred = torch.round(torch.sigmoid(self.forward(x, y)))
        targets = self.idx_mapping(s, test=True).float()

        loss = self.loss_fct(pred, targets)
        accuracy = torch.true_divide(torch.sum(pred == targets), targets.shape[0])

        self.log('test/loss', loss)
        self.log('test/accuracy', accuracy)

        return targets, pred

    def test_epoch_end(self, outputs):
        targets = []
        preds = []

        for idx in range(len(outputs)):
            targets.append(outputs[idx][0])
            preds.append(outputs[idx][1])

        targets = torch.cat(targets, dim=0)
        preds = torch.cat(preds, dim=0)

        # compute group specific scores
        grp_1_idcs = targets == 0
        grp_2_idcs = targets == 1

        grp_1_targets = targets[grp_1_idcs]
        grp_2_targets = targets[grp_2_idcs]

        accuracy_grp_1 = torch.true_divide(torch.sum(preds[grp_1_idcs] == grp_1_targets), grp_1_targets.shape[0])
        accuracy_grp_2 = torch.true_divide(torch.sum(preds[grp_2_idcs] == grp_2_targets), grp_2_targets.shape[0])

        self.log('test/accuracy_grp_1', accuracy_grp_1)
        self.log('test/accuracy_grp_2', accuracy_grp_2)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def idx_mapping(self, x, test=False):
        out = torch.zeros_like(x)

        if test:
            return self.get_idx_mapping(out, x, self.test_index2value)
        else:
            return self.get_idx_mapping(out, x, self.train_index2value)

    def get_loss(self, batch):
        x, y, s = batch

        pred = self.forward(x, y)
        targets = self.idx_mapping(s).float()
        loss = self.loss_fct(pred, targets)
        accuracy = torch.true_divide(torch.sum(torch.round(torch.sigmoid(pred)) == targets), targets.shape[0])

        return loss, accuracy

    def get_idx_mapping(self, out, x, index_to_values_dict):
        if self.target_grp == 'race':
            for key in index_to_values_dict:
                if index_to_values_dict[key][0] == 'Black':
                    out[x == key] = 1
            return out
        elif self.target_grp == 'sex':
            for key in index_to_values_dict:
                if index_to_values_dict[key][1] == 'Female':
                    out[x == key] = 1
            return out
        else:
            for key in index_to_values_dict:
                if index_to_values_dict[key] == 'protected':
                    out[x == key] = 1
            return out


def get_results(args):
    args.version = str(int(time()))

    pl.seed_everything(args.seed)
    np.random.seed(args.seed)

    dataset = FullDataset(args.dataset)
    test_dataset = FullDataset(args.dataset, test=True)

    all_indices = np.random.permutation(np.arange(0, len(dataset), 1))
    train_indices = all_indices[:int(0.9 * len(all_indices))]
    val_indices = all_indices[int(0.9 * len(all_indices)):]
    train_dataset = CustomSubset(dataset, train_indices)
    val_dataset = CustomSubset(dataset, val_indices)

    # set up dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size)

    # set up model
    model = Linear(num_features=dataset.dimensionality + 1,
                   lr=args.learning_rate,
                   train_index2value=dataset.sensitive_index_to_values,
                   test_index2value=test_dataset.sensitive_index_to_values,
                   target_grp=args.target_grp,
                   optimizer=optimizer_dict[args.optimizer])

    # set up logger
    logdir = f'training_logs/{args.dataset}/ci/{args.target_grp}_version_{args.version}'

    # set up callbacks
    early_stopping = EarlyStopping(
        monitor='validation/accuracy',
        min_delta=0.0,
        patience=10,
        verbose=True,
        mode='max'
    )

    checkpoint = ModelCheckpoint(save_weights_only=True,
                                 dirpath=logdir,
                                 mode='max',
                                 verbose=False,
                                 monitor='validation/accuracy')

    callbacks = [early_stopping, checkpoint]

    # set up trainer
    trainer = pl.Trainer(max_steps=args.train_steps,
                         callbacks=callbacks,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)

    # eval best model on test set
    return trainer.test(test_dataloaders=test_loader, ckpt_path='best')


args = Namespace(
    optimizer='Adagrad',
    seed=0,
    learning_rate=0.001,
    batch_size=256,
    train_steps=5000,
    num_workers=8,  # for the Dataloader
    progress_bar=True
)

accuracies = defaultdict(dict)

for dataset, target in itertools.product(datasets, targets):
    ci_args = Namespace(**vars(args))
    ci_args.dataset = dataset
    ci_args.target_grp = target
    for k, v in optimal_hyper_parameters[dataset]['ARL'].items():
        setattr(ci_args, k, v)

    results = get_results(ci_args)[0]

    accuracies[(target, 'total')][dataset] = results['test/accuracy']
    accuracies[(target, 'Group 1')][dataset] = results['test/accuracy_grp_1']
    accuracies[(target, 'Group 2')][dataset] = results['test/accuracy_grp_2']

with open("ci.txt", "w") as ci_file:
    ci_file.write(str(accuracies))
    ci_file.close()
