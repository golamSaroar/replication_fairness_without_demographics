from pytorch_lightning.callbacks import Callback
import torch
from pytorch_lightning.metrics.functional.classification import auroc
from statistics import mean
from torch.utils.data import DataLoader


class Logger(Callback):
    def __init__(self,
                 dataset,
                 name,
                 batch_size,
                 save_scatter=False):
        super().__init__()

        self.dataset = dataset
        self.name = name
        self.batch_size = batch_size
        self.save_scatter = save_scatter
        self.dataloader = DataLoader(self.dataset, self.batch_size, pin_memory=True)

        if self.save_scatter:
            self.scatter_dataloader = DataLoader(self.dataset, 256, shuffle=True, pin_memory=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.name != 'train':
            super().on_validation_end(trainer, pl_module)

            results = get_all_results(pl_module, self.dataloader, self.dataset.minority_group)

            for key in results:
                pl_module.log(f'{self.name}/{key}', results[key])

        if self.save_scatter:
            save_scatter(pl_module, self.scatter_dataloader, self.name)


def save_scatter(pl_module, dataloader, name):
    x, y, s = next(iter(dataloader))
    pl_module.save_scatter(x.to(pl_module.device), y.to(pl_module.device), s.to(pl_module.device), name)


def get_all_results(pl_module, dataloader, minority):
    predictions = []
    memberships = []
    targets = []
    for x, y, s in iter(dataloader):
        x = x.to(pl_module.device)
        batch_predictions = torch.sigmoid(pl_module(x))
        predictions.append(batch_predictions)
        memberships.append(s)
        targets.append(y)

    prediction_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(targets, dim=0).to(prediction_tensor.device)
    membership_tensor = torch.cat(memberships, dim=0).to(prediction_tensor.device)

    aucs = get_group_auc_dict(prediction_tensor, target_tensor, membership_tensor)
    accuracy = torch.mean(((prediction_tensor > 0.5).int() == target_tensor).float()).item()

    results = {
        'min_auc': min(aucs.values()),
        'macro_avg_auc': mean(aucs.values()),
        'micro_avg_auc': auroc(prediction_tensor, target_tensor).item(),
        'minority_auc': aucs[minority],
        'accuracy': accuracy
    }

    return results


def get_group_auc_dict(predictions, targets, memberships):
    groups = memberships.unique().to(predictions.device)
    groups = groups.to(predictions.device)
    targets = targets.to(predictions.device)
    memberships = memberships.to(predictions.device)
    aucs = {}

    for group in groups:
        indices = (memberships == group)
        if torch.sum(targets[indices]) == 0 or torch.sum(1 - targets[indices]) == 0:
            aucs[int(group)] = 0
        else:
            aucs[int(group)] = auroc(predictions[indices], targets[indices]).item()
    return aucs
