from argparse import Namespace
import pytorch_lightning as pl
import numpy as np

from datasets import Dataset, CustomSubset


def train_and_evaluate(conf):
    conf = Namespace(**vars(conf))

    pl.seed_everything(conf.seed)
    np.random.seed(conf.seed)

    dataset = Dataset(conf.dataset, sensitive_label=conf.sensitive_label)
    test_dataset = Dataset(conf.dataset, sensitive_label=conf.sensitive_label, test=True)

    config = {
        "lr": conf.primary_lr,
        "batch_size": conf.batch_size,
        "eta": conf.eta
    }

    # create training and validation set
    permuted_indices = np.random.permutation(np.arange(0, len(dataset)))
    train_indices, val_indices = permuted_indices[:int(0.9 * len(permuted_indices))], permuted_indices[
                                                                                      int(0.9 * len(permuted_indices)):]
    train_dataset, val_dataset = CustomSubset(dataset, train_indices), CustomSubset(dataset, val_indices)

    model, _ = ("model", "trainer")  # TODO: write trainer

    auc_scores = {}

    return auc_scores
