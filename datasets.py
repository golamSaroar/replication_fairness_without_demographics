import os
import pandas as pd
import json
import torch
import itertools

DATASET_INFO = {
    "compas": {
        "sensitive_column_names": ['race', 'sex'],
        "sensitive_column_values": ['Black', 'Female'],
        "target_variable": "is_recid",
        "target_value": "Yes"},
    "uci_adult": {
        "sensitive_column_names": ['race', 'sex'],
        "sensitive_column_values": ['Black', 'Female'],
        "target_variable": "income",
        "target_value": ">50K"},
    "law_school": {
        "sensitive_column_names": ['race', 'sex'],
        "sensitive_column_values": ['Black', 'Female'],
        "target_variable": "pass_bar",
        "target_value": "Passed"
    }
}


class Dataset:
    def __init__(self, dataset_name,
                 test=False,
                 hide_sensitive_columns=True,
                 binarize_protected_groups=True,
                 sensitive_label=False):

        base_path = os.path.join("data", dataset_name)
        data_path = os.path.join(base_path, "test.csv" if test else "train.csv")
        vocab_path = os.path.join(base_path, "vocabulary.json")
        mean_std_path = os.path.join(base_path, "mean_std.json")
        sensitive_column_names = DATASET_INFO[dataset_name]["sensitive_column_names"].copy()
        sensitive_column_values = DATASET_INFO[dataset_name]["sensitive_column_values"].copy()
        target_variable = DATASET_INFO[dataset_name]["target_variable"]
        target_value = DATASET_INFO[dataset_name]["target_value"]

        self.hide_sensitive_columns = hide_sensitive_columns
        self.sensitive_label = sensitive_label

        # load data
        features = pd.read_csv(data_path, sep=',', header=False)
        columns = list(features.columns)

        # load mean and std
        with open(mean_std_path) as json_file:
            mean_std = json.load(json_file)

        # normalize numerical features
        for key in mean_std:
            features[key] -= mean_std[key][0]
            features[key] /= mean_std[key][1]

        # create labels
        labels = (features[target_variable].to_numpy() == target_value).astype(int)
        self.labels = torch.from_numpy(labels)

        if binarize_protected_groups:
            for col, val in zip(sensitive_column_names, sensitive_column_values):
                features[col] = features[col].apply(lambda x: float(x == val))

        unique_values = [tuple(features[col].unique()) for col in sensitive_column_names]
        index_to_values = list(itertools.product(*unique_values))

        # create the inverse dictionary:
        values_to_index = {vals: index for index, vals in enumerate(index_to_values)}
        if binarize_protected_groups:
            self.index_to_values = {
                index: tuple(
                    val if vals[idx] == 1 else "Other"
                    for idx, val in enumerate(sensitive_column_values)
                ) for index, vals in enumerate(index_to_values)}
        else:
            self.index_to_values = {idx: val for idx, val in enumerate(index_to_values)}

        columns.remove(target_variable)
        self.sensitives = features[sensitive_column_names]
        if hide_sensitive_columns:  # remove sensitive columns
            for c in sensitive_column_names:
                columns.remove(c)
        features = features[columns]

        # Create a tensor with protected group membership indices
        self.group_memberships = torch.empty(len(features), dtype=int)  # type: ignore
        for i in range(len(self.group_memberships)):
            s = tuple(self.sensitives.iloc[i])
            self.group_memberships[i] = values_to_index[s]

        # compute the minority group (the one with the fewest members) and group probabilities
        vals, counts = self.group_memberships.unique(return_counts=True)
        self.minority_group = vals[counts.argmin().item()].item()

        # calculate group probabilities for IPW
        if sensitive_label:
            prob_identifier = torch.stack([self.group_memberships, self.labels], dim=1)
            vals, counts = prob_identifier.unique(return_counts=True, dim=0)
            probs = torch.true_divide(counts, torch.sum(counts))
            self.group_probs = probs.reshape(-1, 2)
        else:
            vals, counts = self.group_memberships.unique(return_counts=True)
            self.group_probs = torch.true_divide(counts, torch.sum(counts).float())

        # load vocab
        with open(vocab_path) as json_file:
            vocab = json.load(json_file)

        del vocab[target_variable]

        tensors = []
        for c in columns:
            if c in vocab:
                vals = list(vocab[c])
                val2int = {vals[i]: i for i in range(len(vals))}
                features[c] = features[c].apply(lambda x: val2int[x])
                one_hot = torch.nn.functional.one_hot(
                    torch.tensor(features[c].values).long(),
                    len(vals))
                for i in range(one_hot.size(-1)):
                    tensors.append(one_hot[:, i].float())
            else:
                tensors.append(torch.tensor(features[c].values).float())

        self.features = torch.stack(tensors, dim=1).float()

    def __getitem__(self, index):
        x = self.features[index]
        y = float(self.labels[index])
        s = self.group_memberships[index].item()
        return x, y, s

    def __len__(self):
        return self.features.size(0)

    @property
    def dimensionality(self):
        return self.features.size(1)

    @property
    def sensitive_index_to_values(self):
        return self.index_to_values


class CustomSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        # calculate group probabilities for IPW
        if self.dataset.sensitive_label:
            prob_identifier = torch.stack([self.group_memberships, self.labels], dim=1)
            vals, counts = prob_identifier.unique(return_counts=True, dim=0)
            probs = torch.true_divide(counts, torch.sum(counts))
            self.group_probs = probs.reshape(-1, 2)
        else:
            vals, counts = self.group_memberships.unique(return_counts=True)
            self.group_probs = torch.true_divide(counts, torch.sum(counts).float())

    def __len__(self):
        return len(self.indices)

    @property
    def group_memberships(self):
        return self.dataset.group_memberships[self.indices]

    @property
    def labels(self):
        return self.dataset.labels[self.indices]

    @property
    def dimensionality(self):
        return self.dataset.dimensionality

    @property
    def sensitive_index_to_values(self):
        return self.dataset.index_to_values
