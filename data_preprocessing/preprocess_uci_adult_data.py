import os
import pandas as pd

from data_utils import convert_object_to_category, save_processed_data

base_dir = "data/uci_adult"
train_file = os.path.join(base_dir, 'adult.data')
test_file = os.path.join(base_dir, 'adult.test')

columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
           "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

train_df = pd.read_csv(train_file, sep=',', names=columns, skipinitialspace=True)
test_df = pd.read_csv(test_file, sep=',', names=columns, skiprows=1, skipinitialspace=True)  # first line is not data
test_df['income'] = test_df['income'].apply(lambda x: x[:-1])  # remove the "." from the target column e.g ">50K."

train_df = convert_object_to_category(train_df)
test_df = convert_object_to_category(test_df)

save_processed_data(base_dir, train_df, test_df)
