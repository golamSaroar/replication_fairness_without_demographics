import pandas as pd
import os
import json


# convert object columns to category columns
# https://stackoverflow.com/a/39906514/4082505
def convert_object_to_category(df):
    return pd.concat([
        df.select_dtypes([], ['object']),
        df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
    ], axis=1).reindex(df.columns, axis=1)


def process(df, dictionary, column):
    df[column] = df.apply(lambda x: dictionary[x[column]] if x[column] in dictionary.keys() else 'Other',
                          axis=1).astype('category')


def process_protected_columns(df, race_dict=None, sex_dict=None):
    if race_dict is not None:
        process(df, race_dict, 'race')
    if sex_dict is not None:
        process(df, sex_dict, 'sex')


def write_to_output_file(base_dir, filename, content):
    output_file_path = os.path.join(base_dir, filename)

    with open(output_file_path, mode="w") as output_file:
        output_file.write(json.dumps(content))
        output_file.close()


def construct_vocabulary(df, base_dir):
    categorical_columns = df.select_dtypes(include='category').columns
    vocab_dict = {}

    for column in categorical_columns:
        vocab_dict[column] = list(set(df[column].cat.categories))
    write_to_output_file(base_dir, "vocabulary.json", vocab_dict)


def construct_mean_std(df, base_dir):
    dictionary = df.describe().to_dict()
    mean_std_dict = {}

    for key, value in dictionary.items():
        mean_std_dict[key] = [value['mean'], value['std']]
    write_to_output_file(base_dir, "mean_and_std.json", mean_std_dict)


def save_processed_data(base_dir, train_df, test_df):
    train_data = os.path.join(base_dir, 'train.csv')
    test_data = os.path.join(base_dir, 'test.csv')

    train_df.to_csv(train_data, index=False, header=True)
    test_df.to_csv(test_data, index=False, header=True)

    construct_vocabulary(train_df, base_dir)
    construct_mean_std(train_df, base_dir)
