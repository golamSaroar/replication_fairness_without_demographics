import pandas as pd
import os


def convert_object_to_category(df):
    return pd.concat([
        df.select_dtypes([], ['object']),
        df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
    ], axis=1).reindex(df.columns, axis=1)


def process_dict(df, dict, column):
    df[column] = df.apply(lambda x: dict[x[column]] if x[column] in dict.keys() else 'Other',
                          axis=1).astype('category')


def process_protected_columns(df, race_dict=None, sex_dict=None):
    if race_dict is not None:
        process_dict(df, race_dict, 'race')
    if sex_dict is not None:
        process_dict(df, sex_dict, 'sex')


def write_to_output_file(train_df, test_df, base_dir):
    train_df.to_csv(os.path.join(base_dir, 'train.csv'), index=False, header=True)
    test_df.to_csv(os.path.join(base_dir, 'test.csv'), index=False, header=True)
