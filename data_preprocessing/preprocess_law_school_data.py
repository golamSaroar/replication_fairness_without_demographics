import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import convert_object_to_category, process_protected_columns, save_processed_data

base_dir = "data/law_school"
file_path = os.path.join(base_dir, "lsac.sas7bdat")

columns = ['zfygpa', 'zgpa', 'DOB_yr', 'parttime', 'gender', 'race', 'tier', 'fam_inc', 'lsat', 'ugpa', 'pass_bar',
           'index6040']

df = pd.read_sas(file_path)
df = convert_object_to_category(df)
df = df[columns]

renameColumns = {'gender': 'sex',
                 'index6040': 'weighted_lsat_ugpa',
                 'fam_inc': 'family_income',
                 'tier': 'cluster_tier',
                 'parttime': 'isPartTime'}
df = df.rename(columns=renameColumns)
columns = list(df.columns)

df['pass_bar'] = df['pass_bar'].fillna(value=0.0)
df['pass_bar'] = df.apply(lambda x: 'Passed' if x['pass_bar'] == 1.0 else 'Failed_or_not_attempted', axis=1).astype(
    'category')

df['zfygpa'] = df['zfygpa'].fillna(value=0.0)
df['zgpa'] = df['zgpa'].fillna(value=0.0)
df['DOB_yr'] = df['DOB_yr'].fillna(value=0.0)
df = df.dropna()

df['isPartTime'] = df.apply(lambda x: 'Yes' if x['isPartTime'] == 1.0 else 'No', axis=1).astype('category')

race_dict = {
    3.0: 'Black',
    7.0: 'White'
}

sex_dict = {
    b'female': 'Female',
    b'male': 'Male'
}

process_protected_columns(df, race_dict=race_dict, sex_dict=sex_dict)
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)
save_processed_data(base_dir, train_df, test_df)
