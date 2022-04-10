import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import convert_object_to_category, process_protected_columns, write_to_output_file

base_dir = "data/compas"
file_path = os.path.join(base_dir, "compas-scores-two-years.csv")

columns = ["juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count", "age", "c_charge_degree",
           "c_charge_desc", "age_cat", "sex", "race", "is_recid"]
target_variable = "is_recid"
target_value = "Yes"

df = pd.read_csv(file_path, sep=',', names=columns, skipinitialspace=True)
df = convert_object_to_category(df)
# drop duplicates
df = df[["id"] + columns].drop_duplicates()
df = df[columns]

# Convert race to Black, White and Other
race_dict = {
    "African-American": "Black",
    "Caucasian": "White"
}

process_protected_columns(df, race_dict=race_dict)

df["is_recid"] = df.apply(lambda x: "Yes" if x["is_recid"] == 1.0 else "No", axis=1).astype("category")
df = df.dropna()

train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

write_to_output_file(train_df, test_df, base_dir)
