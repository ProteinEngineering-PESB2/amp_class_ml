import pandas as pd
from one_hot_encoding import OneHotEncoder
import sys
import os

path_input = sys.argv[1]
path_export = sys.argv[2]

print("Reading datasets")
df_data_training = pd.read_csv(f"{path_input}train_dataset.csv")
df_data_validation = pd.read_csv(f"{path_input}validation_dataset.csv")
df_data_testing = pd.read_csv(f"{path_input}test_dataset.csv")

list_dfs = [
    (df_data_training, "training_dataset"),
    (df_data_validation, "validation_dataset"),
    (df_data_testing, "testing_dataset"),
]

command = f"mkdir -p {path_export}one_hot"
print(command)
os.system(command)

for element in list_dfs:

    print("Processing: ", element[1])
    
    df_data = element[0]
    name_export = f"{path_export}one_hot/{element[1]}.csv"

    one_hot_instance = OneHotEncoder(dataset=df_data, column_sequence="sequence", max_length=3000)
    df_coded = one_hot_instance.run_process()
    df_coded["activity"] = df_data["activity"]
    df_coded.to_csv(name_export, index=False)

