import pandas as pd
import sys
import os

from physicochemical_properties import PhysicochemicalEncoder

path_input = sys.argv[1]
path_export = sys.argv[2]
group_to_process = sys.argv[3]
name_encoders = sys.argv[4]

print("Reading datasets")
df_data_training = pd.read_csv(f"{path_input}train_dataset.csv")
df_data_validation = pd.read_csv(f"{path_input}validation_dataset.csv")
df_data_testing = pd.read_csv(f"{path_input}test_dataset.csv")

list_dfs = [
    (df_data_training, "training_dataset"),
    (df_data_validation, "validation_dataset"),
    (df_data_testing, "testing_dataset"),
]

command = f"mkdir -p {path_export}{group_to_process}"
print(command)
os.system(command)

print("Start codifications")

for element in list_dfs:

    print("Processing df: ", element[1])

    df_data = element[0]
    name_export = f"{path_export}{group_to_process}/{element[1]}.csv"

    physicochemical_encoder = PhysicochemicalEncoder(
        dataset=df_data,
        dataset_encoder=pd.read_csv(name_encoders),
        columns_to_ignore=["activity"],
        name_column_seq="sequence"
    )

    physicochemical_encoder.run_process()

    physicochemical_encoder.df_data_encoded.to_csv(name_export, index=False)