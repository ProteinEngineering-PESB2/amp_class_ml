import pandas as pd
import sys
import os

from fft_encoder import FFTTransform

path_process = sys.argv[1]
group_to_process = sys.argv[2]

print("Reading datasets")
df_data_training = pd.read_csv(f"{path_process}{group_to_process}/training_dataset.csv")
df_data_validation = pd.read_csv(f"{path_process}{group_to_process}/validation_dataset.csv")
df_data_testing = pd.read_csv(f"{path_process}{group_to_process}/testing_dataset.csv")

list_dfs = [
    (df_data_training, "training_dataset"),
    (df_data_validation, "validation_dataset"),
    (df_data_testing, "testing_dataset"),
]

command = f"mkdir -p {path_process}{group_to_process}_FFT"
print(command)
os.system(command)

print("Start codifications")

for element in list_dfs:

    print("Processing df: ", element[1])

    df_data = element[0]
    name_export = f"{path_process}{group_to_process}_FFT/{element[1]}.csv"

    print("Applying FFT")
    fft_transform = FFTTransform(
        dataset=df_data,
        size_data=len(df_data.columns)-1,
        columns_to_ignore=["activity"]
    )

    response_coded = fft_transform.encoding_dataset()
    response_coded.to_csv(name_export, index=False)