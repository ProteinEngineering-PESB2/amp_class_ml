import pandas as pd
import sys
import os

from embedding_representations import BioEmbeddings

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

for embedding_type in ["prottrans_uniref", "prottrans_xlu50", "prottrans_t5bdf", "esm1b", "prottrans_xlnet", "prottrans_albert", "prottrans_bert"]:

    command = f"mkdir -p {path_export}{embedding_type}"
    print(command)
    os.system(command)

    print("Start codifications")