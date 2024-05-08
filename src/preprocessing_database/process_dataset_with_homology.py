import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os
from Bio import SeqIO
import random
from sklearn.model_selection import train_test_split
import sys

def read_fasta(name_input):
    matrix_data = []

    with open(name_input) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            row = [
                record.id,
                str(record.seq)
            ]
            matrix_data.append(row)
    
    df_data = pd.DataFrame(data=matrix_data, columns=["seq_id", "sequence"])
    return df_data

def df_to_fasta(df_data, name_fasta):
    doc_export = open(name_fasta, 'w')

    for index in df_data.index:
        sequence = df_data["sequence"][index]

        doc_export.write(f">{index}\n")
        doc_export.write(f"{sequence}\n")
    
    doc_export.close()

name_df = sys.argv[1]
path_export = sys.argv[2]
ration_benchmark = float(sys.argv[3])

df_data = pd.read_csv(name_df)
df_data_positive = df_data[df_data["activity"] == 1]
df_data_negative = df_data[df_data["activity"] == 0]

df_to_fasta(df_data_positive, f"{path_export}positive_data.fasta")
df_to_fasta(df_data_negative, f"{path_export}negative_data.fasta")

command_pos = f"cd-hit -i {path_export}positive_data.fasta -o {path_export}positive_data_filter.fasta -c 0.99"
command_neg = f"cd-hit -i {path_export}negative_data.fasta -o {path_export}negative_data_filter.fasta -c 0.7"

os.system(command_pos)
os.system(command_neg)

df_positive_filter = read_fasta(f"{path_export}positive_data_filter.fasta")
df_negative_filter = read_fasta(f"{path_export}negative_data_filter.fasta")

print("Positive: ", len(df_positive_filter))
print("Negative: ", len(df_negative_filter))

command = f"rm {path_export}*.fasta"
os.system(command)

command = f"rm {path_export}*.clstr"
os.system(command)

positive_sequences = df_positive_filter["sequence"].tolist()
random.shuffle(positive_sequences)

negative_sequences = df_negative_filter["sequence"].tolist()
random.shuffle(negative_sequences)

negative_balanced = negative_sequences[:len(positive_sequences)]

length_benchmark = int(len(positive_sequences)*ration_benchmark)

benchmark_pos = positive_sequences[:length_benchmark]
benchmark_neg = negative_balanced[:length_benchmark]
pos_to_train = positive_sequences[length_benchmark:]
neg_to_train = negative_balanced[length_benchmark:]


df_benchmark_pos = pd.DataFrame()
df_benchmark_pos["sequence"] = benchmark_pos
df_benchmark_pos["activity"] = 1

df_benchmark_neg = pd.DataFrame()
df_benchmark_neg["sequence"] = benchmark_neg
df_benchmark_neg["activity"] = 0

df_benchmark = pd.concat([df_benchmark_pos, df_benchmark_neg])
df_benchmark.to_csv(f"{path_export}benchmark_dataset.csv", index=False)

print(df_benchmark.shape)

df_train_pos = pd.DataFrame()
df_train_pos["sequence"] = pos_to_train
df_train_pos["activity"] = 1

df_train_neg = pd.DataFrame()
df_train_neg["sequence"] = neg_to_train
df_train_neg["activity"] = 0

df_train = pd.concat([df_train_pos, df_train_neg])

X_train, X_test = train_test_split(df_train, test_size=0.1, random_state=42)
X_train_model, X_val = train_test_split(df_train, test_size=0.2, random_state=42)

X_test.to_csv(f"{path_export}test_dataset.csv", index=False)
X_val.to_csv(f"{path_export}validation_dataset.csv", index=False)
X_train_model.to_csv(f"{path_export}train_dataset.csv", index=False)

print(X_test.shape)
print(X_val.shape)
print(X_train_model.shape)