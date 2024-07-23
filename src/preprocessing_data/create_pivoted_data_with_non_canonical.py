import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os
import sys

# collect input params
path_activities = sys.argv[1]
path_export_data = sys.argv[2]

# collect raw datasets
list_activities = os.listdir(path_activities)

list_df_activities = []

sequences = []

for i in range(len(list_activities)):
    df_activity = pd.read_csv(f"{path_activities}{list_activities[i]}")
    df_activity = df_activity.reset_index()
    sequences += df_activity["sequence"].tolist()
    list_df_activities.append(df_activity)

# obtain unique sequences
unique_sequences = list(set(sequences))

df_pivot = pd.DataFrame()
df_pivot["sequence"] = unique_sequences

# merging by activity
for i in range(len(list_activities)):

    df_pivot[list_activities[i].split(".")[0]] = df_pivot["sequence"].isin(list_df_activities[i]["sequence"].tolist()).astype(int)

# exporting pivoted dataset
df_pivot.to_csv(f"{path_export_data}pivoted_sequences_non_filter.csv", index=False)