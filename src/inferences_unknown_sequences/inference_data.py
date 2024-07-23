import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import argparse
from Bio import SeqIO
from joblib import load

import sys
sys.path.insert(0, "../")

from numerical_representation_strategy.embedding_representations import BioEmbeddings

def apply_encoder(name_coder, dataset):

    embedding_instance = BioEmbeddings(
        dataset=dataset,
        seq_column="sequence",
        is_reduced=True,
        device="cuda",
        column_response="id_seq"
    )

    response = None

    if name_coder == "prottrans_t5bdf":
        response = embedding_instance.apply_prottrans_t5bdf()

    elif name_coder == "prottrans_xlu50":
        response = embedding_instance.apply_prottrans_t5_xlu50()
    
    elif name_coder == "esm1b":
        response = embedding_instance.apply_esm1b()
    
    elif name_coder == "prottrans_uniref":
        response = embedding_instance.apply_prottrans_t5_uniref()
    
    elif name_coder == "prottrans_bert":
        response = embedding_instance.apply_prottrans_bert()
    else:
        response = embedding_instance.apply_prottrans_albert()

    return response

def apply_process_data(activity, df_config, df_input_sequences, models, output):
    print("Processing config data")
    filter_data = df_config[df_config["initials"] == activity]
    filter_data.reset_index(inplace=True)

    print("Apply encoder")
    response = apply_encoder(filter_data["encoder"][0], df_input_sequences)

    print("Load model")

    name_folder = filter_data["full_name_dir"][0]
    name_models = filter_data["name_model"][0]

    model_trained = load(f"{models}{name_folder}/{name_models}")

    data_values = response.drop(columns=["id_seq"]).values

    predictions = model_trained.predict(data_values)
    predictions_proba = model_trained.predict_proba(data_values)

    df_input_sequences["response"] = predictions

    df_proba = pd.DataFrame(data=predictions_proba, columns=["prob-0", "prob-1"])
    df_input_sequences = pd.concat([df_input_sequences, df_proba], axis=1)
    
    print("Export data")
    df_input_sequences.to_csv(f"{output}{activity}_predictions.csv", index=False)
    
# parser arguments
print("Process argument")
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file for encoding peptide sequences", required=True)

parser.add_argument("-i", "--input", help="Fasta file with input sequences", required=True)

parser.add_argument("-a", "--activity", help="Choice the activity to evaluate, use all to explore all available models",
                    choices=['ALL', 'AG','AF','AI','MRSA','AP','BBBP','DDV','AB','GN','AM','AMP','AV','CCC','NP','AD','GP','AMC','AO','AND','CP','QS'], required=True)

parser.add_argument("-o", "--output", help="Path to export predictions", required=True)

parser.add_argument("-m", "--models", help="Path to trained models", required=True)

args = parser.parse_args()

# read config dataset
print("Read config dataset")
df_config = pd.read_csv(args.config)

# read fasta file 
print("Read fasta sequences")
matrix_data = []
for record in SeqIO.parse(args.input, "fasta"):
    row = [
        record.id,
        str(record.seq)
    ]
    matrix_data.append(row)

df_input_sequences = pd.DataFrame(data=matrix_data, columns=["id_seq", "sequence"])

activity_error = []

# process encoder
if args.activity != "ALL":
    print("Processing individual activity: ", args.activity)
    apply_process_data(args.activity, df_config, df_input_sequences, args.models, args.output)

else:
    for index in df_config.index:
        activity = df_config["initials"][index]
        try:
            print("Processing activity: ", activity)
            apply_process_data(
                activity, 
                df_config, 
                df_input_sequences, 
                args.models, 
                args.output)
        except:
            activity_error.append(activity)
    
    print(activity_error)
