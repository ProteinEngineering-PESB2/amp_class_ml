import pandas as pd
import sys
sys.path.insert(0, "../")

from numerical_representation.physicochemical_properties import PhysicochemicalEncoder
from numerical_representation.fft_encoder import FFTTransform
from numerical_representation.embedding_representations import BioEmbeddings

df_data = pd.read_csv("../../datasets_per_task/antibacterial/benchmark_dataset.csv")

print("Processing physicochemical properties")

physicochemical_encoder = PhysicochemicalEncoder(
    dataset=df_data,
    dataset_encoder=pd.read_csv("../../input_data_for_coding/cluster_encoders.csv"),
    columns_to_ignore=["activity"],
    name_column_seq="sequence"
)

physicochemical_encoder.run_process()

print(physicochemical_encoder.df_data_encoded)

print("Applying FFT")
fft_transform = FFTTransform(
    dataset=physicochemical_encoder.df_data_encoded,
    size_data=len(physicochemical_encoder.df_data_encoded.columns)-1,
    columns_to_ignore=["activity"]
)

print(fft_transform.encoding_dataset())

print("Applying embedding")
bioembedding_instance = BioEmbeddings(
    df_data, 
    "sequence", 
    is_reduced=True, 
    device = "cuda"
)

response = bioembedding_instance.apply_onehot()
response["activity"] = df_data["activity"]
print(response)
