#!/usr/bin/bash

# One Hot
python run_one_hot.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/

# Physicochemical properties
python run_physicochemical_encoder.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/ Group_0 ../../input_data_for_coding/cluster_encoders.csv
python run_physicochemical_encoder.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/ Group_1 ../../input_data_for_coding/cluster_encoders.csv
python run_physicochemical_encoder.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/ Group_2 ../../input_data_for_coding/cluster_encoders.csv
python run_physicochemical_encoder.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/ Group_3 ../../input_data_for_coding/cluster_encoders.csv
python run_physicochemical_encoder.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/ Group_4 ../../input_data_for_coding/cluster_encoders.csv
python run_physicochemical_encoder.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/ Group_5 ../../input_data_for_coding/cluster_encoders.csv
python run_physicochemical_encoder.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/ Group_6 ../../input_data_for_coding/cluster_encoders.csv
python run_physicochemical_encoder.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/ Group_7 ../../input_data_for_coding/cluster_encoders.csv

# FFT
python run_fft_encoder.py ../../data_coded/antiparasitic/ Group_0
python run_fft_encoder.py ../../data_coded/antiparasitic/ Group_1
python run_fft_encoder.py ../../data_coded/antiparasitic/ Group_2
python run_fft_encoder.py ../../data_coded/antiparasitic/ Group_3
python run_fft_encoder.py ../../data_coded/antiparasitic/ Group_4
python run_fft_encoder.py ../../data_coded/antiparasitic/ Group_5
python run_fft_encoder.py ../../data_coded/antiparasitic/ Group_6
python run_fft_encoder.py ../../data_coded/antiparasitic/ Group_7

# Embedding
python run_embedding.py ../../datasets_per_task/antiparasitic/ ../../data_coded/antiparasitic/