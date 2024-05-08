import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pandas as pd
import sys
from class_models import classification_model
import random 

#funcion para crear los set de datos a entrenar
def create_train_test_split(x_data, response, i):
    X_train, X_test, y_train, y_test = train_test_split(x_data, response, random_state=i, test_size=0.3)
    return X_train, X_test, y_train, y_test

df_training = pd.read_csv(sys.argv[1])
df_validation = pd.read_csv(sys.argv[2])
number_iteration = int(sys.argv[3])
path_export = sys.argv[4]

df_data = pd.concat([df_training, df_validation], axis=0)
df_data = df_data.dropna()

random_state_list = [random.randint(0, 10000) for i in range(1000)]
random.shuffle(random_state_list)

for i in range(number_iteration):
    print("Processing iteration: ", i)
    iteration = random_state_list[i]

    x_data = df_data.drop(columns=["activity"])
    response = df_data["activity"]

    X_train, X_test, y_train, y_test = train_test_split(x_data, response, random_state=iteration, test_size=0.3)
    
    name_export = f"{path_export}exploring_{iteration}.csv"

    class_instance = classification_model(
        X_train, 
        y_train, 
        X_test, 
        y_test,
        iteration,
        name_export
    )

    class_instance.make_exploration()