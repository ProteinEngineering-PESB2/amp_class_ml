from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
import xgboost

#for metrics
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef)
from sklearn.model_selection import (cross_validate, train_test_split)

import numpy as np
from joblib import dump
from joblib import load

class classification_model(object):

    def __init__(
            self,
            dataset=None,
            column_response=None):
        
        self.dataset = dataset
        self.column_response = column_response
        
        self.scores = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
        self.keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted', 'test_precision_weighted', 'test_accuracy']

        self.model = None

    def split_training_validation(self, val_size=0.2, random_state=42):

        training_df, validation_df = train_test_split(self.dataset, test_size=val_size, random_state=random_state)

        self.X_train = training_df.drop(columns=[self.column_response])
        self.y_train = training_df[self.column_response]
        self.X_test = validation_df.drop(columns=[self.column_response])
        self.y_test = validation_df[self.column_response]

    def define_train_val_data(self, dataset_train=None, dataset_val=None):

        self.X_train = dataset_train.drop(columns=[self.column_response])
        self.y_train = dataset_train[self.column_response]
        self.X_test = dataset_val.drop(columns=[self.column_response])
        self.y_test = dataset_val[self.column_response]

    #function to process average performance in cross val training process
    def process_performance_cross_val(self, performances):
        
        row_response = []
        for i in range(len(self.keys)):
            value = np.mean(performances[self.keys[i]])
            row_response.append(value)
        return row_response

    #function to obtain metrics using the testing dataset
    def get_performances(self, description, predict_label, real_label):
        accuracy_value = accuracy_score(real_label, predict_label)
        f1_score_value = f1_score(real_label, predict_label, average='weighted')
        precision_values = precision_score(real_label, predict_label, average='weighted')
        recall_values = recall_score(real_label, predict_label, average='weighted')
        mcc_values = matthews_corrcoef(real_label, predict_label)

        row = [description, accuracy_value, f1_score_value, precision_values, recall_values, mcc_values]
        return row

    #function to train predictive model
    def train_predictive_model(self, name_model, clf_model):
    
        print("Train model with cross validation")
        clf_model.fit(self.X_train, self.y_train)
        response_cv = cross_validate(clf_model, self.X_train, self.y_train, cv=5, scoring=self.scores)
        performances_cv = self.process_performance_cross_val(response_cv)
        
        print("Predict responses and make evaluation")
        responses_prediction = clf_model.predict(self.X_test)
        response = self.get_performances(name_model, responses_prediction, self.y_test)
        response = response + performances_cv
        return response
    
    def training_KNN_model(self, n_neighbors=5, algorithm="auto"):
        
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            algorithm=algorithm
        )

        response_training = self.train_predictive_model("KNeighbors", self.model)
        return response_training
    
    def training_decisiont_tree(self,criterion="gini", splitter="best"):
        
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter
        )        
        
        response_training = self.train_predictive_model("DecisionTree", self.model)
        return response_training
    
    def training_bagging(self, n_estimators=10, bootstrap=True):

        self.model = BaggingClassifier(
            n_estimators=n_estimators,
            bootstrap=bootstrap
        )
        
        response_training = self.train_predictive_model("Bagging", self.model)
        return response_training
    
    def training_random_forest(self, n_estimators=100, criterion="gini", max_features="sqrt"):
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features
        )

        response_training = self.train_predictive_model("RandomForest", self.model)
        return response_training

    def training_ExtraTree(self, n_estimators=100, criterion="gini", max_features="sqrt"):

        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features
        )

        response_training = self.train_predictive_model("ExtraTrees", self.model)
        return response_training
    
    def training_Adaboost(self, n_estimators=50, algorithm="SAMME.R"):
        
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators,
            algorithm=algorithm
        )

        response_training = self.train_predictive_model("AdaBoost", self.model)
        return response_training
    
    def training_GradientBoosting(self, loss="log_loss", n_estimators=100, criterion="friedman_mse"):

        self.model = GradientBoostingClassifier(
            loss=loss,
            n_estimators=n_estimators,
            criterion=criterion
        )

        response_training = self.train_predictive_model("GradientBoosting", self.model)
        return response_training
    
    def training_HistGradient(self, loss="log_loss", max_iter=100, max_bins=255):
        
        self.model = HistGradientBoostingClassifier(
            loss=loss,
            max_iter=max_iter,
            max_bins=max_bins
        )

        response_training = self.train_predictive_model("Hist Gradient Boosting", self.model)
        return response_training

    def training_SVC(self, kernel="rbf"):

        self.model = SVC(
            kernel=kernel
        )

        response_training = self.train_predictive_model("SVC", self.model)
        return response_training

    def training_GaussianProcess(self):
        
        self.model = GaussianProcessClassifier()
        response_training = self.train_predictive_model("Gaussian Process", self.model)
        return response_training
    
    def training_xgboost(self):
        
        self.model = xgboost.XGBClassifier()
        response_training = self.train_predictive_model("XGBoost", self.model)
        return response_training
    
    def exporting_model(self, name_export="demo_model.joblib"):
        dump(self.model, name_export)
    
    def using_model(self, trained_model_file=None, data_to_predict=None):

        load_model = load(trained_model_file)

        predictions = load_model.predict(data_to_predict)
        return predictions
    