import numpy as np
import pandas as pd

class SelectionMethod (object):

    def __select_by_std(self, results_df):
        std_grouped_data = results_df[["Algorithm", "F1", "Recall", "Precision", "Accuracy", "Iteration", "Encoder"]].groupby(by=["Algorithm", "Encoder"]).std()        
        filter_std_accuracy_training = np.quantile(std_grouped_data['Accuracy'], .1)
        filter_std_precision_training = np.quantile(std_grouped_data['Precision'], .1)
        filter_std_recall_training = np.quantile(std_grouped_data['Recall'], .1)
        filter_std_f_score_training = np.quantile(std_grouped_data['F1'], .1)

        std_grouped_data["Accuracy_cat_std"] = (
            std_grouped_data["Accuracy"] <= filter_std_accuracy_training).astype(int)
        std_grouped_data["Precision_cat_std"] = (
            std_grouped_data["Precision"] <= filter_std_precision_training).astype(int)
        std_grouped_data["Recall_cat_std"] = (
            std_grouped_data["Recall"] <= filter_std_recall_training).astype(int)
        std_grouped_data["F-score_cat_std"] = (
            std_grouped_data["F1"] <= filter_std_f_score_training).astype(int)
        return std_grouped_data
    
    def __select_by_mean(self, results_df):
        mean_grouped_data = results_df[["Algorithm", "F1", "Recall", "Precision", "Accuracy", "Iteration", "Encoder"]].groupby(by=["Algorithm", "Encoder"]).mean()
        filter_mean_accuracy = np.quantile(mean_grouped_data['Accuracy'], .9)
        filter_mean_precision = np.quantile(mean_grouped_data['Precision'], .9)
        filter_mean_recall = np.quantile(mean_grouped_data['Recall'], .9)
        filter_mean_f_score = np.quantile(mean_grouped_data['F1'], .9)

        mean_grouped_data["Accuracy_cat_mean"] = (
            mean_grouped_data["Accuracy"] >= filter_mean_accuracy).astype(int)
        mean_grouped_data["Precision_cat_mean"] = (
            mean_grouped_data["Precision"] >= filter_mean_precision).astype(int)
        mean_grouped_data["Recall_cat_mean"] = (
            mean_grouped_data["Recall"] >= filter_mean_recall).astype(int)
        mean_grouped_data["F-score_cat_mean"] = (
            mean_grouped_data["F1"] >= filter_mean_f_score).astype(int)
        return mean_grouped_data
    
    def __count_votes(self, results_df, set_name, metric):
        matrix_data = []
        for index in results_df.index:
            algorithm = index[0]
            encoding = index[1]
            accuracy_value = results_df[f'Accuracy_cat_{metric}'][index]
            f_score_value = results_df[f'F-score_cat_{metric}'][index]
            precision_value = results_df[f'Precision_cat_{metric}'][index]
            recall_value = results_df[f'Recall_cat_{metric}'][index]
            row = [algorithm, encoding, accuracy_value, f_score_value, precision_value, recall_value]
            matrix_data.append(row)
        df_process = pd.DataFrame(matrix_data, columns=["Algorithm", "Encoder", f"mean_accuracy_{set_name}", f"mean_f_score_{set_name}", f"mean_precision_{set_name}", f"mean_recall_{set_name}"])
        return df_process
    
    def select(self, min_votes_number, df_results_train, df_results_test):
        std_grouped_data_training = self.__select_by_std(df_results_train)
        std_grouped_data_testing = self.__select_by_std(df_results_test)
        mean_grouped_data_training = self.__select_by_mean(df_results_train)
        mean_grouped_data_testing = self.__select_by_mean(df_results_test)

        df_process_std_training = self.__count_votes(std_grouped_data_training, "training", "std")
        df_process_std_testing = self.__count_votes(std_grouped_data_testing, "testing", "std")
        df_process_mean_training = self.__count_votes(mean_grouped_data_training, "training", "mean")
        df_process_mean_testing = self.__count_votes(mean_grouped_data_testing, "testing", "mean")

        df_merge = df_process_mean_training.merge(right=df_process_mean_testing, on=["Algorithm", "Encoder"])
        df_merge = df_merge.merge(right=df_process_std_testing, on=["Algorithm", "Encoder"])
        df_merge = df_merge.merge(right=df_process_std_training, on=["Algorithm", "Encoder"])
        df_merge['Voting'] = df_merge.sum(axis=1, numeric_only=True)
        df_merge.sort_values(by="Voting", ascending=False)
        df_merge = df_merge[df_merge['Voting']>min_votes_number]
        return df_merge