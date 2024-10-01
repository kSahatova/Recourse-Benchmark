from typing import List, Union

import os
import torch
import numpy as np
import pandas as pd

from carla.data.catalog import OnlineCatalog, CsvCatalog
from carla.evaluation import remove_nans
from carla.evaluation.api import Evaluation

from carla.models.catalog import train_model
from carla.models.catalog import MLModelCatalog, LinearModel, AnnModel


def load_oracle(oracle_filename: str, oracle_dir: str, model_type: str,
                input_dim: int, n_classes=2, hidden_layers: List = None):
    """
    Loads pretrained oracle model
    """
    if model_type == 'linear':
        oracle = LinearModel(input_dim, num_of_classes=n_classes)
    elif model_type == 'ann':
        oracle = AnnModel(input_dim, hidden_layers=hidden_layers, num_of_classes=n_classes)
    weights_path = os.path.join(oracle_dir, oracle_filename)
    oracle.load_state_dict(torch.load(weights_path))

    return oracle


def train_oracle(data_info: dict, model_type, model_info: dict):
    """
    Performs training of a RandomForest model
    :param data_info:
    :param model_type:
    :param model_info:
    :return:
    """
    data_set = None
    data_name = data_info['data_name']
    if data_name in ['adult', 'compas', 'give_me_some_credit', 'heloc']:
        data_set = OnlineCatalog(data_name)
    else:
        try:
            file_path = data_info['file_path']
            data_set = CsvCatalog(file_path=file_path,
                                  continuous=data_info['continuous'],
                                  categorical=data_info['categorical'],
                                  immutables=data_info['immutables'],
                                  target=data_info['target'])
        except Exception as e:
            print(f'The following error occurred during the loading of the dataset {data_name}:', e)

    assert model_type in ['linear', 'ann'], print('The requested oracle has not been implemented yet')
    catalog_model = MLModelCatalog(
        data_set,
        model_type=model_type,
        load_online=False,
        backend="pytorch"
    )

    target = data_set.target
    y_train = data_set.df_train[target]
    x_train = data_set.df_train.drop(target, axis=1)
    y_test = data_set.df_test[target]
    x_test = data_set.df_test.drop(target, axis=1)
    lr = train_model(catalog_model, x_train=x_train, y_train=y_train,
                     x_test=x_test, y_test=y_test, **model_info)

    oracles_dir = r'D:\PycharmProjects\XAIRobustness\CARLA\carla\evaluation\catalog\oracles'
    model_path = f'{model_type}_model_{data_name}.pt'
    torch.save(lr.state_dict(), os.path.join(oracles_dir, model_path))
    print(f"Model saved to {model_path}")


class OracleScore(Evaluation):
    """
    Calculates
    """

    def __init__(self, mlmodel, oracle):
        super().__init__(mlmodel)
        self.oracle = oracle
        self.columns = ["Oracle_score"]

    def get_evaluation(self, factuals=None, counterfactuals=None):
        # only keep the rows for which counterfactuals could be found
        counterfactuals_without_nans, _ = remove_nans(factuals=factuals, counterfactuals=counterfactuals)

        # return empty dataframe if no successful counterfactuals
        if counterfactuals_without_nans.empty:
            return pd.DataFrame(columns=self.columns)

        arr_cf = self.mlmodel.get_ordered_features(counterfactuals_without_nans).to_numpy()
        mlmodel_predictions = np.argmax(self.mlmodel.predict_proba(arr_cf), axis=1)
        oracle_predictions = np.argmax(self.oracle.predict(arr_cf), axis=1)
        oracle_score = (mlmodel_predictions == oracle_predictions).sum() / len(mlmodel_predictions)

        return pd.DataFrame([oracle_score], columns=self.columns)


if __name__ == '__main__':
    """data_info = {'data_name': 'diabetes', 'file_path': r'D:\PycharmProjects\XAIRobustness\data\tabular\diabetes.csv',
                 'continuous': ['Glucose', 'Insulin', 'BMI',
                                'SkinThickness', 'BloodPressure'],
                 'categorical': [],
                 'immutables': ['Pregnancies', 'Age', 'DiabetesPedigreeFunction'],
                 'target': 'Outcome'}"""

    data_info = {'data_name': 'adult'}

    model_info = {'learning_rate': 0.001,
                  'batch_size': 16,
                  'epochs': 30,
                  'hidden_size': [32],
                  'n_estimators': 0,
                  'max_depth': 0}

    train_oracle(data_info, 'ann', model_info,)

