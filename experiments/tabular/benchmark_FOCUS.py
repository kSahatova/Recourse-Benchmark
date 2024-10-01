import timeit
import pandas as pd
from IPython.display import display

from carla.data import OnlineCatalog
from carla.models import MLModelCatalog
from carla.recourse_methods import FOCUS
import carla.evaluation.catalog as evaluation_catalog
from carla.models.negative_instances import predict_negative_instances


import warnings
warnings.filterwarnings("ignore")


data_name = "adult"
dataset = OnlineCatalog(data_name)

# load catalog model
ml_model = MLModelCatalog(dataset, "forest", backend="sklearn", load_online=False)
ml_model.train(max_depth=2, n_estimators=5, force_train=True)

hyperparams = {
    "optimizer": "adam",
    "lr": 0.001,
    "n_class": 2,
    "n_iter": 1000,
    "sigma": 1.0,
    "temperature": 1.0,
    "distance_weight": 0.01,
    "distance_func": "l1",
}

# define your recourse method
recourse_method = FOCUS(ml_model, hyperparams)

factuals = predict_negative_instances(ml_model, dataset.df)
test_factual = factuals.iloc[:5]

display(test_factual)

hyperparams = {
    "optimizer": "adam",
    "lr": 0.001,
    "n_class": 2,
    "n_iter": 1000,
    "sigma": 1.0,
    "temperature": 1.0,
    "distance_weight": 0.01,
    "distance_func": "l1",
}

df_cfs = recourse_method.get_counterfactuals(test_factual)
display(df_cfs)

