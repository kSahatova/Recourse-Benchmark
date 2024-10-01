from IPython.display import display

import timeit
import pandas as pd
from carla.data import OnlineCatalog
from carla.models import MLModelCatalog
from carla.recourse_methods import Dice
import carla.evaluation.catalog as evaluation_catalog
from carla.models.negative_instances import predict_negative_instances


import warnings
warnings.filterwarnings("ignore")


data_name = "adult"
dataset = OnlineCatalog(data_name)

# load catalog model
model_type = "ann"
ml_model = MLModelCatalog(
    dataset,
    model_type=model_type,
    load_online=True,
    backend="pytorch"
)

hyperparams = {
    'num': 5
}

# define your recourse method
recourse_method = Dice(ml_model, hyperparams)

# get some negative instances
factuals = predict_negative_instances(ml_model, dataset.df_test)
factuals = factuals[:1000]

# first initialize the benchmarking class by passing
# black-box-model, recourse method, and factuals into it
counterfactuals = recourse_method.get_counterfactuals(factuals)


# now you can decide if you want to run all measurements
# or just specific ones.

evaluation_measures = [
    evaluation_catalog.YNN(ml_model, {"y": 6, "cf_label": 1}),
    evaluation_catalog.Distance(ml_model),
    evaluation_catalog.SuccessRate(),
    evaluation_catalog.Redundancy(ml_model, {"cf_label": 1}),
    evaluation_catalog.ConstraintViolation(ml_model),
    evaluation_catalog.AvgTime({"time": timer}),
]

# now run all implemented measurements and create a
# DataFrame which consists of all results
pipeline = [measure.get_evaluation(counterfactuals=counterfactuals, factuals=factuals)
            for measure in evaluation_measures]

results = pd.concat(pipeline, axis=1)
display(results)

