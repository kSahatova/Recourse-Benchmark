import timeit
import pandas as pd
from IPython.display import display
from carla.data.catalog import OnlineCatalog, CsvCatalog

from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

from carla.recourse_methods.catalog import Face

from carla.evaluation.catalog import load_oracle
import carla.evaluation.catalog as evaluation_catalog

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

if isinstance(dataset, OnlineCatalog):
    target = dataset.catalog['target']
elif isinstance(dataset, CsvCatalog):
    target = dataset.target

oracle_input_dim = dataset.df_test.drop(target, axis=1).shape[1]
oracle_filename = f'ann_model_{data_name}.pt'  # 'linear_model_adult.pt'
oracle_dir = r'D:\PycharmProjects\XAIRobustness\CARLA\carla\evaluation\catalog\oracles'
oracle_model = load_oracle(oracle_filename, oracle_dir, model_type='ann',
                           input_dim=oracle_input_dim, hidden_layers=[32], )

# define your recourse method
recourse_method = Face(ml_model)

# get some negative (class = 0) instances
factuals = predict_negative_instances(ml_model, dataset.df_test)
factuals = factuals[:1000]

# first initialize the benchmarking class by passing
# black-box-model, recourse method, and factuals into it
start = timeit.default_timer()
counterfactuals = recourse_method.get_counterfactuals(factuals)
stop = timeit.default_timer()

evaluation_measures = [
    evaluation_catalog.YNN(ml_model, {"y": 5, "cf_label": 1}),
    evaluation_catalog.Distance(ml_model, reduction='mean'),
    evaluation_catalog.SuccessRate(),
    evaluation_catalog.OracleScore(ml_model, oracle_model),
    # evaluation_catalog.Redundancy(ml_model, {"cf_label": 1}),
    # evaluation_catalog.ConstraintViolation(ml_model),
    evaluation_catalog.AvgTime({"time": stop - start}),
]

# now run all implemented measurements and create a
# DataFrame which consists of all results
pipeline = [measure.get_evaluation(counterfactuals=counterfactuals, factuals=factuals)
            for measure in evaluation_measures]
results = pd.concat(pipeline, axis=1)

display(results)
