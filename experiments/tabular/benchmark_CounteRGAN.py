from IPython.display import display

import timeit
from carla.data import CsvCatalog,  OnlineCatalog
from carla.models import MLModelCatalog
from carla.recourse_methods import Dice
import carla.evaluation.catalog as evaluation_catalog
from carla.models.negative_instances import predict_negative_instances


dataset_name = 'compas'
dataset = None
if dataset_name == 'compas':
    dataset = OnlineCatalog(dataset_name)
elif dataset_name == 'diabetes':
    data_path = r'D:\PycharmProjects\XAIRobustness\data\tabular\diabetes.csv'
    dataset = CsvCatalog(file_path=data_path, target='Outcome', categorical=[],
                         continuous=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
                         immutables=['Pregnancies', 'Age', 'DiabetesPedigreeFunction'])


model_type = "ann"
ml_model = MLModelCatalog(dataset, model_type=model_type,
                          load_online=True, backend="pytorch"
)


