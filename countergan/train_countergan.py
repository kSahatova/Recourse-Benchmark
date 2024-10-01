import os
import json
import time
import pickle

import numpy as np
import pandas as pd
import logging
from datetime import datetime

from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import ListedColormap

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

"""
# Install the dev version of the Alibi package
from alibi import __version__ as alibi_version

print(f"Alibi version: {alibi_version}")

alibi_logger = logging.getLogger("alibi")
alibi_logger.setLevel("CRITICAL")

# Disabling eager execution because Alibi is not compatible with it
tf.compat.v1.disable_eager_execution()
print(f"Is TensorFlow running in eager execution mode? -----→ {tf.executing_eagerly()}")
"""

DATA_PATH = r'/data/tabular'
EXPERIMENT_PATH = r'/experiments/tabular'

date = datetime.now().strftime('%Y-%m-%d')


INITIAL_CLASS = 0
DESIRED_CLASS = 1
N_CLASSES = 2

np.set_printoptions(precision=2)
tf.random.set_seed(2020)
np.random.seed(2020)

# Pima indians Diabetes dataset
# https://www.kaggle.com/uciml/pima-indians-diabetes-database
df = pd.read_csv(f"{DATA_PATH}/diabetes.csv", index_col=False)
target_column = "Outcome"
immutable_features = {"Pregnancies", "DiabetesPedigreeFunction", "Age"}

features = set(df.columns) - {target_column}
mutable_features = features - immutable_features
features = list(mutable_features) + list(immutable_features)

x = df[features]
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(df[features].values, y, test_size=0.2)

standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

df[features].sample(5)

