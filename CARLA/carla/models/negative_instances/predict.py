from typing import Any, Tuple, Optional
from typing import Union
import numpy as np
import pandas as pd


def predict_negative_instances(model: Any, data: pd.DataFrame, return_pos: Optional[bool] = False) -> \
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Predicts the data target and retrieves the negative instances. (H^-)

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    model : Tensorflow or PyTorch model
        Object retrieved by load_model()
    data : pd.DataFrame
        Dataset used for predictions
    return_pos: boolean flag
        Identifies whether to return predicted positive class
    Returns
    -------

    """
    # get processed data and remove target
    df = data.copy()
    df["y"] = predict_label(model, df)
    df_neg = df[df["y"] == 0].reset_index(drop=True)
    df_pos = df[df['y'] == 1].reset_index(drop=True)
    df_neg = df_neg.drop("y", axis="columns")
    df_pos = df_pos.drop("y", axis="columns")
    if return_pos:
        return df_neg, df_pos
    return df_neg


def predict_label(model: Any, df: pd.DataFrame, as_prob: bool = False) -> np.ndarray:
    """Predicts the data target

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    model : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    df : pd.DataFrame
        Dataset used for predictions
    Returns
    -------
    predictions :  2d numpy array with predictions
    """

    predictions = model.predict(df)

    if not as_prob:
        predictions = predictions.round()

    return predictions
