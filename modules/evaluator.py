import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(
        model,
        X_test,
        y_test,
        problem_type
):

    predictions = model.predict(
        X_test
    )


    if problem_type == (
            "Classification"
    ):

        metrics = {

            "Accuracy":
                accuracy_score(
                    y_test,
                    predictions
                ),

            "Precision":
                precision_score(
                    y_test,
                    predictions,
                    average="weighted"
                ),

            "Recall":
                recall_score(
                    y_test,
                    predictions,
                    average="weighted"
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    predictions,
                    average="weighted"
                )

        }

        matrix = confusion_matrix(
            y_test,
            predictions
        )

        return (
            metrics,
            matrix
        )


    else:

        metrics = {

            "R²":
                r2_score(
                    y_test,
                    predictions
                ),

            "MAE":
                mean_absolute_error(
                    y_test,
                    predictions
                ),

            "RMSE":
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )

        }

        return (
            metrics,
            None
        )