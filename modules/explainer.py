import pandas as pd


def get_feature_importance(
        model,
        feature_names
):

    if hasattr(
            model,
            "feature_importances_"
    ):

        importance = (
            model.feature_importances_
        )

    elif hasattr(
            model,
            "coef_"
    ):

        importance = (
            abs(
                model.coef_[0]
            )
        )

    else:

        return None

    result = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            importance

    })

    result = result.sort_values(
        by="Importance",
        ascending=False
    )

    return result