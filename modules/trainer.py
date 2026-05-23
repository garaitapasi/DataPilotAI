from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)


def train_models(
        df,
        target,
        problem_type
):

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    # Model selection
    if problem_type == "Classification":

        models = {

            "Random Forest":
                GridSearchCV(

                    RandomForestClassifier(),

                    {

                        "n_estimators": [50, 100],

                        "max_depth": [None, 5, 10]
                    },

                    cv=3
                ),

            "Logistic Regression":
                GridSearchCV(

                    LogisticRegression(
                        max_iter=1000
                    ),

                    {

                        "C": [0.01, 0.1, 1, 10],

                        "solver": [
                            "lbfgs",
                            "liblinear"
                        ]
                    },

                    cv=3
                ),

            "XGBoost":
                GridSearchCV(

                    XGBClassifier(

                        eval_metric="logloss"
                    ),

                    {

                        "n_estimators": [50, 100],

                        "max_depth": [3, 5],

                        "learning_rate": [0.01, 0.1]
                    },

                    cv=3
                )
        }

    else:

        models = {

            "Random Forest":
                GridSearchCV(

                    RandomForestRegressor(),

                    {

                        "n_estimators": [50, 100],

                        "max_depth": [None, 5, 10]
                    },

                    cv=3
                ),

            "Linear Regression":
                LinearRegression(),

            "XGBoost":
                GridSearchCV(

                    XGBRegressor(),

                    {

                        "n_estimators": [50, 100],

                        "max_depth": [3, 5],

                        "learning_rate": [0.01, 0.1]
                    },

                    cv=3
                )
        }

    scores = {}

    train_scores = {}
    best_params = {}
    trained_models = {}
    cv_scores_dict = {}

    # Train all models
    for name, model in (
            models.items()
    ):

        model.fit(
            X_train,
            y_train
        )

        if hasattr(model, "best_params_"):
            best_params[name] = (
                model.best_params_
            )

        if hasattr(model, "best_estimator_"):

            trained_models[name] = (
                model.best_estimator_
            )

        else:

            trained_models[name] = model

        cv_model = (
            model.best_estimator_
            if hasattr(
                model,
                "best_estimator_"
            )
            else model
        )

        cv_scores = cross_val_score(

            cv_model,

            X_train,

            y_train,

            cv=5
        )
        cv_scores_dict[name] = (
            cv_scores.mean()
        )

        train_predictions = (
            model.predict(
                X_train
            )
        )

        test_predictions = (
            model.predict(
                X_test
            )
        )

        # Classification
        if problem_type == (
                "Classification"
        ):

            train_score = (
                accuracy_score(
                    y_train,
                    train_predictions
                )
            )

            test_score = (
                accuracy_score(
                    y_test,
                    test_predictions
                )
            )

        # Regression
        else:

            train_score = (
                r2_score(
                    y_train,
                    train_predictions
                )
            )

            test_score = (
                r2_score(
                    y_test,
                    test_predictions
                )
            )

        scores[
            name
        ] = test_score

        train_scores[
            name
        ] = train_score

    # Best model
    best_model_name = max(
        scores,
        key=scores.get
    )

    best_model_params = (
        best_params.get(
            best_model_name,
            {}
        )
    )

    best_cv_score = (
        cv_scores_dict[
            best_model_name
        ]
    )

    best_model_object = (
        trained_models[
            best_model_name
        ]
    )

    best_train_score = (
        train_scores[
            best_model_name
        ]
    )

    best_test_score = (
        scores[
            best_model_name
        ]
    )

    # Overfitting detection
    difference = abs(
        best_train_score
        -
        best_test_score
    )

    if difference > 0.10:

        diagnostic = (
            "⚠ Possible Overfitting Detected"
        )

    else:

        diagnostic = (
            "✅ Model Generalizes Well"
        )

    return (

        scores,

        best_model_name,

        best_model_object,

        X_test,

        y_test,

        best_train_score,

        best_test_score,

        diagnostic,

        best_cv_score,

        best_model_params
    )