import pandas as pd


def detect_problem(df, target):

    unique_values = df[target].nunique()

    if df[target].dtype == "object":
        return "Classification"

    if unique_values <= 10:
        return "Classification"

    return "Regression"