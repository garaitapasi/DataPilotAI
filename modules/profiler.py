import pandas as pd


def profile_dataset(df):

    rows = df.shape[0]

    profile = []

    for column in df.columns:

        missing_count = df[
            column
        ].isnull().sum()

        missing_percent = (
            missing_count / rows
        ) * 100

        recommendation = (
            "No action needed"
        )

        if missing_count > 0:

            if missing_percent < 5:

                recommendation = (
                    "Remove rows"
                )

            elif missing_percent <= 40:

                if pd.api.types.is_numeric_dtype(
                        df[column]
                ):

                    recommendation = (
                        "Fill with median"
                    )

                else:

                    recommendation = (
                        "Fill with mode"
                    )

            else:

                recommendation = (
                    "Drop column"
                )

        profile.append({

            "Column":
                column,

            "Data Type":
                str(
                    df[column].dtype
                ),

            "Unique Values":
                df[column].nunique(),

            "Missing Count":
                missing_count,

            "Missing %":
                round(
                    missing_percent,
                    2
                ),

            "Recommendation":
                recommendation
        })

    return pd.DataFrame(
        profile
    )