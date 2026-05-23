import pandas as pd


def preprocess_data(df):

    actions = []

    total_rows = len(df)

    for column in df.columns:

        missing_count = df[column].isnull().sum()

        if missing_count == 0:
            continue

        missing_percent = (
            missing_count / total_rows
        ) * 100

        # Too many missing
        if missing_percent > 40:

            df.drop(
                columns=[column],
                inplace=True
            )

            actions.append(
                f"{column} dropped ({round(missing_percent,2)}%)"
            )

        # Very low missing
        elif missing_percent < 5:

            df.dropna(
                subset=[column],
                inplace=True
            )

            actions.append(
                f"{column} rows removed ({round(missing_percent,2)}%)"
            )

        # Moderate missing
        else:

            if pd.api.types.is_numeric_dtype(
                    df[column]
            ):

                median_value = (
                    df[column].median()
                )

                df[column] = df[
                    column
                ].fillna(
                    median_value
                )

                actions.append(
                    f"{column} filled with median"
                )

            else:

                mode_value = df[
                    column
                ].mode()[0]

                df[column] = df[
                    column
                ].fillna(
                    mode_value
                )

                actions.append(
                    f"{column} filled with mode"
                )

    # Encode categorical columns
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:

        df[column] = df[
            column
        ].astype(
            "category"
        ).cat.codes

        actions.append(
            f"{column} encoded"
        )

    # Final safety check
    df = df.dropna()

    return df, actions