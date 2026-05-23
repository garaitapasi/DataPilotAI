from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def perform_clustering(
        df
):

    numeric_df = (
        df.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )
    )


    scaler = (
        StandardScaler()
    )


    scaled_data = (
        scaler.fit_transform(
            numeric_df
        )
    )


    best_k = 2

    best_score = -1

    k_values = []

    inertias = []


    for k in range(
            2,
            6
    ):

        model = (
            KMeans(

                n_clusters=k,

                random_state=42
            )
        )




        labels = (
            model.fit_predict(
                scaled_data
            )
        )
        k_values.append(k)

        inertias.append(
            model.inertia_
        )


        score = (
            silhouette_score(

                scaled_data,

                labels
            )
        )


        if score > best_score:

            best_score = score

            best_k = k


    final_model = (
        KMeans(

            n_clusters=best_k,

            random_state=42
        )
    )


    labels = (
        final_model.fit_predict(
            scaled_data
        )
    )


    numeric_df[
        "Cluster"
    ] = labels


    return (

        numeric_df,

        best_k,

        best_score,

        k_values,

        inertias
    )