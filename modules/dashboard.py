import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

def test_dashboard():

    st.success(
        "Dashboard module working"
    )


def render_regression_dashboard(
    message,
    metrics,
    tab2,
    tab3,
    problem_type
):
    if problem_type != "Regression":
        return

    if (
        "R²" in metrics
        and
        message.get("y_test") is not None
        and
        message.get("predictions") is not None
    ):

        # ---------------------------
        # TAB 2 → REGRESSION PLOTS
        # ---------------------------

        with tab2:

            st.subheader(
                "Regression Visualization"
            )

            y_test = message["y_test"]

            predictions = message["predictions"]

            fig, ax = plt.subplots(
                figsize=(4, 2)
            )

            ax.scatter(
                y_test,
                predictions
            )

            ax.set_xlabel("Actual")

            ax.set_ylabel("Predicted")

            ax.set_title(
                "Actual vs Predicted"
            )

            st.pyplot(
                fig,
                use_container_width=False
            )

            residuals = (
                y_test - predictions
            )

            fig2, ax2 = plt.subplots(
                figsize=(4, 2)
            )

            ax2.scatter(
                predictions,
                residuals
            )

            ax2.axhline(
                y=0,
                color="red"
            )

            ax2.set_xlabel(
                "Predicted"
            )

            ax2.set_ylabel(
                "Residuals"
            )

            ax2.set_title(
                "Residual Plot"
            )

            st.pyplot(
                fig2,
                use_container_width=False
            )

        # ---------------------------
        # TAB 3 → SHAP
        # ---------------------------

        with tab3:

            try:

                import shap

                st.subheader(
                    "Regression SHAP Analysis"
                )

                model = (
                    st.session_state
                    .training_results["model"]
                )

                X_test = (
                    st.session_state
                    .training_results["X_test"]
                )

                sample_data = X_test[:100]

                explainer = shap.Explainer(
                    model,
                    sample_data
                )

                shap_values = explainer(
                    sample_data
                )

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                shap.plots.beeswarm(
                    shap_values,
                    show=False
                )

                st.pyplot(fig)

            except Exception as e:

                st.warning(
                    f"Regression SHAP failed: {str(e)}"
                )

def render_clustering_dashboard(
    message,
    tab2,
    tab3,
    problem_type
):

    if message.get("problem_type") == "Clustering":

        cluster_df = message.get("cluster_data")

        if (
                cluster_df is not None
                and
                "Cluster" in cluster_df.columns
        ):

            numeric_cols = (
                cluster_df
                .select_dtypes(include=["number"])
                .columns
                .tolist()
            )

            feature_cols = [

                col for col in numeric_cols

                if col != "Cluster"
            ]

            # ---------------------------
            # TAB 2 → VISUALIZATIONS
            # ---------------------------

            with tab2:

                # PCA Scatter Plot
                if len(feature_cols) >= 2:

                    st.subheader(
                        "Cluster Visualization (PCA)"
                    )

                    try:

                        pca = PCA(
                            n_components=2
                        )

                        pca_result = pca.fit_transform(
                            cluster_df[feature_cols]
                        )

                        pca_df = pd.DataFrame({

                            "PC1":
                                pca_result[:, 0],

                            "PC2":
                                pca_result[:, 1],

                            "Cluster":
                                cluster_df["Cluster"]
                                .astype(str)
                        })

                        fig = px.scatter(

                            pca_df,

                            x="PC1",

                            y="PC2",

                            color="Cluster",

                            title="PCA Cluster Projection"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        explained_var = (
                            pca
                            .explained_variance_ratio_
                        )

                        st.caption(

                            f"Explained variance: "

                            f"PC1 = {explained_var[0]:.2%}, "

                            f"PC2 = {explained_var[1]:.2%}"
                        )

                    except Exception as e:

                        st.warning(

                            f"PCA visualization failed: {str(e)}"
                        )

                # Cluster Distribution
                st.subheader(
                    "Cluster Distribution"
                )

                cluster_counts = (

                    cluster_df["Cluster"]

                    .value_counts()

                    .sort_index()

                    .rename_axis("Cluster")

                    .reset_index(name="Count")
                )

                st.bar_chart(
                    cluster_counts.set_index("Cluster")
                )

                st.subheader(
                    "Elbow Method"
                )

                fig = px.line(

                    x=message["k_values"],

                    y=message["inertias"],

                    markers=True
                )

                fig.update_layout(

                    xaxis_title="K",

                    yaxis_title="Inertia"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ---------------------------
            # TAB 3 → ANALYSIS
            # ---------------------------

            with tab3:

                if len(feature_cols) > 0:

                    st.subheader(
                        "Cluster Profile Heatmap"
                    )

                    cluster_profile = (

                        cluster_df

                        .groupby("Cluster")[feature_cols]

                        .mean()

                        .round(2)
                    )

                    st.dataframe(
                        cluster_profile
                    )

                    st.subheader(
                        "Cluster Insights"
                    )

                    for cluster in cluster_profile.index:

                        st.write(
                            f"Cluster {cluster}"
                        )

                        top_features = (

                            cluster_profile

                            .loc[cluster]

                            .sort_values(
                                ascending=False
                            )

                            .head(3)
                        )

                        for feature, value in top_features.items():
                            st.write(
                                f"- {feature}: {round(value, 2)}"
                            )

                    fig, ax = plt.subplots(
                        figsize=(8, 4)
                    )

                    sns.heatmap(

                        cluster_profile,

                        annot=True,

                        cmap="YlGnBu",

                        ax=ax
                    )

                    ax.set_title(
                        "Average Feature Values by Cluster"
                    )

                    st.pyplot(fig)

def render_classification_dashboard(
    message,
    metrics,
    tab2,
    tab3,
    problem_type
):
    if problem_type != "Classification":
        return
    with tab2:
        if message.get("matrix") is not None:
            matrix = message["matrix"]

            fig, ax = plt.subplots(figsize=(4, 2))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig, use_container_width=False)

            try:
                if matrix.shape == (2, 2):
                    tn, fp, fn, tp = matrix.ravel()

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.error(f"FP: {fp}")
                    with c2:
                        st.error(f"FN: {fn}")
                    with c3:
                        st.success(f"TP: {tp}")
                    with c4:
                        st.success(f"TN: {tn}")

                    if "Accuracy" in metrics:
                        if metrics["Accuracy"] > 0.9:
                            st.success("Excellent model performance")
                        elif metrics["Accuracy"] > 0.75:
                            st.info("Good model performance")
                        else:
                            st.warning("Model performance can be improved")



            except Exception:
                pass

        try:

            model = (
                st.session_state
                .training_results["model"]
            )

            X_test = (
                st.session_state
                .training_results["X_test"]
            )

            y_test = (
                st.session_state
                .training_results["y_test"]
            )

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

            fpr, tpr, _ = roc_curve(
                y_test,
                y_prob
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            fig, ax = plt.subplots(
                figsize=(4, 2)
            )

            ax.plot(
                fpr,
                tpr,
                label=f"AUC = {roc_auc:.2f}"
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )

            ax.set_xlabel(
                "False Positive Rate"
            )

            ax.set_ylabel(
                "True Positive Rate"
            )

            ax.set_title(
                "ROC Curve"
            )

            ax.legend()

            st.pyplot(fig)

        except Exception:
            pass

    with tab3:
        try:

            import shap
            import numpy as np

            st.subheader("SHAP Explainability")

            model = st.session_state.training_results["model"]

            X_test = st.session_state.training_results["X_test"]

            sample_data = X_test[:100]

            explainer = shap.Explainer(
                model,
                sample_data
            )

            shap_values = explainer(sample_data)

            # ---------------------------
            # FIX MULTI-DIMENSION ISSUE
            # ---------------------------

            values = shap_values.values

            # MULTICLASS
            if len(values.shape) == 3:
                values = values[:, :, 1]

                shap_values.values = values

            fig, ax = plt.subplots(figsize=(10, 5))

            shap.plots.beeswarm(
                shap_values,
                show=False
            )

            st.pyplot(fig)

        except Exception as e:

            st.warning(
                f"SHAP visualization failed: {str(e)}"
            )

