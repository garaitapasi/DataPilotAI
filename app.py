import joblib

from modules.profiler import profile_dataset
from modules.preprocessor import preprocess_data
from modules.chatbot import ask_llm
from modules.router import detect_intent
from modules.detector import detect_problem
from modules.trainer import train_models
from modules.explainer import get_feature_importance
from modules.evaluator import evaluate_model
from modules.clustering import perform_clustering
from modules.dashboard import *
from modules.report_generator import generate_report


st.set_page_config(
    page_title="DataPilot AI",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 DataPilot AI")
st.info(
    "💡 You can ask: ` Dataset Overview`, `Preprocess`, `Train`, `Evaluate`, "
    "`Feature Importance`, `Next step`, or `You can set target like - Target column is Outcome`"
)
st.subheader("Conversational Data Science Copilot")

# ---------------------------
# SESSION STATE INITIALIZATION
# ---------------------------
defaults = {
    "df": None,
    "original_df": None,
    "current_file": None,
    "messages": [],
    "training_results": None,
    "target_column": None,
    "is_preprocessed": False,
    "analysis_mode": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------
# FILE UPLOADER
# ---------------------------
file = st.file_uploader(
    "Upload Dataset",
    type=["csv"],
    key="file_uploader"
)

# FILE REMOVED → FULL RESET
if file is None:
    st.session_state.df = None
    st.session_state.original_df = None
    st.session_state.current_file = None
    st.session_state.messages = []
    st.session_state.training_results = None
    st.session_state.target_column = None
    st.session_state.is_preprocessed = False
    st.session_state.analysis_mode = None
    st.stop()


if file:
    if st.session_state.current_file != file.name:
        df = pd.read_csv(file)

        st.session_state.df = df
        st.session_state.original_df = df.copy()
        st.session_state.current_file = file.name
        st.session_state.messages = []
        st.session_state.training_results = None
        st.session_state.target_column = None
        st.session_state.is_preprocessed = False
        st.session_state.analysis_mode = None

    df = st.session_state.df

    # ---------------------------
    # SIDEBAR
    # ---------------------------
    with st.sidebar:
        st.title("🚀 DataPilot AI")
        st.divider()

        st.subheader("Dataset Status")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        st.write(f"Missing Values: {df.isnull().sum().sum()}")
        st.write(f"Duplicates: {df.duplicated().sum()}")

        st.divider()

        st.subheader("Workflow Status")
        st.write(f"Preprocessed: {st.session_state.is_preprocessed}")
        st.write(f"Target: {st.session_state.target_column}")
        st.write(f"Trained: {st.session_state.training_results is not None}")

        st.divider()

        st.subheader("Workflow Progress")
        if st.session_state.df is not None:
            st.success("Dataset Uploaded")

        if st.session_state.is_preprocessed:
            st.success("Preprocessing Completed")

        if st.session_state.training_results is not None:
            st.success("Training Completed")

    st.success("Dataset loaded successfully!")
    st.subheader("DataPilot AI Chat")

    # ---------------------------
    # CHAT RENDER
    # ---------------------------
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # ANALYSIS SELECTOR
            if message.get("type") == "analysis_selector":
                mode = st.radio(
                    "Choose one",
                    ["I have a target column", "I don't have a target column"],
                    key=f"analysis_mode_{idx}"
                )

                if st.button("Continue", key=f"analysis_continue_{idx}"):

                    # SUPERVISED
                    if mode == "I have a target column":
                        st.session_state.messages.pop()
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "target_selector",
                            "content": "Please select the target column first."
                        })

                    # UNSUPERVISED
                    else:
                        with st.spinner("Performing clustering..."):
                            try:
                                (
                                    cluster_df,

                                    best_k,

                                    score,

                                    k_values,

                                    inertias

                                ) = perform_clustering(df)
                            except Exception as e:
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Clustering failed: {str(e)}"
                                })
                                st.stop()

                        st.session_state.training_results = {
                            "problem_type": "Clustering",
                            "best_k": best_k,
                            "score": score,
                            "cluster_data": cluster_df,
                            "k_values": k_values,
                            "inertias": inertias
                        }

                        st.session_state.messages.pop()
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "clustering",
                            "content": "Clustering Completed.",
                            "data": cluster_df,
                            "best_k": best_k,
                            "score": score
                        })

                    st.rerun()

            # TARGET SELECTOR
            elif message.get("type") == "target_selector":
                selected_target = st.selectbox(
                    "Choose target column",
                    options=df.columns,
                    key=f"target_selector_box_{idx}"
                )

                if st.button("Start Training", key=f"start_training_{idx}"):
                    st.session_state.target_column = selected_target

                    with st.spinner("Training models..."):
                        try:
                            target = st.session_state.target_column
                            problem_type = detect_problem(df, target)

                            (
                                scores,
                                best_model,
                                model,
                                X_test,
                                y_test,
                                train_score,
                                test_score,
                                diagnostic,
                                best_cv_score,
                                best_model_params
                            ) = train_models(df, target, problem_type)

                            importance_df = get_feature_importance(
                                model,
                                df.drop(columns=[target]).columns
                            )
                            # SAVE TRAINED MODEL
                            joblib.dump(
                                model,
                                "trained_model.pkl"
                            )

                        except Exception as e:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Training failed: {str(e)}"
                            })
                            st.rerun()

                    st.session_state.training_results = {
                        "model": model,
                        "X_test": X_test,
                        "y_test": y_test,
                        "problem_type": problem_type,
                        "importance": importance_df,
                        "best_model": best_model,
                        "scores": scores,
                        "train_score": train_score,
                        "test_score": test_score,
                        "diagnostic": diagnostic,
                        "cv_score": best_cv_score,
                        "best_params": best_model_params,
                    }

                    st.session_state.messages.pop()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "training",
                        "content": "Training Completed.",
                        "problem_type": problem_type,
                        "scores": scores,
                        "best_model": best_model,
                        "train_score": train_score,
                        "test_score": test_score,
                        "diagnostic": diagnostic,
                        "cv_score": best_cv_score,
                        "best_params": best_model_params,
                    })

                    st.rerun()

            # OVERVIEW
            elif message.get("type") == "overview":
                m = message["metrics"]

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Rows", m["Rows"])
                with c2:
                    st.metric("Columns", m["Columns"])
                with c3:
                    st.metric("Missing Values", m["Missing Values"])
                with c4:
                    st.metric("Duplicates", m["Duplicates"])

                st.dataframe(message["profile"])

                missing_chart = df.isnull().sum()

                st.subheader("Missing Values Chart")

                st.bar_chart(missing_chart)

                # ---------------------------
                # CORRELATION HEATMAP
                # ---------------------------

                st.subheader("Correlation Heatmap")

                numeric_df = df.select_dtypes(include="number")

                if not numeric_df.empty:
                    corr = numeric_df.corr()

                    fig, ax = plt.subplots(figsize=(10, 6))

                    sns.heatmap(
                        corr,
                        annot=True,
                        cmap="coolwarm",
                        ax=ax
                    )

                    ax.set_title("Feature Correlation Matrix")

                    st.pyplot(fig)

                # DATA PREVIEW
                st.dataframe(message["preview"])

            # PREPROCESS
            elif message.get("type") == "preprocess":
                for action in message["actions"]:
                    st.write("✅", action)

                st.dataframe(message["preview"])

                st.download_button(
                    "Download Processed Dataset",
                    st.session_state.df.to_csv(index=False),
                    "processed_dataset.csv",
                    mime="text/csv"
                )

            # TRAINING
            elif message.get("type") == "training":
                st.success("Training Completed")
                st.write("Task:", message["problem_type"])

                for name, score in message["scores"].items():
                    st.write(f"{name}: {round(score, 4)}")

                score_df = pd.DataFrame({
                    "Model": list(message["scores"].keys()),
                    "Score": list(message["scores"].values())
                })

                score_df = score_df.sort_values(

                    by="Score",

                    ascending=False
                )
                st.subheader(
                    "Model Leaderboard"
                )

                st.dataframe(
                    score_df,
                    use_container_width=True
                )

                st.subheader("Model Comparison")
                fig = px.bar(
                    score_df,
                    x="Model",
                    y="Score",
                    title="Model Comparison"
                )

                st.plotly_chart(fig)

                st.success(f"Best Model: {message['best_model']}")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Train Score", round(message["train_score"], 4))
                with c2:
                    st.metric("Test Score", round(message["test_score"], 4))

                st.info(message["diagnostic"])

                st.metric(
                    "CV Score",
                    round(
                        message["cv_score"],
                        4
                    )
                )

                st.subheader(
                    "Best Hyperparameters"
                )

                st.json(
                    message["best_params"]
                )

                # DOWNLOAD MODEL
                with open(
                        "trained_model.pkl",
                        "rb"
                ) as f:

                    st.download_button(

                        "Download Trained Model",

                        f,

                        file_name="trained_model.pkl"
                    )

            # FEATURE
            elif message.get("type") == "feature":
                st.dataframe(message["data"])

                if "Feature" in message["data"].columns:
                    st.bar_chart(message["data"].set_index("Feature"))
                elif "Cluster" in message["data"].columns:
                    st.bar_chart(message["data"].set_index("Cluster"))

            # EVALUATION
            elif message.get("type") == "evaluation":
                metrics = message["metrics"]

                problem_type = (
                    st.session_state
                    .training_results["problem_type"]
                )

                # ---------------------------
                # DYNAMIC TABS
                # ---------------------------

                if problem_type == "Classification":

                    tab1, tab2, tab3 = st.tabs([

                        "Metrics",

                        "Visualizations",

                        "SHAP Explainability"
                    ])

                elif problem_type == "Regression":

                    tab1, tab2, tab3 = st.tabs([

                        "Metrics",

                        "Regression Plots",

                        "SHAP Explainability"
                    ])

                else:

                    tab1, tab2, tab3 = st.tabs([

                        "Metrics",

                        "Cluster Visualizations",

                        "Cluster Analysis"
                    ])

                # ---------------------------
                # METRICS TAB
                # ---------------------------

                with tab1:

                    cols = st.columns(len(metrics))

                    for i, (k, v) in enumerate(metrics.items()):
                        with cols[i]:
                            st.metric(

                                k,

                                round(v, 4)
                                if isinstance(v, (int, float))
                                else v
                            )
                # ---------------------------
                # CLUSTERING EVALUATION
                # ---------------------------
                # ---------------------------
                # CLUSTERING EVALUATION
                # ---------------------------

                render_clustering_dashboard(
                    message,
                    tab2,
                    tab3,
                    problem_type
                )
                # ---------------------------
                # REGRESSION EVALUATION
                # ---------------------------
                render_regression_dashboard(
                    message,
                    metrics,
                    tab2,
                    tab3,
                    problem_type
                )



                # ---------------------------
                # CLASSIFICATION EVALUATION
                # ---------------------------
                render_classification_dashboard(
                    message,
                    metrics,
                    tab2,
                    tab3,
                    problem_type
                )

                # ---------------------------
                # AI INSIGHTS
                # ---------------------------

                st.subheader(
                    "AI Insights"
                )

                insights = []



                # ---------------------------
                # CLASSIFICATION
                # ---------------------------

                if problem_type == "Classification":

                    accuracy = metrics.get(
                        "Accuracy",
                        0
                    )

                    if accuracy > 0.9:

                        insights.append(
                            "Excellent classification performance detected."
                        )

                    elif accuracy > 0.75:

                        insights.append(
                            "Good classification performance."
                        )

                    else:

                        insights.append(
                            "Model accuracy can be improved."
                        )

                # ---------------------------
                # REGRESSION
                # ---------------------------

                elif problem_type == "Regression":

                    r2 = metrics.get(
                        "R²",
                        0
                    )

                    if r2 > 0.9:

                        insights.append(
                            "Regression fit is excellent."
                        )

                    elif r2 > 0.75:

                        insights.append(
                            "Regression fit is reasonably strong."
                        )

                    else:

                        insights.append(
                            "Regression model may require tuning."
                        )

                # ---------------------------
                # CLUSTERING
                # ---------------------------

                elif problem_type == "Clustering":

                    score = metrics.get(
                        "Silhouette Score",
                        0
                    )

                    if score > 0.5:

                        insights.append(
                            "Clusters are well separated."
                        )

                    else:

                        insights.append(
                            "Clusters may overlap significantly."
                        )

                # ---------------------------
                # OVERFITTING CHECK
                # ---------------------------

                results = st.session_state.training_results

                train_score = results.get(
                    "train_score"
                )

                test_score = results.get(
                    "test_score"
                )

                if (
                        train_score is not None
                        and
                        test_score is not None
                ):

                    gap = abs(
                        train_score - test_score
                    )

                    if gap > 0.1:
                        insights.append(
                            "Possible overfitting detected due to train-test gap."
                        )

                # ---------------------------
                # DISPLAY INSIGHTS
                # ---------------------------

                for insight in insights:
                    st.info(insight)

                generate_report(

                    metrics,

                    st.session_state.training_results,

                    insights
                )

                with open(
                        "report.pdf",
                        "rb"
                ) as f:

                    st.download_button(

                        "Download AI Report",

                        f,

                        file_name="DataPilot_Report.pdf"
                    )
            # PREPROCESS SUGGESTION
            elif message.get("type") == "preprocess_suggestion":
                if st.button("Yes, Preprocess Dataset", key=f"auto_preprocess_{idx}"):
                    with st.spinner("Preprocessing dataset..."):
                        try:
                            processed_df, actions = preprocess_data(df)
                        except Exception as e:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Preprocessing failed: {str(e)}"
                            })
                            st.stop()

                    st.session_state.df = processed_df
                    st.session_state.is_preprocessed = True

                    st.session_state.messages.pop()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "preprocess",
                        "content": "Preprocessing Completed.",
                        "actions": actions,
                        "preview": processed_df.head()
                    })

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Dataset is now ready for training."
                    })

                    st.rerun()

            # CLUSTERING
            elif message.get("type") == "clustering":
                st.success("Clustering Completed")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Best Clusters", message["best_k"])
                with c2:
                    st.metric("Silhouette Score", round(message["score"], 4))

                st.dataframe(message["data"])

                cluster_df = message["data"]
                numeric_cols = cluster_df.select_dtypes(include=["number"]).columns.tolist()


    # ---------------------------
    # USER INPUT
    # ---------------------------
    user_message = st.chat_input("Ask me anything...")

    if user_message:
        st.session_state.messages.append({
            "role": "user",
            "content": user_message
        })

        intent = detect_intent(user_message)

        # OVERVIEW
        if intent == "overview":
            profile_report = profile_dataset(df)

            health = {
                "Rows": df.shape[0],
                "Columns": df.shape[1],
                "Missing Values": df.isnull().sum().sum(),
                "Duplicates": df.duplicated().sum()
            }

            st.session_state.messages.append({
                "role": "assistant",
                "type": "overview",
                "content": "Here is your dataset overview.",
                "metrics": health,
                "profile": profile_report,
                "preview": df.head()
            })

        # SET TARGET
        elif intent == "set_target":
            found_column = None

            for col in df.columns:
                if col.lower() in user_message.lower():
                    found_column = col
                    break

            if found_column:
                st.session_state.target_column = found_column
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Target column set to '{found_column}'."
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Could not find that column in dataset."
                })

        # PREPROCESS
        elif intent == "preprocess":
            try:
                processed_df, actions = preprocess_data(df)
                st.session_state.df = processed_df
                st.session_state.is_preprocessed = True

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "preprocess",
                    "content": "Preprocessing Completed.",
                    "actions": actions,
                    "preview": processed_df.head()
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Preprocessing failed: {str(e)}"
                })

        # TRAIN
        elif intent == "train":
            missing_values = df.isnull().sum().sum()

            if missing_values > 0:
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "preprocess_suggestion",
                    "content": f"""Dataset needs preprocessing before training.

Missing Values Found:
{missing_values}

Would you like me to preprocess it automatically?"""
                })
                st.rerun()

            st.session_state.messages.append({
                "role": "assistant",
                "type": "analysis_selector",
                "content": "Please choose analysis type."
            })
            st.rerun()

        # FEATURE
        elif intent == "feature":
            results = st.session_state.training_results

            if results is None:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Please train a model first."
                })
                st.rerun()

            if results["problem_type"] == "Clustering":
                cluster_df = results["cluster_data"]

                feature_map = (
                    cluster_df
                    .groupby("Cluster")
                    .mean(numeric_only=True)
                    .reset_index()
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "feature",
                    "content": "Here are cluster-wise average feature values.",
                    "data": feature_map
                })

                st.rerun()

            importance_df = results["importance"]

            st.session_state.messages.append({
                "role": "assistant",
                "type": "feature",
                "content": "Here is feature importance.",
                "data": importance_df
            })

        # EVALUATION
        elif intent == "evaluation":
            results = st.session_state.training_results

            if results is None:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Please train a model first."
                })
                st.rerun()

            if results["problem_type"] == "Clustering":
                metrics = {
                    "Best Clusters": results["best_k"],
                    "Silhouette Score": round(results["score"], 4)
                }

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "evaluation",
                    "content": "Here is the clustering evaluation report.",
                    "problem_type": "Clustering",
                    "metrics": metrics,
                    "matrix": None,
                    "predictions": None,
                    "y_test": None,
                    "cluster_data": results["cluster_data"],
                    "k_values": results["k_values"],
                    "inertias": results["inertias"]
                })

            else:
                try:
                    metrics, matrix = evaluate_model(
                        results["model"],
                        results["X_test"],
                        results["y_test"],
                        results["problem_type"]
                    )
                    metrics["CV Score"] = round(
                        results["cv_score"],
                        4
                    )

                    predictions = results["model"].predict(results["X_test"])
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Evaluation failed: {str(e)}"
                    })
                    st.stop()

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "evaluation",
                    "content": "Here is the evaluation report.",
                    "metrics": metrics,
                    "matrix": matrix if results["problem_type"] == "Classification" else None,
                    "predictions": predictions if results["problem_type"] == "Regression" else None,
                    "y_test": results["y_test"] if results["problem_type"] == "Regression" else None
                })

        # NEXT STEP
        elif intent == "next_step":
            recommendation = ""

            if not st.session_state.is_preprocessed:
                recommendation = """
Recommended next steps:

1. Review dataset overview
2. Preprocess the dataset
3. Select target column
4. Train models
"""
            elif not st.session_state.training_results:
                recommendation = """
Recommended next steps:

1. Select target column
2. Train models
3. Evaluate best model
"""
            else:
                recommendation = """
Recommended next steps:

1. View feature importance
2. Generate evaluation report
3. Ask AI questions
"""

            st.session_state.messages.append({
                "role": "assistant",
                "content": recommendation
            })

        # LLM
        else:
            training_context = ""

            if st.session_state.training_results:
                tr = st.session_state.training_results

                scores_text = ""
                if "scores" in tr and tr["scores"] is not None:
                    scores_text = "\n".join(
                        [f"{k}: {round(v, 4)}" for k, v in tr["scores"].items()]
                    )

                feature_text = ""
                if "importance" in tr and tr["importance"] is not None:
                    top_features = tr["importance"].head(5)

                    if "Feature" in top_features.columns and "Importance" in top_features.columns:
                        feature_text = "\n".join(
                            [
                                f"{row['Feature']} : {round(row['Importance'], 4)}"
                                for _, row in top_features.iterrows()
                            ]
                        )

                best_model_text = tr.get("best_model", "N/A")
                train_score_text = tr.get("train_score", "N/A")
                test_score_text = tr.get("test_score", "N/A")
                diagnostic_text = tr.get("diagnostic", "N/A")
                insights = []

                if "diagnostic" in tr:
                    insights.append(
                        tr["diagnostic"]
                    )

                if tr.get("cv_score", 0) > 0.9:

                    insights.append(
                        "Cross-validation score is excellent."
                    )

                elif tr.get("cv_score", 0) > 0.75:

                    insights.append(
                        "Cross-validation score is reasonably strong."
                    )

                else:

                    insights.append(
                        "Cross-validation score is weak."
                    )

                insights_text = "\n".join(
                    insights
                )
                training_context = f"""
Problem Type:
{tr["problem_type"]}

Best Model:
{best_model_text}

Train Score:
{train_score_text}

Test Score:
{test_score_text}

Diagnostic:
{diagnostic_text}

All Model Scores:
{scores_text}

Top Features:
{feature_text}

CV Score:
{tr.get("cv_score", "N/A")}

Best Hyperparameters:
{tr.get("best_params", {})}

AI Insights:
{insights_text}

"""

            missing_values = df.isnull().sum().sum()
            duplicate_values = df.duplicated().sum()

            data_types = "\n".join(
                [f"{col}: {dtype}" for col, dtype in df.dtypes.items()]
            )

            context = f"""
Dataset Information:

Rows:
{df.shape[0]}

Columns:
{df.shape[1]}

Missing Values:
{missing_values}

Duplicate Rows:
{duplicate_values}

Target Column:
{st.session_state.target_column}

Column Names:
{", ".join(df.columns)}

Data Types:
{data_types}

{training_context}
"""

            with st.spinner("Thinking..."):
                try:
                    answer = ask_llm(context, user_message)
                except Exception as e:
                    answer = f"LLM Error: {str(e)}"

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        st.rerun()