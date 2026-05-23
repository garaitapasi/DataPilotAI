from reportlab.platypus import (
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from reportlab.platypus.doctemplate import (
    SimpleDocTemplate
)


def generate_report(
    metrics,
    training_results,
    insights
):

    # ---------------------------
    # CREATE PDF
    # ---------------------------

    doc = SimpleDocTemplate(
        "report.pdf"
    )

    styles = getSampleStyleSheet()

    elements = []

    # ---------------------------
    # SAFE VALUES
    # ---------------------------

    problem_type = training_results.get(
        "problem_type",
        "Unknown"
    )

    best_model = training_results.get(
        "best_model",
        "KMeans Clustering"
    )

    cv_score = training_results.get(
        "cv_score",
        "Not Applicable"
    )

    train_score = training_results.get(
        "train_score",
        "Not Applicable"
    )

    test_score = training_results.get(
        "test_score",
        "Not Applicable"
    )

    best_params = training_results.get(
        "best_params",
        {}
    )

    # ---------------------------
    # TITLE
    # ---------------------------

    elements.append(
        Paragraph(
            "DataPilot AI Report",
            styles["Title"]
        )
    )

    # ---------------------------
    # PROBLEM TYPE
    # ---------------------------

    elements.append(
        Spacer(1, 12)
    )

    elements.append(
        Paragraph(
            f"Problem Type: {problem_type}",
            styles["BodyText"]
        )
    )

    # ---------------------------
    # METRICS
    # ---------------------------

    elements.append(
        Spacer(1, 12)
    )

    elements.append(
        Paragraph(
            "Evaluation Metrics",
            styles["Heading2"]
        )
    )

    for k, v in metrics.items():

        elements.append(
            Paragraph(
                f"{k}: {v}",
                styles["BodyText"]
            )
        )

    # ---------------------------
    # MODEL INFO
    # ---------------------------

    elements.append(
        Spacer(1, 12)
    )

    elements.append(
        Paragraph(
            "Model Information",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            f"Best Model: {best_model}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"CV Score: {cv_score}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Train Score: {train_score}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Test Score: {test_score}",
            styles["BodyText"]
        )
    )

    # ---------------------------
    # CLUSTERING INFO
    # ---------------------------

    if problem_type == "Clustering":

        elements.append(
            Spacer(1, 12)
        )

        elements.append(
            Paragraph(
                "Clustering Information",
                styles["Heading2"]
            )
        )

        elements.append(
            Paragraph(
                f"Best Number of Clusters: "
                f"{training_results.get('best_k', 'N/A')}",
                styles["BodyText"]
            )
        )

        elements.append(
            Paragraph(
                f"Silhouette Score: "
                f"{training_results.get('score', 'N/A')}",
                styles["BodyText"]
            )
        )

    # ---------------------------
    # HYPERPARAMETERS
    # ---------------------------

    elements.append(
        Spacer(1, 12)
    )

    elements.append(
        Paragraph(
            "Best Hyperparameters",
            styles["Heading2"]
        )
    )

    if best_params:

        for k, v in best_params.items():

            elements.append(
                Paragraph(
                    f"{k}: {v}",
                    styles["BodyText"]
                )
            )

    else:

        elements.append(
            Paragraph(
                "No hyperparameters available.",
                styles["BodyText"]
            )
        )

    # ---------------------------
    # AI INSIGHTS
    # ---------------------------

    elements.append(
        Spacer(1, 12)
    )

    elements.append(
        Paragraph(
            "AI Insights",
            styles["Heading2"]
        )
    )

    if insights:

        for insight in insights:

            elements.append(
                Paragraph(
                    f"• {insight}",
                    styles["BodyText"]
                )
            )

    else:

        elements.append(
            Paragraph(
                "No insights generated.",
                styles["BodyText"]
            )
        )

    # ---------------------------
    # BUILD PDF
    # ---------------------------

    doc.build(elements)