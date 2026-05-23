from reportlab.platypus import *

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

    styles = (
        getSampleStyleSheet()
    )

    elements = []

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

            "Best Model Information",

            styles["Heading2"]
        )
    )

    elements.append(

        Paragraph(

            f"Best Model: "
            f"{training_results['best_model']}",

            styles["BodyText"]
        )
    )

    elements.append(

        Paragraph(

            f"CV Score: "
            f"{training_results['cv_score']}",

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

    for k, v in training_results[
        "best_params"
    ].items():

        elements.append(

            Paragraph(

                f"{k}: {v}",

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

    for insight in insights:

        elements.append(

            Paragraph(

                f"• {insight}",

                styles["BodyText"]
            )
        )

    # ---------------------------
    # BUILD PDF
    # ---------------------------

    doc.build(elements)