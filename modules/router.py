def detect_intent(message):

    message = message.lower()


    # overview
    if any(
            x in message
            for x in [
                "overview",
                "summary",
                "dataset overview"
            ]
    ):
        return "overview"


    # preprocess
    elif any(
            x in message
            for x in [
                "preprocess",
                "clean",
                "clean data"
            ]
    ):
        return "preprocess"


    # training
    elif any(
            x in message
            for x in [
                "train",
                "train model",
                "train models"
            ]
    ):
        return "train"


    # feature importance
    elif any(
            x in message
            for x in [
                "feature",
                "importance",
                "feature importance"
            ]
    ):
        return "feature"


    # evaluation
    elif any(
            x in message
            for x in [
                "evaluation",
                "evaluate",
                "report",
                "evaluation report"
            ]
    ):
        return "evaluation"

    if any(

            x in message

            for x in [

                "next",

                "next step",

                "what should i do",

                "what to do next",

                "what next",

                "suggest next step",

                "recommend next step"

            ]

    ):
        return "next_step"

    if any(
            x in message
            for x in [
                "target column is",
                "set target",
                "target is"
            ]
    ):
        return "set_target"


    return "chat"