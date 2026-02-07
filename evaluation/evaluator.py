from evaluation.grounding_metrics import faithfulness_score
from evaluation.answer_metrics import compression_ratio

def evaluate(answer, context, latency):
    return {
        "faithfulness": faithfulness_score(answer, context),
        "latency_sec": round(latency, 2),
        "compression": compression_ratio(answer, context),
    }
