from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset


def compute_ragas_metrics(question: str, answer: str, contexts: list):
    """
    Unsupervised RAGAS evaluation.
    Correct for YouTube / real-world RAG systems.
    """

    if not contexts:
        return {
            "ragas_faithfulness": 0.0
        }

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    })

    scores = evaluate(
        dataset,
        metrics=[faithfulness]
    )

    return {
        "ragas_faithfulness": round(scores["faithfulness"][0], 3)
    }
