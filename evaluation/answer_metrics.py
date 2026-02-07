from rouge_score import rouge_scorer

def compute_rouge(reference: str, hypothesis: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 3),
        "rougeL": round(scores["rougeL"].fmeasure, 3),
    }

def compression_ratio(answer: str, context: str):
    if not context.strip():
        return 0.0
    return round(len(answer.split()) / len(context.split()), 3)
