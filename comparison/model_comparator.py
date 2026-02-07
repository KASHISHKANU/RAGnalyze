from rag.rag_pipeline import run_rag

def compare_models(models, retriever, prompt, question):
    results = {}
    for model in models:
        answer, context, latency = run_rag(
            retriever, prompt, question, model
        )
        results[model] = {
            "answer": answer,
            "latency": round(latency, 2)
        }
    return results
