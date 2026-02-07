from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def faithfulness_score(answer: str, context: str, threshold=0.75):
    if not context.strip():
        return 0.0

    ans_vec = embeddings.embed_query(answer)
    ctx_vec = embeddings.embed_query(context)

    score = cosine(ans_vec, ctx_vec)
    return round(score, 3)
