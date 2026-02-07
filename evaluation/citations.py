import re
import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def sentence_citations(answer: str, docs, threshold=0.78):
    """
    Returns:
    [
      {
        "sentence": "...",
        "citations": [0, 2],
        "score": 0.84
      }
    ]
    """

    sentences = split_sentences(answer)
    doc_texts = [d.page_content for d in docs]

    doc_embeddings = embeddings.embed_documents(doc_texts)
    results = []

    for sent in sentences:
        sent_vec = embeddings.embed_query(sent)

        sims = [cosine(sent_vec, doc_vec) for doc_vec in doc_embeddings]
        cited = [i for i, s in enumerate(sims) if s >= threshold]

        results.append({
            "sentence": sent,
            "citations": cited,
            "max_score": round(max(sims), 3)
        })

    return results
