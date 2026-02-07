from langchain_community.retrievers import BM25Retriever
from typing import List


def build_hybrid_retriever(chunks, vector_store, k: int = 5):
    """
    Manual Hybrid Retriever:
    - BM25 (keyword)
    - Semantic (embeddings)

    Returns:
        retrieve(query: str) -> List[Document]
    """
    
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    semantic = vector_store.as_retriever(search_kwargs={"k": k})

    def retrieve(query: str) -> List:
        bm25_docs = bm25.invoke(query)
        semantic_docs = semantic.invoke(query)

        seen = set()
        hybrid_docs = []

        for doc in bm25_docs + semantic_docs:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                hybrid_docs.append(doc)

        return hybrid_docs[:k]

    return retrieve
