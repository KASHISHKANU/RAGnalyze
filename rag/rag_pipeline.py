import time
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def run_rag(
    retriever,         
    prompt,             
    question: str,
    model: str = "gpt-4o",
    temperature: float = 0.0
):
    """
    Full RAG pipeline:
    1. Retrieve documents
    2. Build context
    3. Generate answer
    4. Measure latency

    Returns:
        answer (str)
        context (str)
        latency (float)
        retrieved_docs (List[Document])
    """
    retrieved_docs = retriever(question)

    if not retrieved_docs:
        return (
            "No relevant context found. Try refining your question.",
            "",
            0.0,
            []
        )

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs if doc.page_content
    )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )

    chain = prompt | llm | StrOutputParser()
    start = time.time()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    latency = time.time() - start

    return answer, context, latency, retrieved_docs
