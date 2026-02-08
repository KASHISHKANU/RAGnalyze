import os
from dotenv import load_dotenv
from typing import List

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from evaluation.ragas_metrics import compute_ragas_metrics
from fastapi.responses import StreamingResponse
from evaluation.cost_estimator import estimate_cost

load_dotenv()

from rag.ingestion import load_documents
from rag.chunking import chunk_docs
from rag.vector_store import build_vector_store
from rag.retriever import build_hybrid_retriever
from rag.prompt_engine import get_prompt
from rag.rag_pipeline import run_rag, run_rag_stream

from evaluation.evaluator import evaluate
from evaluation.answer_metrics import compute_rouge
from evaluation.citations import sentence_citations
from evaluation.hallucination import hallucination_stats

app = FastAPI(title="RAGnalyze")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

@app.post("/analyze")
def analyze_api(data: dict):
    return {
        "hallucination": "12%",
        "grounding": 0.91,
        "rouge_l": 0.87,
        "compression": 1.42
    }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": [
                "gpt-4o",
                "gpt-3.5-turbo"]

        }
    )

@app.post("/analyze-ui", response_class=HTMLResponse)
def analyze_ui(
    request: Request,
    url: str = Form(...),
    question: str = Form(...),
    models: List[str] = Form(...)
):
    docs = load_documents(url)
    chunks = chunk_docs(docs)
    vector_store = build_vector_store(chunks)
    retriever = build_hybrid_retriever(chunks, vector_store)
    prompt = get_prompt()

    results = []

    for model in models:
        answer, context, latency, retrieved_docs = run_rag(
            retriever=retriever,
            prompt=prompt,
            question=question,
            model=model
        )

        metrics = evaluate(
            answer=answer,
            context=context,
            latency=latency
        )

        citations = sentence_citations(answer, retrieved_docs)
        hallucination_pct = hallucination_stats(citations)

        cost_info = estimate_cost(
        context=context,
        answer=answer,
        model=model)


        ragas_scores = compute_ragas_metrics(
        question=question,
        answer=answer,
        contexts=[doc.page_content for doc in retrieved_docs])


        results.append({
        "model": model,
        "answer": answer,
        "context": context,
        "latency": latency,
        "metrics": metrics,
        "ragas": ragas_scores,
        "cost": cost_info,
        "docs": retrieved_docs,
        "citations": citations,
        "hallucination_pct": hallucination_pct})


    rouge_scores = []
    if len(results) > 1:
        reference = results[0]
        for challenger in results[1:]:
            rouge = compute_rouge(
                reference["answer"],
                challenger["answer"]
            )
            rouge_scores.append({
                "model": challenger["model"],
                "rouge_l": rouge["rougeL"],
                "latency_delta": round(
                    challenger["latency"] - reference["latency"], 2
                )
            })

    fastest = min(results, key=lambda x: x["latency"])
    best_reasoning = max(results, key=lambda x: x["metrics"]["faithfulness"])

    chart_data = {
        "labels": [r["model"] for r in results],
        "latency": [round(r["latency"], 2) for r in results],
        "faithfulness": [r["metrics"]["faithfulness"] for r in results],
        "hallucination": [r["hallucination_pct"] for r in results]
    }

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "url": url,
            "question": question,
            "results": results,
            "rouge_scores": rouge_scores,
            "fastest": fastest["model"],
            "best_reasoning": best_reasoning["model"],
            "chart_data": chart_data
        }
    )

@app.post("/stream-answer")
def stream_answer(
    url: str = Form(...),
    question: str = Form(...),
    model: str = Form(...)
):
    docs = load_documents(url)
    chunks = chunk_docs(docs)
    vector_store = build_vector_store(chunks)
    retriever = build_hybrid_retriever(chunks, vector_store)
    prompt = get_prompt()

    generator = run_rag_stream(
        retriever=retriever,
        prompt=prompt,
        question=question,
        model=model
    )

    return StreamingResponse(generator, media_type="text/plain")
