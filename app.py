import os
import sys
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

load_dotenv()

from rag.ingestion import load_documents
from rag.chunking import chunk_docs
from rag.vector_store import build_vector_store
from rag.retriever import build_hybrid_retriever
from rag.prompt_engine import get_prompt
from rag.rag_pipeline import run_rag

from evaluation.evaluator import evaluate
from evaluation.answer_metrics import compute_rouge
from evaluation.citations import sentence_citations
from evaluation.hallucination import hallucination_stats

st.set_page_config(
    page_title="RAGnalyze",
    page_icon="🧠",
    layout="wide"
)
css_path = os.path.join(BASE_DIR, "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 RAGnalyze</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Research-Grade RAG Evaluation, Grounding & Benchmarking</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    url = st.text_input("🔗 YouTube / Website URL")
    question = st.text_area(
        "❓ Ask a question from the content",
        height=130,
        placeholder="e.g. What are the key takeaways?"
    )

    models = st.multiselect(
        "🤖 Models for comparison",
        ["gpt-4o", "gpt-3.5-turbo"],
        default=["gpt-4o"]
    )

    run = st.button("🚀 Run Analysis")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📌 What makes this different")
    st.write(
        """
        • Same context, same prompt → **controlled comparison**  
        • Sentence-level **semantic citations**  
        • **Hallucination % per sentence**  
        • ROUGE-L & compression metrics  
        • Designed as a **benchmarking system**, not a demo
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

if run:
    if not url or not question:
        st.error("Please provide both URL and question.")
        st.stop()

    with st.spinner("🔍 Ingesting content & building index..."):
        docs = load_documents(url)
        chunks = chunk_docs(docs)
        vector_store = build_vector_store(chunks)
        retriever = build_hybrid_retriever(chunks, vector_store)
        prompt = get_prompt()

    results = []

    with st.spinner("🧠 Running RAG pipelines..."):
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

            results.append({
                "model": model,
                "answer": answer,
                "context": context,
                "latency": latency,
                "metrics": metrics,
                "docs": retrieved_docs,
                "citations": citations,
                "hallucination_pct": hallucination_pct
            })

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Answers",
        "📊 Metrics",
        "🧪 Model Comparison",
        "📚 Retrieved Context",
        "🔍 Sentence Grounding"
    ])

    with tab1:
        cols = st.columns(len(results))
        for col, res in zip(cols, results):
            with col:
                st.markdown(f"### 🤖 {res['model']}")
                st.write(res["answer"])

    with tab2:
        cols = st.columns(len(results))
        for col, res in zip(cols, results):
            with col:
                st.markdown(f"### 📊 {res['model']}")
                st.metric("Faithfulness", res["metrics"]["faithfulness"])
                st.metric("Latency (s)", res["metrics"]["latency_sec"])
                st.metric("Compression Ratio", res["metrics"]["compression"])
                st.metric("Hallucination Risk (%)", res["hallucination_pct"])

    with tab3:
        if len(results) > 1:
            reference = results[0]  
            for challenger in results[1:]:
                rouge = compute_rouge(
                    reference["answer"],
                    challenger["answer"]
                )

                st.markdown(f"### ⚔️ {challenger['model']} vs {reference['model']}")
                st.metric("ROUGE-L (vs GPT-4o)", rouge["rougeL"])
                st.metric(
                    "Latency Δ (s)",
                    round(challenger["latency"] - reference["latency"], 2)
                )
        else:
            st.info("Select multiple models to see comparison.")

    with tab4:
        for i, doc in enumerate(results[0]["docs"], 1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content)

    with tab5:
        for res in results:
            st.markdown(f"### 🤖 {res['model']}")
            st.metric("Hallucination Risk (%)", res["hallucination_pct"])

            for i, c in enumerate(res["citations"], 1):
                if c["citations"]:
                    st.markdown(
                        f"✅ **Sentence {i}:** {c['sentence']}  \n"
                        f"Supported by chunks: {', '.join(str(x+1) for x in c['citations'])}"
                    )
                else:
                    st.markdown(
                        f"⚠️ **Sentence {i}:** {c['sentence']}  \n"
                        f"*No strong evidence found*"
                    )

    fastest = min(results, key=lambda x: x["latency"])
    best_reasoning = max(results, key=lambda x: x["metrics"]["faithfulness"])

    st.markdown("## 🏆 Model Winners")
    st.success(f"⚡ Fastest Model → {fastest['model']}")
    st.success(f"🧠 Best Reasoning → {best_reasoning['model']}")
