🧠 RAGnalyze

Research-Grade RAG Evaluation, Grounding & Benchmarking Platform
RAGnalyze is a research-oriented Retrieval-Augmented Generation (RAG) benchmarking system designed to quantitatively evaluate LLM outputs beyond simple prompt → response workflows.
Unlike typical RAG demos, RAGnalyze focuses on measurement, grounding, and comparison, enabling controlled experiments across models using the same retrieved evidence.

🚀 Why RAGnalyze Exists

Most GenAI projects today:
Call the same OpenAI API
Use similar prompts
Judge output quality subjectively (“looks good”)

- RAGnalyze is different.
It answers:
Which model reasons better given the same evidence?
How grounded is each sentence in retrieved context?
How much hallucination risk exists?
How concise and efficient is the model’s reasoning?
This makes RAGnalyze a benchmarking and evaluation system, not just a chatbot.

🧩 Core Features
🔍 Hybrid Retrieval (Production-Grade)

BM25 (keyword) + Dense Embeddings

Manual hybrid retrieval (framework-agnostic, stable)
Deduplication & controlled context assembly

🧠 Controlled RAG Pipeline

Same context + same prompt across models
Eliminates evaluation bias
Enables fair model comparison

📊 Quantitative Evaluation (Rare in Student Projects)

Semantic Faithfulness Score (embedding-based)
Compression Ratio (answer efficiency)
Latency Tracking
ROUGE-L for model-to-model answer similarity

🧪 Model Benchmarking

Side-by-side evaluation (e.g. GPT-4o vs GPT-3.5)
Delta metrics (quality vs speed trade-offs)
Winner badges (Best Reasoning, Fastest Model)

🔬 Sentence-Level Grounding (Advanced)

Semantic similarity between each answer sentence and retrieved chunks
Sentence-level citations
Hallucination risk % per sentence
Transparent evidence inspection

🎛 Research-Style UI

Tabbed evaluation dashboard
Metrics, grounding, context, and comparison separated cleanly

Designed like an internal research tool, not a demo app

🧠 System Architecture

User Query
   ↓
Hybrid Retrieval (BM25 + Embeddings)
   ↓
Context Assembly
   ↓
LLM Generation (Per Model)
   ↓
Evaluation Pipeline
   ├── Faithfulness
   ├── Compression
   ├── ROUGE-L
   ├── Hallucination %
   ↓
Interactive Research Dashboard


📁 Project Structure
RAGnalyze/
│
├── app.py                     # Streamlit research dashboard
├── assets/
│   └── styles.css             # Custom UI styling
│
├── rag/
│   ├── ingestion.py           # YouTube + Web ingestion (fault-tolerant)
│   ├── chunking.py            # Recursive text splitting
│   ├── vector_store.py        # FAISS vector index
│   ├── retriever.py           # Manual hybrid retriever
│   ├── prompt_engine.py       # Strict system prompts
│   └── rag_pipeline.py        # Framework-agnostic RAG pipeline
│
├── evaluation/
│   ├── evaluator.py           # Unified evaluation logic
│   ├── answer_metrics.py      # ROUGE + compression
│   ├── grounding_metrics.py   # Semantic faithfulness
│   ├── citations.py           # Sentence-level grounding
│   └── hallucination.py       # Hallucination % estimation
│
├── comparison/
│   └── model_comparator.py    # Controlled model comparisons
│
├── requirements.txt
└── README.md

📊 Metrics Explained (Interview-Ready)
Metric	What it Measures	Why it Matters
Faithfulness	Semantic alignment between answer & context	Hallucination control
Compression Ratio	Answer length vs context length	Reasoning efficiency
ROUGE-L	Similarity vs reference model	Answer quality
Latency	End-to-end response time	Production readiness
Hallucination %	Unsupported sentences	Trustworthiness

🧪 Example Evaluation Output
Model: GPT-4o
Faithfulness: 0.81
Compression Ratio: 0.19
Hallucination Risk: 9%
Latency: 6.2s

Model: GPT-3.5
Faithfulness: 0.63
Compression Ratio: 0.27
Hallucination Risk: 28%
Latency: 2.1s

🧠 Key Design Decisions

Manual Hybrid Retrieval instead of fragile ensemble abstractions
Callable retrievers for framework independence
Embedding-based grounding instead of string matching
Strict data contracts using LangChain Document objects
Version-safe LangChain usage (invoke() over deprecated APIs)

🧑‍💻 Tech Stack
Python 3.10
Streamlit (Research UI)
LangChain (Core + Community)
OpenAI (LLMs & Embeddings)
FAISS (Vector Store)
ROUGE-Score
YouTube Transcript API
yt-dlp

🚀 Running Locally
conda create -n ragnalyze python=3.10
conda activate ragnalyze
pip install -r requirements.txt
streamlit run app.py

Set environment variable:
OPENAI_API_KEY=your_key_here

☁️ Deployment
Designed for Render:
No GPU required
No conda-only dependencies
Stable pip-based environment
Fault-tolerant ingestion

🧠 How to Describe This in Interviews (Use This)

“I built a research-grade RAG benchmarking system that evaluates grounding, hallucination risk, compression efficiency, and answer quality using sentence-level semantic citations under controlled retrieval settings.”
This is not a chatbot.
This is an evaluation system.

🔮 Future Roadmap
Token & cost dashboard
One-click PDF evaluation reports
Cross-encoder re-ranking
Prompt & metric tracking (MLflow)
SaaS benchmark platform for enterprise RAG systems

⭐ Why This Project Stands Out

Most people build RAG demos.
RAGnalyze builds RAG evaluation infrastructure.
That difference matters.