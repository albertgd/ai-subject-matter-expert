# AI Subject Matter Expert

**Build a conversational AI expert on any subject using public internet data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Point it at any subject — quantum physics, climate change, personal finance, cooking — and it will research the internet, build a knowledge base, and give you a conversational expert that cites its sources.

---

## What It Does

```
Internet                  Processing Pipeline              AI Expert
────────                  ───────────────────              ─────────
DuckDuckGo / Tavily  ──►  Clean Text          ──►  ChromaDB     ──►  Chat
Wikipedia            ──►  Remove PII          ──►  Semantic      ──►  Explore
Any website          ──►  LLM Structure       ──►  Index         ──►  UI
                     ──►  Save JSON           ──►  HF Dataset    ──►  CLI
```

1. **Research** — AI generates smart search queries → searches the web (DuckDuckGo/Tavily) → fetches pages + Wikipedia articles
2. **Clean** — Remove boilerplate, normalize text, fix encoding
3. **Anonymize PII** — Replace personal names/SSN/phone/email using Presidio + spaCy
4. **Structure** — LLM extracts summary, key points, learnings, and topic tags from each document
5. **Index** — Store in ChromaDB with multilingual sentence-transformer embeddings
6. **Chat** — Conversational RAG agent cites real sources in every answer
7. **Share** — Upload cleaned dataset to HuggingFace Hub

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/albertgd/ai-subject-matter-expert.git
cd ai-subject-matter-expert
./setup.sh

# 2. Configure
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (or OPENAI/GOOGLE) and your SUBJECT

# 3. Research your subject
source venv/bin/activate
python scripts/research.py --max 50

# 4. Process (clean + PII + structure)
python scripts/process_data.py --limit 50 --fast-model

# 5. Build the knowledge base
python scripts/build_rag.py

# 6. Launch the app
streamlit run src/app.py
# Open http://localhost:8501
```

---

## Changing the Subject

The only thing you need to change is your `.env`:

```bash
SUBJECT=quantum physics
SUBJECT_DESCRIPTION=quantum mechanics, quantum computing, and quantum field theory
SUBJECT_KEYWORDS=quantum,entanglement,superposition,qubit,Schrodinger,Heisenberg
```

Then run the pipeline again. Everything — research queries, agent prompts, UI labels, dataset name — adapts automatically.

**More examples:**

```bash
# Climate change
SUBJECT=climate change
SUBJECT_DESCRIPTION=climate science, global warming, and environmental policy
SUBJECT_KEYWORDS=climate change,global warming,CO2,greenhouse gas,IPCC,carbon

# Personal finance
SUBJECT=personal finance
SUBJECT_DESCRIPTION=personal finance, investing, budgeting, and wealth management
SUBJECT_KEYWORDS=investing,stocks,bonds,ETF,budgeting,savings,retirement,401k

# Divorce law (original use case)
SUBJECT=divorce law
SUBJECT_DESCRIPTION=divorce and family law proceedings
SUBJECT_KEYWORDS=divorce,custody,alimony,child support,marital property
```

---

## Project Structure

```
ai-subject-matter-expert/
├── src/
│   ├── app.py                       # Streamlit web UI (7 pages)
│   ├── config.py                    # All configuration (reads from .env)
│   ├── research/
│   │   ├── base.py                  # Base collector (retry, save, dedup)
│   │   ├── ai_researcher.py         # LLM generates queries → search → fetch
│   │   ├── search.py                # DuckDuckGo (free) + Tavily adapters
│   │   ├── web_fetcher.py           # Generic URL fetcher + text extractor
│   │   └── wikipedia.py             # Wikipedia API source
│   ├── processors/
│   │   ├── text_cleaner.py          # Boilerplate removal, normalization
│   │   ├── pii_remover.py           # Presidio + spaCy anonymization
│   │   └── content_structurer.py    # LLM extracts summary/key_points/topics
│   ├── rag/
│   │   ├── vector_store.py          # ChromaDB collections manager
│   │   ├── indexer.py               # Loads processed docs into vector store
│   │   └── retriever.py             # Multi-collection semantic retrieval
│   └── agents/
│       ├── sme_agent.py             # Conversational RAG agent + CLI
│       └── llm_factory.py           # Shared LLM builder (Anthropic/OpenAI/Google)
│
├── scripts/
│   ├── research.py                  # Run AI-powered web research
│   ├── process_data.py              # Clean + PII + LLM structure
│   ├── build_rag.py                 # Index into ChromaDB
│   └── upload_to_hf.py              # Upload to HuggingFace Hub
│
├── data/
│   ├── raw/                         # Fetched JSON files (by source)
│   ├── processed/                   # Cleaned + structured JSON files
│   ├── vector_db/                   # ChromaDB persistent storage
│   └── hf_dataset/                  # HuggingFace dataset output
│
├── tests/
│   ├── conftest.py                  # Shared fixtures (FakeEmbeddings)
│   ├── test_research.py             # Research module unit tests
│   ├── test_processors.py           # Processor unit tests
│   ├── test_rag.py                  # RAG system tests
│   └── test_agent.py                # Agent tests (mocked LLM)
│
├── .env.example                     # Environment variables template
├── requirements.txt                 # Python dependencies
├── setup.sh                         # One-command setup
└── run.sh                           # Interactive launcher
```

---

## Configuration

Copy `.env.example` to `.env`:

```bash
# Required — at least one LLM key
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Subject (the main thing to configure)
SUBJECT=artificial intelligence
SUBJECT_DESCRIPTION=artificial intelligence, machine learning, and deep learning
SUBJECT_KEYWORDS=artificial intelligence,machine learning,deep learning,neural networks,LLM

# Search (DuckDuckGo is free and needs no key; Tavily gives better quality)
TAVILY_API_KEY=          # optional

# HuggingFace (only needed to upload dataset)
HF_TOKEN=hf_...
HF_REPO_ID=your-username/your-subject-sme
```

---

## Pipeline in Detail

### Step 1: Research

```bash
# All sources (web + Wikipedia)
python scripts/research.py

# Specific sources
python scripts/research.py --web          # AI web researcher only
python scripts/research.py --wikipedia    # Wikipedia only

# Control volume
python scripts/research.py --max 100 --queries 15

# Output: data/raw/{source}/*.json
```

The AI researcher:
1. Asks the LLM to generate N diverse search queries for your subject
2. Searches the web (DuckDuckGo/Tavily) for each query
3. Scores results for relevance using keyword matching
4. Fetches and cleans the top pages

### Step 2: Process

```bash
# Full pipeline (clean + PII + LLM structure)
python scripts/process_data.py

# With options
python scripts/process_data.py --limit 200 --fast-model --skip-structured

# Skip LLM structuring (faster, no API credits used)
python scripts/process_data.py --no-structure

# Output: data/processed/*.json
```

**PII anonymization** (always applied):
- Names: `John Smith` → `[PERSON_1]`
- Phones: `555-123-4567` → `[PHONE]`
- SSNs: `123-45-6789` → `[SSN]`
- Emails: `john@example.com` → `[EMAIL]`

**LLM structuring** (optional, uses API credits):
- `summary` — 3-5 sentence overview
- `key_points` — 5-8 most important facts/concepts
- `learnings` — reusable knowledge for RAG retrieval
- `topics` — subject tag list

### Step 3: Build Knowledge Base

```bash
python scripts/build_rag.py           # incremental
python scripts/build_rag.py --rebuild # full rebuild
python scripts/build_rag.py --stats   # show counts
```

ChromaDB collections:

| Collection | Contents | Best for |
|---|---|---|
| `{subject}_documents` | Full text chunks | Detailed content search |
| `{subject}_learnings` | Key points + learnings | Quick insight lookup |
| `{subject}_summaries` | Document overviews | Topic browsing |

### Step 4: Use the Agent

**Web UI:**
```bash
streamlit run src/app.py
```

**CLI:**
```bash
python -m src.agents.sme_agent
python -m src.agents.sme_agent "What is backpropagation?"
```

**Python API:**
```python
from src.agents.sme_agent import SMEAgent

agent = SMEAgent()

answer = agent.chat("Explain transformer attention mechanisms")
answer, sources = agent.chat_with_sources("How does BERT work?")

for source in sources:
    print(f"  [{source['source_name']}] {source['title']} — score {source['score']:.2f}")

# Multi-turn
agent.chat("What is gradient descent?")
agent.chat("How does it relate to backpropagation?")  # maintains context
agent.reset()
```

### Step 5: Upload to HuggingFace

```bash
python scripts/upload_to_hf.py --repo-id username/my-subject-dataset
python scripts/upload_to_hf.py --repo-id username/repo --private
python scripts/upload_to_hf.py --local-only   # save without uploading
```

```python
from datasets import load_dataset

ds = load_dataset("username/my-subject-dataset", split="train")
# Features: source_id, title, source_name, url, date, author,
#           text, summary, key_points, learnings, topics, structured
```

---

## Testing

```bash
pytest tests/ -v                      # full suite (47 tests)
pytest tests/test_research.py -v     # research module
pytest tests/test_processors.py -v   # text cleaner, PII, structurer
pytest tests/test_rag.py -v          # vector store, indexer, retriever
pytest tests/test_agent.py -v        # agent (mocked LLM)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude (Anthropic) / GPT-4o (OpenAI) / Gemini (Google) |
| AI Web Research | LLM query generation + DuckDuckGo / Tavily |
| Wikipedia | wikipedia-api |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (free, multilingual) |
| Vector DB | ChromaDB (local, no server needed) |
| PII Removal | Microsoft Presidio + spaCy |
| Dataset | HuggingFace `datasets` |
| UI | Streamlit |
| Testing | pytest |

---

## License

MIT License.

Dataset (when published): CC BY 4.0. Content sourced from public internet sources.

---

## Author

**Albert García Díaz** — [@albertgd](https://github.com/albertgd)
