# AI Subject Matter Expert

**Build a conversational AI expert on any subject using public internet data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Point it at any subject — quantum physics, climate change, divorce law, cooking — and it will research the internet, build a knowledge base, and give you a conversational expert that cites its sources.

**100% free to run** using [Groq](https://console.groq.com/keys) (Llama 3.3 70B) or [Google Gemini](https://aistudio.google.com/apikey) free tiers. No credit card required.

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
git clone https://github.com/albertgd/ai-subject-matter-expert.git
cd ai-subject-matter-expert
./setup.sh   # installs everything (including spaCy model)
./run.sh     # opens the app at http://localhost:8501
```

The app walks you through the rest — no terminal commands or file editing needed:

1. **Setup page** — pick a free LLM provider (Groq recommended), enter your API key, drill down to your subject using the 3-level preset browser
2. **Pipeline page** → click **Build Everything** — researches the web, processes docs, builds the KB
3. **Chat page** — ask questions, get grounded answers with cited sources

---

## Free API Keys

No paid API key is required. The two easiest free options:

| Provider | Model | Limit | Link |
|---|---|---|---|
| **Groq** | Llama 3.3 70B | Generous free tier, no credit card | [console.groq.com/keys](https://console.groq.com/keys) |
| **Google Gemini** | Gemini 2.0 Flash | Free tier via AI Studio | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

Paid providers (Anthropic Claude, OpenAI GPT-4) are also supported.

---

## Subject Presets

The Setup page includes **22 subject categories** with **3 levels of specialisation**. Click any level to use it, or drill down further:

```
⚖️ Law
  └── Family Law
        └── Divorce
        └── Child Custody & Support
        └── Adoption & Surrogacy
  └── Criminal Law
        └── Criminal Defence
        └── White Collar Crime
        └── Sentencing & Parole
  └── Employment & Labour Law
  └── Corporate & Commercial Law
  └── Immigration Law

🤖 Artificial Intelligence
  └── Machine Learning
        └── Deep Learning
        └── Reinforcement Learning
        └── MLOps
  └── Natural Language Processing
        └── Large Language Models
        └── ...
  └── Computer Vision
  └── AI Ethics & Safety
```

Other top-level categories: Cybersecurity, Data Science, Climate Change, Personal Finance, Nutrition & Health, Fitness & Exercise, Mental Health, Medicine & Healthcare, Space Exploration, Genetics & Genomics, History, Game Development, Cooking & Culinary Arts, Electric Vehicles, Ecology & Environment, Blockchain & Crypto, Music Theory & Production, Mobile App Development, Architecture & Engineering — each with sub-specialties.

You can also use **Custom** to enter any subject of your own.

---

## UI Pages

| Page | What it does |
|---|---|
| 🛠️ **Setup** | Pick LLM provider + API key + choose subject with 3-level preset drill-down |
| 🏠 **Home** | Status dashboard + one-click "Build Everything" |
| 💬 **Chat** | Ask questions, get grounded answers with cited sources |
| ⚙️ **Pipeline** | Run research / processing / KB build with real-time logs + reset button |
| 🔍 **Knowledge Base** | Search and browse indexed content |
| 📦 **Dataset** | View, download JSONL, upload to HuggingFace |
| 👤 **About** | Project info and author links |

---

## Project Structure

```
ai-subject-matter-expert/
├── src/
│   ├── app.py                       # Streamlit web UI (8 pages)
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
│       └── llm_factory.py           # Shared LLM builder — add new providers here
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

Copy `.env.example` to `.env` (or use the Setup page — it writes `.env` for you):

```bash
# Required — at least one LLM key (Groq is free)
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Subject (set via the Setup page or edit directly)
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

Already-fetched URLs are skipped automatically on subsequent runs.

### Step 2: Process

```bash
# Process only new docs (default — skips already-processed)
python scripts/process_data.py

# Force reprocess everything
python scripts/process_data.py --reprocess

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
python scripts/build_rag.py           # incremental (default)
python scripts/build_rag.py --rebuild # full rebuild
python scripts/build_rag.py --stats   # show counts
```

ChromaDB collections:

| Collection | Contents | Best for |
|---|---|---|
| `{subject}_documents` | Full text chunks | Detailed content search |
| `{subject}_learnings` | Key points + learnings | Quick insight lookup |
| `{subject}_summaries` | Document overviews | Topic browsing |

### Reset Everything

To wipe all data and start fresh, use the **🗑️ Danger Zone** section at the bottom of the Pipeline page. It deletes `data/raw/`, `data/processed/`, and `data/vector_db/` after a two-step confirmation.

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

## Adding a New LLM Provider

All provider logic lives in one place: `src/agents/llm_factory.py`. Add a new `elif` block there and it's automatically available across the entire pipeline — research, processing, and chat.

---

## Testing

```bash
pytest tests/ -v                      # full suite
pytest tests/test_research.py -v     # research module
pytest tests/test_processors.py -v   # text cleaner, PII, structurer
pytest tests/test_rag.py -v          # vector store, indexer, retriever
pytest tests/test_agent.py -v        # agent (mocked LLM)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM (free) | Groq (Llama 3.3 70B) / Google Gemini 2.0 Flash |
| LLM (paid) | Anthropic Claude / OpenAI GPT-4o |
| AI Web Research | LLM query generation + DuckDuckGo / Tavily |
| Wikipedia | wikipedia-api |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (free, multilingual) |
| Vector DB | ChromaDB (local, no server needed) |
| PII Removal | Microsoft Presidio + spaCy `en_core_web_lg` |
| Dataset | HuggingFace `datasets` |
| UI | Streamlit |
| Testing | pytest |

---

## License

MIT License.

Dataset (when published): CC BY 4.0. Content sourced from public internet sources.

---

## Author

**Albert García Díaz**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-albertgd-0077b5?logo=linkedin)](https://www.linkedin.com/in/albertgd)
[![X](https://img.shields.io/badge/X-albertgd-000000?logo=x)](https://twitter.com/albertgd)
[![GitHub](https://img.shields.io/badge/GitHub-albertgd-24292f?logo=github)](https://github.com/albertgd)
