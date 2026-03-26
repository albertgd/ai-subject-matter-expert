# AI Subject Matter Expert

**Build a conversational AI trained on public court cases and legal documents**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A general-purpose pipeline for building AI subject matter experts from publicly available sources. Pre-configured for **divorce and family law**, but easily adapted to any legal domain.

---

## What It Does

```
Public Sources          Processing Pipeline            AI Expert
──────────────         ───────────────────            ─────────
CourtListener  ──►  Clean Text          ──►  ChromaDB     ──►  Chat
Harvard CAP    ──►  Remove PII          ──►  Semantic      ──►  Explore
Justia         ──►  LLM Structure       ──►  Index         ──►  UI
BAILII         ──►  Save JSON           ──►  HF Dataset    ──►  CLI
```

1. **Scrape** — Collect court opinions from free public APIs (CourtListener, Harvard Caselaw Access) and websites (Justia, BAILII)
2. **Clean** — Remove boilerplate, normalize text, fix encoding
3. **Anonymize PII** — Replace personal names/SSN/phone/email using Presidio + spaCy
4. **Structure** — LLM extracts facts, ruling, reasoning, and learnings for each case
5. **Index** — Store in ChromaDB with multilingual sentence-transformer embeddings
6. **Chat** — Conversational RAG agent cites real cases in every answer
7. **Share** — Upload cleaned dataset to HuggingFace Hub

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/albertgd/ai-subject-matter-expert.git
cd ai-subject-matter-expert
./setup.sh

# 2. Configure API keys
cp .env.example .env
# Edit .env — add at minimum ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 3. Collect a small batch of cases to test
source venv/bin/activate
python scripts/collect_data.py --max 20 --courtlistener

# 4. Process them (clean + PII + structure)
python scripts/process_data.py --limit 20 --fast-model

# 5. Build the knowledge base
python scripts/build_rag.py

# 6. Launch the web app
streamlit run src/app.py
# Open http://localhost:8501
```

---

## Data Sources

| Source | Type | URL | Auth |
|---|---|---|---|
| **CourtListener** | US federal & state court opinions | [courtlistener.com](https://www.courtlistener.com) | Optional token (free) |
| **Harvard Caselaw Access** | US cases 1658–present | [case.law](https://case.law) | Optional (free registration) |
| **Justia** | US case law & guides | [law.justia.com](https://law.justia.com) | None |
| **BAILII** | British & Irish cases | [bailii.org](https://www.bailii.org) | None |

---

## Project Structure

```
ai-subject-matter-expert/
├── src/
│   ├── app.py                    # Streamlit web UI (7 pages)
│   ├── config.py                 # All configuration (reads from .env)
│   ├── scrapers/
│   │   ├── base_scraper.py       # Base class with retry/rate-limit logic
│   │   ├── courtlistener_scraper.py  # CourtListener REST API
│   │   ├── caselaw_scraper.py        # Harvard Caselaw Access API
│   │   └── web_scraper.py            # Justia + BAILII HTML scraping
│   ├── processors/
│   │   ├── text_cleaner.py       # Boilerplate removal, normalization
│   │   ├── pii_remover.py        # Presidio + spaCy anonymization
│   │   └── case_structurer.py    # LLM extracts facts/ruling/learnings
│   ├── rag/
│   │   ├── vector_store.py       # ChromaDB collections manager
│   │   ├── indexer.py            # Loads processed cases into vector store
│   │   └── retriever.py          # Multi-collection semantic retrieval
│   └── agents/
│       └── sme_agent.py          # Conversational RAG agent + CLI
│
├── scripts/
│   ├── collect_data.py           # Run all scrapers
│   ├── process_data.py           # Clean + PII + LLM structure
│   ├── build_rag.py              # Index into ChromaDB
│   └── upload_to_hf.py           # Upload to HuggingFace Hub
│
├── data/
│   ├── raw/                      # Scraped JSON files (by source)
│   ├── processed/                # Cleaned + structured JSON files
│   ├── vector_db/                # ChromaDB persistent storage
│   └── hf_dataset/               # HuggingFace dataset output
│
├── tests/
│   ├── test_scrapers.py          # Scraper unit tests (offline)
│   ├── test_processors.py        # Processor unit tests
│   ├── test_rag.py               # RAG system tests
│   └── test_agent.py             # Agent tests (mocked LLM)
│
├── .env.example                  # Environment variables template
├── requirements.txt              # Python dependencies
├── setup.sh                      # One-command setup
└── run.sh                        # Interactive launcher
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required — at least one LLM key
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Optional — increases CourtListener rate limit
COURTLISTENER_API_TOKEN=

# Optional — full text access for Harvard Caselaw
CASELAW_API_KEY=

# Optional — only needed to upload dataset to HuggingFace
HF_TOKEN=hf_...
HF_REPO_ID=your-username/divorce-cases-en

# Optional — change the domain (default: divorce)
DOMAIN=divorce
DOMAIN_DESCRIPTION=divorce law and family law proceedings in the United States
SEARCH_KEYWORDS=divorce,custody,alimony,child support,marital property
```

---

## Pipeline in Detail

### Step 1: Collect Data

```bash
# All sources, 500 cases each (default)
python scripts/collect_data.py

# Specific sources
python scripts/collect_data.py --courtlistener --max 100
python scripts/collect_data.py --caselaw --max 100
python scripts/collect_data.py --web --max 50

# Output: data/raw/{source_name}/*.json
```

### Step 2: Process Data

```bash
# Full pipeline (clean + PII + LLM structure)
python scripts/process_data.py

# With options
python scripts/process_data.py --limit 200 --fast-model --skip-structured

# Skip LLM structuring (faster, but no extracted fields)
python scripts/process_data.py --no-structure

# Output: data/processed/*.json
```

**PII anonymization** (always applied):
- Personal names: `John Smith` → `[PERSON_1]`
- Phone numbers: `555-123-4567` → `[PHONE]`
- SSNs: `123-45-6789` → `[SSN]`
- Emails: `john@example.com` → `[EMAIL]`

**LLM structuring** (optional, uses API credits):
- `facts` — case background and relevant facts
- `ruling` — court's final decision
- `reasoning` — legal principles and doctrine applied
- `learnings` — 5-8 distilled legal knowledge points for RAG
- `summary` — 3-5 sentence overview
- `practice_areas` — topic tags (custody, alimony, etc.)

### Step 3: Build Knowledge Base

```bash
# Incremental (only new cases)
python scripts/build_rag.py

# Full rebuild
python scripts/build_rag.py --rebuild

# Check stats
python scripts/build_rag.py --stats
```

ChromaDB collections created:
| Collection | Contents | Best for |
|---|---|---|
| `{domain}_opinions` | Facts, reasoning, rulings (chunked) | Detailed case research |
| `{domain}_learnings` | Distilled legal principles | Quick principle lookup |
| `{domain}_summaries` | Case overviews | Topic browsing |

### Step 4: Use the Agent

**Web UI (Streamlit):**
```bash
streamlit run src/app.py
```

**CLI chat:**
```bash
python -m src.agents.sme_agent
# or
python -m src.agents.sme_agent "What is equitable distribution of property?"
```

**Python API:**
```python
from src.agents.sme_agent import SMEAgent

agent = SMEAgent()

# Single question
answer = agent.chat("What factors determine child custody?")
print(answer)

# With sources
answer, sources = agent.chat_with_sources("How is alimony calculated?")
for source in sources:
    print(f"  {source['title']} ({source['source_name']}) — score {source['score']:.2f}")

# Multi-turn conversation
agent.chat("What is the best interests standard?")
agent.chat("How does a parent's relocation affect this?")  # maintains context
agent.reset()  # start fresh
```

### Step 5: Upload to HuggingFace

```bash
# Requires HF_TOKEN in .env
python scripts/upload_to_hf.py --repo-id username/divorce-cases-en

# Private dataset
python scripts/upload_to_hf.py --repo-id username/repo --private

# Save locally without uploading
python scripts/upload_to_hf.py --local-only
```

Once uploaded:
```python
from datasets import load_dataset

ds = load_dataset("username/divorce-cases-en", split="train")
# Dataset({
#   features: ['source_id', 'title', 'source_name', 'url', 'date',
#              'court', 'text', 'facts', 'ruling', 'reasoning',
#              'learnings', 'summary', 'practice_areas', 'structured'],
#   num_rows: 500
# })
```

---

## Testing

```bash
# Full test suite
pytest tests/ -v

# Individual test files
pytest tests/test_scrapers.py -v    # Scraper unit tests
pytest tests/test_processors.py -v  # Processor tests
pytest tests/test_rag.py -v         # RAG system tests (creates temp ChromaDB)
pytest tests/test_agent.py -v       # Agent tests (mocked LLM)
```

---

## Adapting to a Different Domain

This system is domain-configurable. To adapt it for a different subject matter (e.g., patent law, criminal law, medical malpractice):

1. Edit `.env`:
   ```
   DOMAIN=patent
   DOMAIN_DESCRIPTION=patent law and intellectual property disputes
   SEARCH_KEYWORDS=patent infringement,prior art,claim construction,obviousness,PTAB
   ```

2. The scrapers will automatically use the new keywords.
3. The agent's system prompt will reflect the new domain.
4. The UI will update its labels and example questions.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Claude (Anthropic) / GPT-4o (OpenAI) / Gemini (Google) | Multi-provider support |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Free, no API key, multilingual |
| Vector DB | ChromaDB | Local, persistent, no server needed |
| PII removal | Microsoft Presidio + spaCy en_core_web_lg | State-of-art NER-based anonymization |
| Scraping | requests + BeautifulSoup + Playwright | Progressive enhancement |
| Dataset | HuggingFace `datasets` | Industry standard |
| UI | Streamlit | Rapid prototyping |
| Testing | pytest | Standard Python testing |

---

## Legal Disclaimer

This tool is for **educational and research purposes only**.
It does not constitute legal advice.
Always consult a qualified attorney for legal matters.
Case summaries and structured fields are generated by AI and may contain errors.

---

## License

MIT License. See [LICENSE](LICENSE).

Dataset (when published): CC BY 4.0. Source cases are public court records.

---

## Author

**Albert García Díaz** — [@albertgd](https://github.com/albertgd)

Related project: [legal-companion](https://github.com/albertgd/legal-companion) — Spanish family law AI
