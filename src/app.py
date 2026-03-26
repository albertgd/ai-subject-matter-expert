"""
AI Subject Matter Expert — Streamlit Web Interface

Pages:
  1. Home          — Overview and quick stats
  2. Chat          — Conversational RAG agent
  3. Explore       — Browse knowledge base by topic
  4. Data Pipeline — Research, process, build RAG
  5. Dataset       — View and download processed dataset
  6. HuggingFace   — Upload dataset to HF Hub
  7. About         — Tech stack and architecture
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.config import (
    SUBJECT, SUBJECT_DESCRIPTION, SUBJECT_KEYWORDS,
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
    HF_TOKEN, HF_REPO_ID,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR,
)

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title=f"AI SME — {SUBJECT.title()}",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #1e3a8a; text-align: center; padding: 0.8rem 0;
    }
    .sub-header {
        text-align: center; color: #475569;
        font-size: 1.1rem; margin-bottom: 1.5rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border: 1px solid #c7d2fe;
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #1e3a8a; }
    .stat-label { font-size: 0.85rem; color: #64748b; margin-top: 0.2rem; }
    .source-card {
        background: #f8fafc; border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.4rem 0;
    }
    .step-box {
        background: #f0fdf4; border: 1px solid #86efac;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────
def count_files(directory: Path) -> int:
    try:
        return len(list(directory.rglob("*.json")))
    except Exception:
        return 0


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_agent():
    from src.agents.sme_agent import SMEAgent
    return SMEAgent()


@st.cache_resource(show_spinner="Loading vector store...")
def get_store():
    from src.rag.vector_store import VectorStore
    return VectorStore()


def run_script(cmd: list) -> tuple[str, int]:
    """Run a Python script and stream output."""
    result = subprocess.run(
        [sys.executable] + cmd,
        capture_output=True, text=True, cwd=ROOT,
    )
    return result.stdout + result.stderr, result.returncode


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 🧠 AI SME")
    st.markdown(f"**Subject:** {SUBJECT.title()}")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Home", "Chat", "Explore", "Data Pipeline", "Dataset", "HuggingFace", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    raw_count = count_files(RAW_DATA_DIR)
    proc_count = count_files(PROCESSED_DATA_DIR)
    st.caption(f"Raw docs: {raw_count:,}")
    st.caption(f"Processed: {proc_count:,}")

    llm_active = (
        "Anthropic" if ANTHROPIC_API_KEY else
        "Google" if GOOGLE_API_KEY else
        "OpenAI" if OPENAI_API_KEY else
        "None configured"
    )
    st.caption(f"LLM: {llm_active}")


# ══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════
if page == "Home":
    st.markdown(f'<div class="main-header">AI Subject Matter Expert</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">{SUBJECT_DESCRIPTION}</div>',
        unsafe_allow_html=True,
    )

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    raw_n = count_files(RAW_DATA_DIR)
    proc_n = count_files(PROCESSED_DATA_DIR)

    try:
        store = get_store()
        stats = store.stats()
        kb_total = sum(stats.values())
    except Exception:
        stats = {}
        kb_total = 0

    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{raw_n}</div>'
            f'<div class="stat-label">Raw Documents</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{proc_n}</div>'
            f'<div class="stat-label">Processed</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{kb_total:,}</div>'
            f'<div class="stat-label">Knowledge Base Chunks</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        status = "Ready" if kb_total > 0 else "Empty"
        color = "#16a34a" if kb_total > 0 else "#dc2626"
        st.markdown(
            f'<div class="stat-card"><div class="stat-number" style="color:{color}">{status}</div>'
            f'<div class="stat-label">Knowledge Base</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("How it works")
    cols = st.columns(4)
    steps = [
        ("1. Research", "AI generates smart queries, searches the web and Wikipedia for your subject"),
        ("2. Process", "Clean text, remove PII, extract structured knowledge with LLM"),
        ("3. Build RAG", "Index into ChromaDB vector store (3 collections)"),
        ("4. Chat", "Ask questions — get grounded answers with sources"),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="step-box"><strong>{title}</strong><br/>{desc}</div>', unsafe_allow_html=True)

    if stats:
        st.markdown("---")
        st.subheader("Knowledge Base Collections")
        cols = st.columns(3)
        for col, (name, count) in zip(cols, stats.items()):
            with col:
                st.metric(name.title(), f"{count:,} chunks")

    st.markdown("---")
    st.subheader("Subject Keywords")
    st.write(" • ".join(SUBJECT_KEYWORDS))


# ══════════════════════════════════════════════════════════
# PAGE 2 — CHAT
# ══════════════════════════════════════════════════════════
elif page == "Chat":
    st.title(f"Chat with your {SUBJECT.title()} Expert")

    if not (ANTHROPIC_API_KEY or OPENAI_API_KEY or GOOGLE_API_KEY):
        st.error("No LLM API key configured. Add ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY to .env")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None

    # Agent init
    if st.session_state.agent is None:
        with st.spinner("Loading agent..."):
            try:
                from src.agents.sme_agent import SMEAgent
                st.session_state.agent = SMEAgent()
                if not st.session_state.agent.is_ready() if hasattr(st.session_state.agent, 'is_ready') else not st.session_state.agent._kb_ready:
                    st.info(
                        "Knowledge base is empty — agent will use general knowledge only. "
                        "Go to **Data Pipeline** to build it."
                    )
            except Exception as e:
                st.error(f"Failed to load agent: {e}")
                st.stop()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        url = src.get("url", "")
                        title = src.get("title", "Unknown")
                        link = f"[{title}]({url})" if url else title
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>{link}</strong><br/>'
                            f'From: {src.get("source_name","")} | '
                            f'Score: {src.get("score", 0):.3f}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Input
    if prompt := st.chat_input(f"Ask anything about {SUBJECT}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.agent.chat_with_sources(prompt)
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"Sources ({len(sources)})"):
                            for src in sources:
                                url = src.get("url", "")
                                title = src.get("title", "Unknown")
                                link = f"[{title}]({url})" if url else title
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<strong>{link}</strong><br/>'
                                    f'From: {src.get("source_name","")} | '
                                    f'Score: {src.get("score", 0):.3f}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        if st.session_state.agent:
            st.session_state.agent.reset()
        st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE 3 — EXPLORE
# ══════════════════════════════════════════════════════════
elif page == "Explore":
    st.title("Explore the Knowledge Base")

    try:
        store = get_store()
        stats = store.stats()
        total = sum(stats.values())
    except Exception as e:
        st.error(f"Could not load vector store: {e}")
        st.stop()

    if total == 0:
        st.warning("Knowledge base is empty. Go to **Data Pipeline** to build it.")
        st.stop()

    st.subheader("Search")
    query = st.text_input("Search query", placeholder=f"Enter a topic related to {SUBJECT}...")
    col1, col2, col3 = st.columns(3)
    with col1:
        k_docs = st.slider("Document chunks", 1, 20, 5)
    with col2:
        collection = st.selectbox("Collection", ["documents", "learnings", "summaries"])

    if query:
        with st.spinner("Searching..."):
            results = store.search_with_score(query, k=k_docs, collection=collection)

        st.markdown(f"**{len(results)} results** in `{collection}`")
        for doc, score in results:
            meta = doc.metadata
            url = meta.get("url", "")
            title = meta.get("title", "Unknown")
            with st.expander(f"{title} (score: {score:.3f})"):
                if url:
                    st.markdown(f"[{meta.get('source_name', url)}]({url})")
                st.caption(f"Date: {meta.get('date', 'N/A')} | Topics: {meta.get('topics', '')}")
                st.text(doc.page_content[:800])

    st.markdown("---")
    st.subheader("Collection Statistics")
    cols = st.columns(3)
    for col, (name, count) in zip(cols, stats.items()):
        with col:
            st.metric(name.title(), f"{count:,} chunks")


# ══════════════════════════════════════════════════════════
# PAGE 4 — DATA PIPELINE
# ══════════════════════════════════════════════════════════
elif page == "Data Pipeline":
    st.title("Data Pipeline")
    st.markdown(f"Build the knowledge base for **{SUBJECT}** from public internet sources.")

    # Status
    raw_n = count_files(RAW_DATA_DIR)
    proc_n = count_files(PROCESSED_DATA_DIR)
    try:
        store = get_store()
        kb_stats = store.stats()
        kb_total = sum(kb_stats.values())
    except Exception:
        kb_stats = {}
        kb_total = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Raw Documents", raw_n)
    with col2:
        st.metric("Processed", proc_n)
    with col3:
        st.metric("KB Chunks", kb_total)

    st.markdown("---")

    # Step 1: Research
    st.subheader("Step 1 — Research")
    st.markdown(
        "AI generates search queries → searches the web and Wikipedia → fetches and saves pages."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        max_docs = st.number_input("Max docs per source", 10, 1000, 100, step=10)
    with col2:
        n_queries = st.number_input("AI search queries", 5, 30, 10)
    with col3:
        research_source = st.selectbox("Source", ["All", "Web (AI)", "Wikipedia"])

    if st.button("Run Research", type="primary"):
        cmd = ["scripts/research.py", "--max", str(max_docs), "--queries", str(n_queries)]
        if research_source == "Web (AI)":
            cmd.append("--web")
        elif research_source == "Wikipedia":
            cmd.append("--wikipedia")

        with st.spinner("Researching..."):
            output, code = run_script(cmd)
        if code == 0:
            st.success("Research complete!")
        else:
            st.error("Research failed (see output)")
        st.code(output)
        st.rerun()

    st.markdown("---")

    # Step 2: Process
    st.subheader("Step 2 — Process")
    st.markdown("Clean text, remove PII, extract structured knowledge with LLM.")
    col1, col2 = st.columns(2)
    with col1:
        proc_limit = st.number_input("Limit (0 = all)", 0, 10000, 0, step=10)
    with col2:
        use_fast_model = st.checkbox("Use fast model (cheaper)")
    skip_struct = st.checkbox("Skip already-structured documents")
    no_struct = st.checkbox("Skip LLM structuring (clean + PII only, much faster)")

    if st.button("Run Processing", type="primary"):
        cmd = ["scripts/process_data.py"]
        if proc_limit > 0:
            cmd += ["--limit", str(proc_limit)]
        if use_fast_model:
            cmd.append("--fast-model")
        if skip_struct:
            cmd.append("--skip-structured")
        if no_struct:
            cmd.append("--no-structure")

        with st.spinner("Processing..."):
            output, code = run_script(cmd)
        if code == 0:
            st.success("Processing complete!")
        else:
            st.error("Processing failed (see output)")
        st.code(output)
        st.rerun()

    st.markdown("---")

    # Step 3: Build RAG
    st.subheader("Step 3 — Build Knowledge Base")
    st.markdown("Index processed documents into ChromaDB for semantic search.")
    rebuild = st.checkbox("Force full rebuild (delete existing index)")

    if st.button("Build Knowledge Base", type="primary"):
        cmd = ["scripts/build_rag.py"]
        if rebuild:
            cmd.append("--rebuild")

        with st.spinner("Building..."):
            output, code = run_script(cmd)
        if code == 0:
            st.success("Knowledge base built!")
        else:
            st.error("Build failed (see output)")
        st.code(output)
        st.cache_resource.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE 5 — DATASET
# ══════════════════════════════════════════════════════════
elif page == "Dataset":
    st.title("Dataset Viewer")

    docs = []
    for path in sorted(PROCESSED_DATA_DIR.rglob("*.json")):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

    if not docs:
        st.warning("No processed documents yet. Run the Data Pipeline first.")
        st.stop()

    st.metric("Total Documents", len(docs))

    # Search filter
    search = st.text_input("Filter by title or topic", "")
    if search:
        docs = [
            d for d in docs
            if search.lower() in d.get("title", "").lower()
            or search.lower() in ",".join(d.get("topics", [])).lower()
        ]
        st.caption(f"{len(docs)} matching documents")

    # Table view
    rows = []
    for d in docs[:200]:
        rows.append({
            "Title": d.get("title", "")[:80],
            "Source": d.get("source_name", ""),
            "Date": d.get("date", ""),
            "Topics": ", ".join(d.get("topics", [])),
            "Structured": "Yes" if d.get("structured") else "No",
        })

    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Detail view
    st.markdown("---")
    titles = [d.get("title", d.get("source_id", "?"))[:80] for d in docs[:100]]
    selected = st.selectbox("View document detail", titles)
    if selected:
        doc = next((d for d in docs if d.get("title", "").startswith(selected[:40])), None)
        if doc:
            st.subheader(doc.get("title", ""))
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Source:** {doc.get('source_name', '')}")
                st.write(f"**Date:** {doc.get('date', 'N/A')}")
                st.write(f"**URL:** {doc.get('url', '')}")
            with col2:
                st.write(f"**Topics:** {', '.join(doc.get('topics', []))}")
                st.write(f"**Author:** {doc.get('author', 'N/A')}")
                st.write(f"**Structured:** {'Yes' if doc.get('structured') else 'No'}")

            if doc.get("summary"):
                st.markdown("**Summary:**")
                st.write(doc["summary"])
            if doc.get("key_points"):
                st.markdown("**Key Points:**")
                st.write(doc["key_points"])
            if doc.get("learnings"):
                st.markdown("**Learnings:**")
                st.write(doc["learnings"])

    # Download
    st.markdown("---")
    if st.button("Download as JSONL"):
        jsonl = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs)
        st.download_button(
            "Download JSONL",
            data=jsonl,
            file_name=f"{SUBJECT.replace(' ', '_')}_dataset.jsonl",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════
# PAGE 6 — HUGGINGFACE
# ══════════════════════════════════════════════════════════
elif page == "HuggingFace":
    st.title("Upload to HuggingFace Hub")

    if not HF_TOKEN:
        st.error(
            "HF_TOKEN not set in .env. "
            "Get a token from huggingface.co/settings/tokens"
        )

    repo_id = st.text_input("Repository ID", value=HF_REPO_ID or f"username/{SUBJECT.replace(' ', '-')}-sme")
    private = st.checkbox("Private repository", value=False)
    no_full_text = st.checkbox("Exclude full text (smaller dataset)", value=False)

    proc_n = count_files(PROCESSED_DATA_DIR)
    st.info(f"{proc_n} processed documents ready to upload.")

    if st.button("Upload to HuggingFace", type="primary", disabled=not HF_TOKEN):
        cmd = ["scripts/upload_to_hf.py", "--repo-id", repo_id]
        if private:
            cmd.append("--private")
        if no_full_text:
            cmd.append("--no-full-text")

        with st.spinner("Uploading..."):
            output, code = run_script(cmd)
        if code == 0:
            st.success(f"Uploaded! View at https://huggingface.co/datasets/{repo_id}")
        else:
            st.error("Upload failed (see output)")
        st.code(output)


# ══════════════════════════════════════════════════════════
# PAGE 7 — ABOUT
# ══════════════════════════════════════════════════════════
elif page == "About":
    st.title("About This Project")

    st.markdown(f"""
## AI Subject Matter Expert

A fully generic system to build a domain-specific AI expert on **any subject** using public internet data.

### Current Subject
**{SUBJECT.title()}** — {SUBJECT_DESCRIPTION}

### Architecture

| Component | Technology |
|-----------|-----------|
| Web Research | DuckDuckGo / Tavily + AI query generation |
| Wikipedia | wikipedia-api |
| Text Cleaning | Custom rule-based cleaner |
| PII Removal | Microsoft Presidio + spaCy |
| Knowledge Extraction | LLM (Claude / GPT-4 / Gemini) |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (multilingual) |
| Agent | LangChain + RAG |
| UI | Streamlit |
| Dataset Publishing | HuggingFace Hub |

### Pipeline

```
Internet  →  AI Researcher  →  Raw Docs
                                    ↓
                              TextCleaner
                                    ↓
                              PIIRemover
                                    ↓
                          ContentStructurer (LLM)
                                    ↓
                           ChromaDB (3 collections)
                                    ↓
                           SME Agent (RAG + LLM)
                                    ↓
                            HuggingFace Dataset
```

### Changing the Subject

Edit your `.env` file:
```
SUBJECT=quantum physics
SUBJECT_DESCRIPTION=quantum mechanics, quantum computing, and quantum field theory
SUBJECT_KEYWORDS=quantum,entanglement,superposition,qubit,Schrodinger
```

Then run the full pipeline again.
""")
