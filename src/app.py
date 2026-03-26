"""
AI Subject Matter Expert — Streamlit Web Interface

Pages:
  1. Home          — Overview and quick stats
  2. Chat          — Conversational RAG agent
  3. Explore       — Browse cases by topic
  4. Data Pipeline — Scrape, process, build RAG
  5. Dataset       — View and download processed dataset
  6. HuggingFace   — Upload dataset to HF Hub
  7. About         — Tech stack and architecture
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.config import (
    DOMAIN, DOMAIN_DESCRIPTION, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    GOOGLE_API_KEY, HF_TOKEN, HF_REPO_ID,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR,
    SEARCH_KEYWORDS,
)


# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title=f"AI {DOMAIN.title()} SME",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        padding: 0.8rem 0;
    }
    .sub-header {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #c7d2fe;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .source-card {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #78350f;
    }
    div[data-testid="stButton"] > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────
def _check_api_keys() -> bool:
    return bool(ANTHROPIC_API_KEY or OPENAI_API_KEY or GOOGLE_API_KEY)


def _get_agent():
    """Initialize SME agent once per session."""
    if "sme_agent" not in st.session_state:
        with st.spinner("Loading knowledge base and AI model..."):
            from src.agents.sme_agent import SMEAgent
            st.session_state.sme_agent = SMEAgent()
            st.session_state.chat_messages = []
    return st.session_state.sme_agent


def _get_kb_stats():
    """Get knowledge base stats (cached in session)."""
    if "kb_stats" not in st.session_state:
        try:
            from src.rag.vector_store import VectorStore
            store = VectorStore()
            st.session_state.kb_stats = store.stats()
        except Exception:
            st.session_state.kb_stats = {"opinions": 0, "learnings": 0, "summaries": 0}
    return st.session_state.kb_stats


def _count_files(directory: Path, pattern: str = "*.json") -> int:
    try:
        return len(list(directory.rglob(pattern)))
    except Exception:
        return 0


def _render_sources(sources: list):
    """Render retrieved source cards in an expander."""
    if not sources:
        return
    with st.expander(f"Sources consulted ({len(sources)})", expanded=False):
        for s in sources:
            st.markdown(
                f'<div class="source-card">'
                f'<strong>{s["title"][:80]}</strong><br>'
                f'<small>{s["source_name"]} · {s["date"]} · {s["court"][:60]}</small><br>'
                f'<small>Relevance: {s["score"]:.2f} · Section: {s["section"]}</small>'
                + (f'<br><a href="{s["url"]}" target="_blank">View original</a>' if s.get("url") else "")
                + '</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════
def page_home():
    st.markdown(f'<div class="main-header">AI {DOMAIN.title()} Subject Matter Expert</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Conversational AI trained on public court cases · {DOMAIN_DESCRIPTION.title()}</div>',
        unsafe_allow_html=True,
    )

    # ── Stats row ──────────────────────────────────────
    kb_stats = _get_kb_stats()
    raw_count = _count_files(RAW_DATA_DIR)
    processed_count = _count_files(PROCESSED_DATA_DIR)
    total_chunks = sum(kb_stats.values())

    col1, col2, col3, col4 = st.columns(4)
    for col, number, label in [
        (col1, raw_count, "Scraped Cases"),
        (col2, processed_count, "Processed Cases"),
        (col3, total_chunks, "Knowledge Base Chunks"),
        (col4, kb_stats.get("learnings", 0), "Legal Principles"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-number">{number:,}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("What This System Does")
        st.markdown(f"""
        This AI Subject Matter Expert was built by collecting thousands of public
        **{DOMAIN} law** court cases from free, open sources:

        - **CourtListener** — US federal & state court opinions (free API)
        - **Harvard Caselaw Access** — historical US cases (free API)
        - **Justia** — US case law & legal guides
        - **BAILII** — British and Irish legal decisions

        For each case, the system:
        1. **Scrapes** raw text from the source
        2. **Cleans** and normalizes the text
        3. **Removes PII** using Microsoft Presidio + spaCy
        4. **Structures** it with an LLM (facts, ruling, reasoning, learnings)
        5. **Indexes** into ChromaDB for semantic search
        6. **Answers** your questions, citing real cases
        """)

    with col_b:
        st.subheader("How to Use")
        st.markdown("""
        **1. Ask questions in the Chat tab**
        The AI retrieves relevant cases and principles from the knowledge base,
        then generates a grounded, cited answer.

        **2. Explore cases by topic**
        Browse the knowledge base by practice area or keyword.

        **3. Build/update the knowledge base**
        Use the Data Pipeline tab to scrape new cases, process them, and rebuild the RAG index.

        **4. Share the dataset**
        Upload the cleaned, PII-removed dataset to HuggingFace for others to use.
        """)

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">DISCLAIMER: This tool is for educational and research purposes only. '
        'It does not constitute legal advice. Always consult a qualified attorney for legal matters.</div>',
        unsafe_allow_html=True,
    )

    # ── Quick links ────────────────────────────────────
    st.markdown("---")
    st.subheader("Quick Start")
    st.markdown(f"""
    ```bash
    # 1. Set up
    ./setup.sh

    # 2. Add your API key to .env
    ANTHROPIC_API_KEY=sk-ant-...

    # 3. Collect data from public sources (start with small batch)
    python scripts/collect_data.py --max 50

    # 4. Process + clean data
    python scripts/process_data.py

    # 5. Build the RAG knowledge base
    python scripts/build_rag.py

    # 6. Launch this app
    streamlit run src/app.py
    ```
    """)


# ════════════════════════════════════════════════════════
# PAGE: CHAT
# ════════════════════════════════════════════════════════
def page_chat():
    st.header(f"Chat with the {DOMAIN.title()} Law Expert")
    st.markdown(
        f"Ask any question about {DOMAIN_DESCRIPTION}. "
        "The AI retrieves real court cases from the knowledge base and cites them in its answer."
    )

    if not _check_api_keys():
        st.error("No LLM API key configured. Add ANTHROPIC_API_KEY (or OPENAI/GOOGLE) to your .env file.")
        return

    # ── Sidebar controls ───────────────────────────────
    with st.sidebar:
        st.subheader("Search Settings")
        k_opinions = st.slider("Case law chunks", min_value=1, max_value=10, value=4)
        k_learnings = st.slider("Legal principles", min_value=1, max_value=6, value=3)

        if st.button("New conversation", use_container_width=True):
            if "sme_agent" in st.session_state:
                st.session_state.sme_agent.reset()
            st.session_state.chat_messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("**Example Questions**")
        examples = [
            f"What factors do courts consider for child custody?",
            f"How is property divided in a divorce?",
            f"When is alimony (spousal support) awarded?",
            f"What is the 'best interests of the child' standard?",
            f"Can a divorce settlement be modified after the fact?",
            f"What constitutes marital vs. separate property?",
            f"How do courts handle relocation with children post-divorce?",
        ]
        for ex in examples:
            if st.button(ex[:50] + "..." if len(ex) > 50 else ex, key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state["chat_prefill"] = ex
                st.rerun()

        st.markdown("---")
        kb_stats = _get_kb_stats()
        st.caption(f"KB: {sum(kb_stats.values()):,} chunks indexed")

    # ── Load agent ─────────────────────────────────────
    try:
        agent = _get_agent()
        agent.k_opinions = k_opinions
        agent.k_learnings = k_learnings
    except Exception as e:
        st.error(f"Failed to load agent: {e}")
        return

    if not agent._kb_ready:
        st.warning(
            "The knowledge base is empty. Run `python scripts/build_rag.py` to populate it. "
            "The agent will still answer from its training data but won't cite specific cases."
        )

    # ── Chat history ───────────────────────────────────
    for msg in st.session_state.get("chat_messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Input ──────────────────────────────────────────
    prefill = st.session_state.pop("chat_prefill", "")
    user_input = st.chat_input(f"Ask about {DOMAIN} law...") or prefill

    if user_input:
        # Show user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching case law and generating answer..."):
                try:
                    answer, sources = agent.chat_with_sources(user_input)
                    st.markdown(answer)
                    _render_sources(sources)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = f"Error generating answer: {e}"
                    st.error(err)
                    st.session_state.chat_messages.append({"role": "assistant", "content": err, "sources": []})


# ════════════════════════════════════════════════════════
# PAGE: EXPLORE
# ════════════════════════════════════════════════════════
def page_explore():
    st.header("Explore the Knowledge Base")
    st.markdown("Browse cases by topic, practice area, or keyword.")

    kb_stats = _get_kb_stats()
    if sum(kb_stats.values()) == 0:
        st.warning("Knowledge base is empty. Run `python scripts/build_rag.py` first.")
        return

    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Opinion Chunks", kb_stats.get("opinions", 0))
    col2.metric("Legal Principles", kb_stats.get("learnings", 0))
    col3.metric("Case Summaries", kb_stats.get("summaries", 0))

    st.markdown("---")

    col_search, col_n = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("Search topic", placeholder="e.g. child custody relocation")
    with col_n:
        n_results = st.number_input("Results", min_value=3, max_value=30, value=10)

    # Practice area quick filters
    st.markdown("**Quick filters:**")
    practice_areas = [
        "All", "custody", "alimony", "property division", "child support",
        "spousal support", "modification", "enforcement", "domestic violence",
    ]
    cols = st.columns(len(practice_areas))
    selected_area = None
    for i, area in enumerate(practice_areas):
        if cols[i].button(area.title(), key=f"pa_{area}", use_container_width=True):
            search_query = area if area != "All" else ""
            selected_area = area if area != "All" else None

    if search_query:
        try:
            from src.rag.retriever import Retriever
            retriever = Retriever()
            results = retriever.retrieve_for_topic(search_query, n_results=n_results)

            if not results:
                st.info("No results found. Try a different search term.")
            else:
                st.markdown(f"**{len(results)} results for '{search_query}':**")
                for r in results:
                    with st.expander(f"{r['title'][:80]} — {r['source_name']}", expanded=False):
                        cols = st.columns([2, 1, 1])
                        cols[0].markdown(f"**Court:** {r['court'][:60]}")
                        cols[1].markdown(f"**Date:** {r['date']}")
                        cols[2].markdown(f"**Relevance:** {r['score']:.2f}")
                        if r.get("practice_areas"):
                            st.markdown(f"**Practice areas:** {r['practice_areas']}")
                        if r.get("url"):
                            st.markdown(f"[View original case]({r['url']})")
        except Exception as e:
            st.error(f"Search failed: {e}")


# ════════════════════════════════════════════════════════
# PAGE: DATA PIPELINE
# ════════════════════════════════════════════════════════
def page_data_pipeline():
    st.header("Data Pipeline")
    st.markdown(
        "Collect cases from public sources, process them, and build the knowledge base. "
        "Each step can be run independently."
    )

    # ── Current status ─────────────────────────────────
    st.subheader("Current Status")
    col1, col2, col3 = st.columns(3)
    raw_count = _count_files(RAW_DATA_DIR)
    processed_count = _count_files(PROCESSED_DATA_DIR)
    kb_stats = _get_kb_stats()

    col1.metric("Raw Cases Scraped", raw_count, help=str(RAW_DATA_DIR))
    col2.metric("Cases Processed", processed_count, help=str(PROCESSED_DATA_DIR))
    col3.metric("KB Chunks", sum(kb_stats.values()), help=str(VECTOR_DB_DIR))

    st.markdown("---")

    # ── Step 1: Collect ────────────────────────────────
    with st.expander("Step 1: Collect Data from Public Sources", expanded=True):
        st.markdown("""
        Scrapes publicly available court cases from:
        - **CourtListener** (US court opinions — free REST API)
        - **Harvard Caselaw Access** (US historical cases — free API)
        - **Justia** (US case law — HTML scraping)
        - **BAILII** (British & Irish cases — HTML scraping)
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            max_cases = st.number_input("Max cases per source", min_value=5, max_value=1000, value=50)
        with col2:
            sources = st.multiselect(
                "Sources to scrape",
                ["CourtListener", "CaseLaw Access", "Justia/BAILII"],
                default=["CourtListener"],
            )
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_collect = st.button("Run Scraper", type="primary", use_container_width=True)

        if run_collect:
            with st.spinner("Scraping public sources... (this may take several minutes)"):
                try:
                    import subprocess
                    source_flags = []
                    if "CourtListener" in sources:
                        source_flags.append("--courtlistener")
                    if "CaseLaw Access" in sources:
                        source_flags.append("--caselaw")
                    if "Justia/BAILII" in sources:
                        source_flags.append("--web")

                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "collect_data.py"),
                        "--max", str(max_cases),
                    ] + source_flags

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        st.success("Scraping complete!")
                        st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                    else:
                        st.error("Scraping failed.")
                        st.code(result.stderr[-2000:])
                    # Clear cached stats
                    st.session_state.pop("kb_stats", None)
                except Exception as e:
                    st.error(f"Failed to run scraper: {e}")

    # ── Step 2: Process ────────────────────────────────
    with st.expander("Step 2: Process & Clean Data", expanded=False):
        st.markdown("""
        For each scraped case:
        1. **Clean text** — remove boilerplate, normalize whitespace
        2. **Remove PII** — anonymize personal names, phone numbers, emails, SSNs
        3. **Structure** — LLM extracts facts, ruling, reasoning, and learnings
        4. **Save** — write cleaned JSON to `data/processed/`

        *Note: Structuring with an LLM uses API credits (approx. $0.001-0.005 per case with Claude Haiku).*
        """)

        col1, col2 = st.columns(2)
        with col1:
            skip_structured = st.checkbox("Skip already-processed cases", value=True)
            limit = st.number_input("Process at most N cases (0 = all)", min_value=0, value=100)
        with col2:
            use_cheap_model = st.checkbox("Use fast/cheap model for structuring", value=True,
                                          help="Claude Haiku / GPT-4o-mini instead of full model")
            st.markdown("<br>", unsafe_allow_html=True)
            run_process = st.button("Process Data", type="primary", use_container_width=True)

        if run_process:
            with st.spinner("Processing cases... (this calls the LLM for each case)"):
                try:
                    import subprocess
                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "process_data.py"),
                    ]
                    if limit > 0:
                        cmd.extend(["--limit", str(limit)])
                    if skip_structured:
                        cmd.append("--skip-structured")
                    if use_cheap_model:
                        cmd.append("--fast-model")

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        st.success("Processing complete!")
                        st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                    else:
                        st.error("Processing failed.")
                        st.code(result.stderr[-2000:])
                    st.session_state.pop("kb_stats", None)
                except Exception as e:
                    st.error(f"Failed to run processor: {e}")

    # ── Step 3: Build RAG ──────────────────────────────
    with st.expander("Step 3: Build Knowledge Base (RAG Index)", expanded=False):
        st.markdown("""
        Indexes all processed cases into ChromaDB vector collections:
        - `opinions` — case facts, reasoning, rulings (chunked)
        - `learnings` — distilled legal principles
        - `summaries` — concise case overviews
        """)

        col1, col2 = st.columns(2)
        with col1:
            force_rebuild = st.checkbox("Force rebuild (delete existing index)", value=False)
        with col2:
            run_rag = st.button("Build Knowledge Base", type="primary", use_container_width=True)

        if run_rag:
            with st.spinner("Building knowledge base... (may take a few minutes for large datasets)"):
                try:
                    import subprocess
                    cmd = [sys.executable, str(ROOT / "scripts" / "build_rag.py")]
                    if force_rebuild:
                        cmd.append("--rebuild")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        st.success("Knowledge base built!")
                        st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                    else:
                        st.error("Build failed.")
                        st.code(result.stderr[-2000:])
                    st.session_state.pop("kb_stats", None)
                    st.session_state.pop("sme_agent", None)  # Force agent reload
                except Exception as e:
                    st.error(f"Failed to build RAG: {e}")


# ════════════════════════════════════════════════════════
# PAGE: DATASET VIEWER
# ════════════════════════════════════════════════════════
def page_dataset():
    st.header("Dataset Viewer")
    st.markdown("Browse and download the processed, PII-cleaned case dataset.")

    processed_count = _count_files(PROCESSED_DATA_DIR)
    if processed_count == 0:
        st.warning("No processed cases found. Run the Data Pipeline first.")
        return

    st.info(f"Found **{processed_count}** processed cases in `{PROCESSED_DATA_DIR}`")

    # Load cases
    if st.button("Load Dataset", type="primary"):
        import json
        cases = []
        for path in sorted(PROCESSED_DATA_DIR.rglob("*.json"))[:500]:  # Limit for display
            try:
                cases.append(json.loads(path.read_text()))
            except Exception:
                pass

        if cases:
            import pandas as pd
            rows = []
            for c in cases:
                rows.append({
                    "ID": c.get("source_id", ""),
                    "Title": c.get("title", "")[:60],
                    "Source": c.get("source_name", ""),
                    "Date": c.get("date", ""),
                    "Court": c.get("court", "")[:40],
                    "Practice Areas": ", ".join(c.get("practice_areas", [])),
                    "Structured": "Yes" if c.get("structured") else "No",
                    "Text Length": len(c.get("text", "")),
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Download
            import io
            jsonl = "\n".join(
                json.dumps(c, ensure_ascii=False) for c in cases
            )
            st.download_button(
                "Download Dataset (JSONL)",
                data=jsonl.encode("utf-8"),
                file_name=f"{DOMAIN}_cases_dataset.jsonl",
                mime="application/jsonlines",
            )

            # Sample case viewer
            st.markdown("---")
            st.subheader("View a Case")
            case_ids = [c.get("source_id", f"case_{i}") for i, c in enumerate(cases)]
            selected = st.selectbox("Select case", case_ids)
            case = next((c for c in cases if c.get("source_id") == selected), None)
            if case:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {case.get('title', '')}")
                    st.markdown(f"**Source:** {case.get('source_name', '')}")
                    st.markdown(f"**Date:** {case.get('date', '')}")
                    st.markdown(f"**Court:** {case.get('court', '')}")
                with col2:
                    st.markdown(f"**URL:** {case.get('url', '')}")
                    st.markdown(f"**Practice Areas:** {', '.join(case.get('practice_areas', []))}")
                    st.markdown(f"**Structured:** {case.get('structured', False)}")

                if case.get("summary"):
                    st.markdown("**Summary:**")
                    st.info(case["summary"])
                if case.get("facts"):
                    with st.expander("Facts"):
                        st.write(case["facts"])
                if case.get("ruling"):
                    with st.expander("Ruling"):
                        st.write(case["ruling"])
                if case.get("learnings"):
                    with st.expander("Learnings / Legal Principles"):
                        st.write(case["learnings"])


# ════════════════════════════════════════════════════════
# PAGE: HUGGINGFACE UPLOAD
# ════════════════════════════════════════════════════════
def page_huggingface():
    st.header("Upload Dataset to HuggingFace")
    st.markdown(
        "Upload the processed, PII-cleaned dataset to HuggingFace Hub "
        "so others can use it for research and fine-tuning."
    )

    # Status check
    hf_token = HF_TOKEN or os.getenv("HF_TOKEN", "")
    if not hf_token:
        st.warning(
            "No HuggingFace token found. Add `HF_TOKEN=hf_...` to your `.env` file. "
            "Get a token at https://huggingface.co/settings/tokens"
        )
    else:
        st.success(f"HF token configured (ends in ...{hf_token[-4:]})")

    processed_count = _count_files(PROCESSED_DATA_DIR)
    st.metric("Cases ready to upload", processed_count)

    if processed_count == 0:
        st.warning("No processed cases found. Complete the Data Pipeline first.")
        return

    st.markdown("---")
    st.subheader("Upload Configuration")

    repo_id = st.text_input(
        "HuggingFace Repository ID",
        value=HF_REPO_ID or "your-username/divorce-cases-en",
        help="Format: username/dataset-name"
    )

    col1, col2 = st.columns(2)
    with col1:
        make_private = st.checkbox("Make dataset private", value=False)
        include_full_text = st.checkbox("Include full text field", value=True)
    with col2:
        dataset_description = st.text_area(
            "Dataset description",
            value=f"Public {DOMAIN} law court cases collected from CourtListener, Harvard Caselaw Access, Justia, and BAILII. PII-anonymized. Includes structured fields: facts, ruling, reasoning, learnings.",
            height=100,
        )

    if st.button("Upload to HuggingFace", type="primary", disabled=not hf_token):
        if not repo_id or "/" not in repo_id:
            st.error("Please provide a valid repository ID in format 'username/dataset-name'")
            return

        with st.spinner("Uploading to HuggingFace... (this may take several minutes for large datasets)"):
            try:
                import subprocess
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "upload_to_hf.py"),
                    "--repo-id", repo_id,
                    "--description", dataset_description,
                ]
                if make_private:
                    cmd.append("--private")
                if not include_full_text:
                    cmd.append("--no-full-text")

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    st.success(f"Dataset uploaded!")
                    st.markdown(f"View at: https://huggingface.co/datasets/{repo_id}")
                    st.code(result.stdout)
                else:
                    st.error("Upload failed.")
                    st.code(result.stderr[-3000:])
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.subheader("Use the Uploaded Dataset")
    st.code(f"""from datasets import load_dataset

ds = load_dataset("{repo_id or 'your-username/divorce-cases-en'}", split="train")
print(ds)
# Dataset({{
#   features: ['source_id','title','source_name','url','date','court',
#              'text','facts','ruling','reasoning','learnings','summary','practice_areas'],
#   num_rows: {processed_count}
# }})
""")


# ════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════
def page_about():
    st.header("About This System")

    st.markdown(f"""
    ## Architecture

    This system follows a four-stage pipeline to build a Subject Matter Expert AI:

    ```
    Public Sources          Data Pipeline               AI Agent
    ─────────────          ─────────────               ────────
    CourtListener  ──►  Scrape & Clean  ──►  Structure  ──►  SME Agent
    Harvard CAP    ──►  Remove PII      ──►  Index      ──►  (RAG + LLM)
    Justia         ──►  Normalize       ──►  ChromaDB   ──►  Streamlit UI
    BAILII         ──►  Save JSON       ──►  HF Dataset ──►  CLI Chat
    ```

    ## Tech Stack

    | Component | Technology |
    |---|---|
    | LLM | Claude (Anthropic) / GPT-4 (OpenAI) / Gemini (Google) |
    | RAG | ChromaDB + sentence-transformers |
    | Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (free, no API needed) |
    | PII Removal | Microsoft Presidio + spaCy en_core_web_lg |
    | Web Scraping | requests + BeautifulSoup + Playwright |
    | Case Structuring | LLM (Claude Haiku / GPT-4o-mini) |
    | Dataset Hosting | HuggingFace Hub |
    | UI | Streamlit |

    ## Data Sources

    | Source | Coverage | Access |
    |---|---|---|
    | [CourtListener](https://www.courtlistener.com) | US federal & state courts | Free API |
    | [Harvard Caselaw Access](https://case.law) | US cases 1658-present | Free API (registration for full text) |
    | [Justia](https://law.justia.com) | US case law + guides | Free (HTML) |
    | [BAILII](https://www.bailii.org) | British & Irish cases | Free (HTML) |

    ## Privacy & Ethics

    - All personal names are replaced with `[PERSON_N]` placeholders using NER
    - Phone numbers, SSNs, emails replaced with typed placeholders
    - Court/party names in case titles are preserved (public record)
    - Only public domain documents are scraped
    - Respect for `robots.txt` and rate limits

    ## Legal Disclaimer

    This tool is for **educational and research purposes only**.
    It does not constitute legal advice.
    Always consult a qualified attorney for legal matters.
    The accuracy of AI-generated summaries and analysis is not guaranteed.

    ## Author

    **Albert García Díaz**
    - GitHub: [@albertgd](https://github.com/albertgd)
    - Related project: [legal-companion](https://github.com/albertgd/legal-companion)
    """)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown(f"### AI {DOMAIN.title()} SME")
        st.markdown("---")

        pages = {
            "Home": page_home,
            "Chat": page_chat,
            "Explore Cases": page_explore,
            "Data Pipeline": page_data_pipeline,
            "Dataset Viewer": page_dataset,
            "Upload to HuggingFace": page_huggingface,
            "About": page_about,
        }

        icons = {
            "Home": "🏠",
            "Chat": "💬",
            "Explore Cases": "🔍",
            "Data Pipeline": "⚙️",
            "Dataset Viewer": "📊",
            "Upload to HuggingFace": "🤗",
            "About": "ℹ️",
        }

        selection = st.radio(
            "Navigation",
            list(pages.keys()),
            format_func=lambda x: f"{icons[x]} {x}",
        )

        st.markdown("---")

        # API key status
        if _check_api_keys():
            st.success("LLM API key configured")
        else:
            st.error("No LLM API key. Edit .env")

        if HF_TOKEN:
            st.success("HuggingFace token configured")
        else:
            st.warning("No HF token (for upload only)")

        st.markdown("---")
        st.caption(f"Domain: {DOMAIN}")
        st.caption("AI Subject Matter Expert v1.0")

    pages[selection]()


if __name__ == "__main__":
    main()
