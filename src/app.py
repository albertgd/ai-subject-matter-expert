"""
AI Subject Matter Expert — Streamlit Web Interface

Auto-detects setup state:
  - No API key configured → Setup wizard
  - No data yet          → Pipeline page
  - Ready                → Home / Chat
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="AI Subject Matter Expert",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

  .page-title {
    font-size: 2rem; font-weight: 800; color: #1e293b;
    margin-bottom: 0.2rem;
  }
  .page-sub {
    color: #64748b; font-size: 1rem; margin-bottom: 1.5rem;
  }
  .card {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
  }
  .card-blue  { border-left: 4px solid #3b82f6; }
  .card-green { border-left: 4px solid #22c55e; }
  .card-amber { border-left: 4px solid #f59e0b; }
  .card-red   { border-left: 4px solid #ef4444; }

  .metric-box {
    background: linear-gradient(135deg,#f8fafc,#f1f5f9);
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1rem; text-align: center;
  }
  .metric-num { font-size: 2rem; font-weight: 800; color: #1e293b; }
  .metric-lbl { font-size: 0.8rem; color: #64748b; margin-top: 2px; }

  .step-badge {
    display: inline-block; background: #3b82f6; color: white;
    border-radius: 50%; width: 28px; height: 28px; text-align: center;
    line-height: 28px; font-weight: 700; font-size: 0.85rem; margin-right: 8px;
  }
  .step-done { background: #22c55e !important; }

  .source-chip {
    display: inline-block; background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 20px; padding: 2px 10px; font-size: 0.78rem;
    color: #475569; margin: 2px;
  }

  .preset-btn button {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important; font-size: 0.85rem !important;
    padding: 0.3rem 0.8rem !important;
  }
  .log-box {
    background: #0f172a; color: #86efac; font-family: monospace;
    font-size: 0.78rem; padding: 1rem; border-radius: 8px;
    max-height: 380px; overflow-y: auto; white-space: pre-wrap;
  }
  div[data-testid="stChatMessage"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Env helpers ───────────────────────────────────────────────────────────────
ENV_PATH = ROOT / ".env"

def read_env() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

def write_env(updates: dict):
    """Merge updates into .env (create if missing)."""
    env = read_env()
    env.update(updates)
    lines = []
    for k, v in env.items():
        lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    # Reload into process env
    for k, v in updates.items():
        os.environ[k] = v

def get_env(key: str, default: str = "") -> str:
    return read_env().get(key, os.getenv(key, default))

def is_configured() -> bool:
    return bool(
        get_env("ANTHROPIC_API_KEY") or
        get_env("OPENAI_API_KEY") or
        get_env("GOOGLE_API_KEY")
    )

def count_json(directory: Path) -> int:
    try:
        return len(list(directory.rglob("*.json")))
    except Exception:
        return 0

def kb_stats() -> dict:
    try:
        from src.config import VECTOR_DB_DIR, COLLECTION_NAME
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        return store.stats()
    except Exception:
        return {"documents": 0, "learnings": 0, "summaries": 0}

def stream_command(cmd: list, placeholder):
    """Run a command and stream its output into a Streamlit placeholder."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable] + cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=ROOT, env=env,
    )
    lines = []
    for line in process.stdout:
        lines.append(line.rstrip())
        placeholder.markdown(
            f'<div class="log-box">' + "\n".join(lines[-60:]) + "</div>",
            unsafe_allow_html=True,
        )
    process.wait()
    return process.returncode


# ── Navigation ────────────────────────────────────────────────────────────────
configured = is_configured()
raw_n   = count_json(ROOT / "data" / "raw")
proc_n  = count_json(ROOT / "data" / "processed")

# Auto-land on Setup if not configured
default_page = "Setup" if not configured else "Home"

with st.sidebar:
    st.markdown("## 🧠 AI SME")
    subject = get_env("SUBJECT", "—")
    st.caption(f"Subject: **{subject}**")
    st.markdown("---")

    pages = ["Home", "Chat", "Pipeline", "Knowledge Base", "Dataset", "Setup"]
    icons  = ["🏠", "💬", "⚙️", "🔍", "📦", "🛠️"]
    labels = [f"{i}  {p}" for i, p in zip(icons, pages)]

    default_idx = pages.index(default_page)
    choice = st.radio("", labels, index=default_idx, label_visibility="collapsed")
    page = pages[labels.index(choice)]

    st.markdown("---")
    st.caption(f"Raw docs: **{raw_n:,}**")
    st.caption(f"Processed: **{proc_n:,}**")
    if configured:
        provider = (
            "Anthropic" if get_env("ANTHROPIC_API_KEY") else
            "Google"    if get_env("GOOGLE_API_KEY")    else "OpenAI"
        )
        st.caption(f"LLM: **{provider}**")
    else:
        st.caption("⚠️ Not configured")


# ══════════════════════════════════════════════════════════════════════════════
# SETUP PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Setup":
    st.markdown('<div class="page-title">🛠️ Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Configure once — everything else happens inside the app.</div>', unsafe_allow_html=True)

    # ── Step 1: LLM ──────────────────────────────────────────────────────────
    st.markdown("### Step 1 — LLM API Key")
    st.markdown("The AI needs an LLM to generate search queries, extract knowledge, and answer questions.")

    provider = st.radio(
        "Provider",
        ["Anthropic (Claude)", "OpenAI (GPT-4)", "Google (Gemini)"],
        horizontal=True,
        index=0,
    )

    key_map = {
        "Anthropic (Claude)": ("ANTHROPIC_API_KEY", "sk-ant-..."),
        "OpenAI (GPT-4)":     ("OPENAI_API_KEY",    "sk-..."),
        "Google (Gemini)":    ("GOOGLE_API_KEY",     "AIza..."),
    }
    env_key, placeholder_text = key_map[provider]
    current_key = get_env(env_key, "")
    masked = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else ""

    api_key = st.text_input(
        f"{provider} API Key",
        value="" if not masked else "",
        placeholder=masked or placeholder_text,
        type="password",
    )
    if not api_key and masked:
        st.caption(f"Currently set: `{masked}`")

    st.markdown("---")

    # ── Step 2: Subject ───────────────────────────────────────────────────────
    st.markdown("### Step 2 — Choose a Subject")
    st.markdown("What topic should your AI expert specialize in?")

    PRESETS = {
        "🤖 Artificial Intelligence": {
            "subject": "artificial intelligence",
            "desc":    "artificial intelligence, machine learning, and deep learning",
            "kws":     "artificial intelligence,machine learning,deep learning,neural networks,LLM,transformers,GPT",
        },
        "🌍 Climate Change": {
            "subject": "climate change",
            "desc":    "climate science, global warming, and environmental policy",
            "kws":     "climate change,global warming,CO2,greenhouse gas,IPCC,carbon,renewable energy",
        },
        "💰 Personal Finance": {
            "subject": "personal finance",
            "desc":    "personal finance, investing, budgeting, and wealth management",
            "kws":     "investing,stocks,bonds,ETF,budgeting,savings,retirement,401k,compound interest",
        },
        "⚕️ Nutrition & Health": {
            "subject": "nutrition and health",
            "desc":    "nutrition science, healthy eating, and preventive health",
            "kws":     "nutrition,diet,vitamins,protein,carbohydrates,metabolism,gut health,supplements",
        },
        "🚀 Space Exploration": {
            "subject": "space exploration",
            "desc":    "space science, astronomy, and space missions",
            "kws":     "NASA,SpaceX,Mars,moon,rocket,satellite,asteroid,telescope,cosmology",
        },
        "⚖️ Divorce Law": {
            "subject": "divorce law",
            "desc":    "divorce and family law proceedings",
            "kws":     "divorce,custody,alimony,child support,marital property,separation",
        },
        "🧬 Genetics": {
            "subject": "genetics",
            "desc":    "genetics, genomics, and molecular biology",
            "kws":     "DNA,gene,genome,mutation,CRISPR,heredity,chromosome,protein,RNA",
        },
        "🎨 Custom": {
            "subject": "", "desc": "", "kws": "",
        },
    }

    cols = st.columns(4)
    selected_preset = st.session_state.get("preset", "🤖 Artificial Intelligence")
    for i, (label, _) in enumerate(PRESETS.items()):
        with cols[i % 4]:
            if st.button(label, use_container_width=True, key=f"preset_{label}"):
                st.session_state["preset"] = label
                st.rerun()

    preset_data = PRESETS.get(selected_preset, PRESETS["🤖 Artificial Intelligence"])
    st.markdown(f"**Selected:** {selected_preset}")
    st.markdown("---")

    cur_subject = get_env("SUBJECT", preset_data["subject"])
    cur_desc    = get_env("SUBJECT_DESCRIPTION", preset_data["desc"])
    cur_kws     = get_env("SUBJECT_KEYWORDS", preset_data["kws"])

    subject_val = st.text_input("Subject name", value=preset_data["subject"] or cur_subject)
    desc_val    = st.text_input("Description", value=preset_data["desc"] or cur_desc)
    kws_val     = st.text_input("Keywords (comma-separated)", value=preset_data["kws"] or cur_kws)

    st.markdown("---")

    # ── Save ──────────────────────────────────────────────────────────────────
    if st.button("💾  Save Configuration", type="primary", use_container_width=True):
        updates = {}
        if api_key:
            updates[env_key] = api_key
        if subject_val:
            updates["SUBJECT"] = subject_val
        if desc_val:
            updates["SUBJECT_DESCRIPTION"] = desc_val
        if kws_val:
            updates["SUBJECT_KEYWORDS"] = kws_val

        if not updates.get(env_key) and not get_env(env_key):
            st.error("Please enter your API key.")
        else:
            write_env(updates)
            st.success("Configuration saved!")
            time.sleep(0.8)
            st.rerun()

    if is_configured():
        st.info("Configuration is complete. Go to **Pipeline** to start researching, or **Chat** to talk to your expert.")


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Home":
    subject = get_env("SUBJECT", "your subject")
    st.markdown(f'<div class="page-title">🧠 AI Subject Matter Expert</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{get_env("SUBJECT_DESCRIPTION", subject)}</div>', unsafe_allow_html=True)

    if not configured:
        st.warning("Not configured yet. Go to **Setup** to add your API key and choose a subject.")

    # Stats
    stats = kb_stats()
    kb_total = sum(stats.values())
    c1, c2, c3, c4 = st.columns(4)
    def metric(col, num, label, color="#1e293b"):
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-num" style="color:{color}">{num}</div>'
                f'<div class="metric-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )
    metric(c1, f"{raw_n:,}",     "Raw Documents")
    metric(c2, f"{proc_n:,}",    "Processed")
    metric(c3, f"{kb_total:,}",  "KB Chunks")
    metric(c4, "Ready" if kb_total > 0 else "Empty", "Status",
           "#16a34a" if kb_total > 0 else "#dc2626")

    st.markdown("---")

    # ── Quick start ───────────────────────────────────────────────────────────
    if kb_total == 0 and configured:
        st.markdown("### Quick Start")
        st.markdown("Build your knowledge base in one click:")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            qs_max   = st.number_input("Max documents to fetch", 10, 500, 50, step=10)
            qs_fast  = st.checkbox("Fast mode (skip LLM structuring — much quicker)", value=False)
        with col_right:
            st.markdown("<br/>", unsafe_allow_html=True)
            run_all = st.button("🚀  Build Everything", type="primary", use_container_width=True)

        if run_all:
            log_ph = st.empty()
            with st.spinner(""):
                st.markdown("**Step 1/3 — Researching the web...**")
                rc = stream_command(
                    ["scripts/research.py", "--max", str(qs_max)], log_ph
                )
                if rc != 0:
                    st.error("Research step failed.")
                    st.stop()

                st.markdown("**Step 2/3 — Processing documents...**")
                cmd = ["scripts/process_data.py"]
                if qs_fast:
                    cmd.append("--no-structure")
                rc = stream_command(cmd, log_ph)
                if rc != 0:
                    st.error("Processing step failed.")
                    st.stop()

                st.markdown("**Step 3/3 — Building knowledge base...**")
                rc = stream_command(["scripts/build_rag.py"], log_ph)

            if rc == 0:
                st.success("Knowledge base is ready! Go to Chat to start asking questions.")
                st.balloons()
                st.cache_resource.clear()

    elif kb_total > 0:
        st.markdown("### Knowledge Base")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Documents", f"{stats.get('documents',0):,} chunks")
        with c2: st.metric("Learnings",  f"{stats.get('learnings',0):,} chunks")
        with c3: st.metric("Summaries",  f"{stats.get('summaries',0):,} chunks")
        st.markdown("Your expert is ready. Go to **Chat** to ask questions.")

    st.markdown("---")
    st.markdown("### How It Works")
    steps = [
        ("1", "Research", "AI generates smart queries → searches DuckDuckGo/Tavily → fetches pages + Wikipedia"),
        ("2", "Process",  "Clean text, remove PII (names/emails/phones), extract structured knowledge with LLM"),
        ("3", "Index",    "Embed documents into ChromaDB with multilingual sentence transformers"),
        ("4", "Chat",     "Ask anything — grounded answers with cited sources"),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            done = (
                (num == "1" and raw_n > 0) or
                (num == "2" and proc_n > 0) or
                (num in ("3", "4") and kb_total > 0)
            )
            badge_cls = "step-badge step-done" if done else "step-badge"
            st.markdown(
                f'<div class="card card-blue">'
                f'<span class="{badge_cls}">{num}</span><strong>{title}</strong>'
                f'<br/><small style="color:#64748b">{desc}</small></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Chat":
    subject = get_env("SUBJECT", "your subject")
    st.markdown(f'<div class="page-title">💬 Chat</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Ask anything about {subject}.</div>', unsafe_allow_html=True)

    if not configured:
        st.error("Go to **Setup** to configure your API key first.")
        st.stop()

    # Load agent (cached)
    @st.cache_resource(show_spinner="Loading expert...")
    def load_agent():
        # Reload env into process so the agent picks up the keys
        env = read_env()
        for k, v in env.items():
            os.environ[k] = v
        from src.agents.sme_agent import SMEAgent
        return SMEAgent()

    try:
        agent = load_agent()
    except Exception as e:
        st.error(f"Could not load agent: {e}")
        st.stop()

    kb_ready = agent._kb_ready
    if not kb_ready:
        st.info("Knowledge base is empty — the agent will use general knowledge only. Go to **Pipeline** to build it.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} sources"):
                    for src in msg["sources"]:
                        url   = src.get("url", "")
                        title = src.get("title", "Source")
                        link  = f"[{title}]({url})" if url else title
                        chips = ""
                        for tag in src.get("topics", "").split(","):
                            if tag.strip():
                                chips += f'<span class="source-chip">{tag.strip()}</span>'
                        st.markdown(
                            f'<div class="card card-blue">'
                            f'<strong>{link}</strong><br/>'
                            f'<span style="color:#64748b;font-size:0.8rem">'
                            f'{src.get("source_name","")} · score {src.get("score",0):.2f}'
                            f'</span><br/>{chips}</div>',
                            unsafe_allow_html=True,
                        )

    if prompt := st.chat_input(f"Ask about {subject}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = agent.chat_with_sources(prompt)
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"📚 {len(sources)} sources"):
                            for src in sources:
                                url   = src.get("url", "")
                                title = src.get("title", "Source")
                                link  = f"[{title}]({url})" if url else title
                                chips = ""
                                for tag in src.get("topics", "").split(","):
                                    if tag.strip():
                                        chips += f'<span class="source-chip">{tag.strip()}</span>'
                                st.markdown(
                                    f'<div class="card card-blue">'
                                    f'<strong>{link}</strong><br/>'
                                    f'<span style="color:#64748b;font-size:0.8rem">'
                                    f'{src.get("source_name","")} · score {src.get("score",0):.2f}'
                                    f'</span><br/>{chips}</div>',
                                    unsafe_allow_html=True,
                                )
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.messages:
        if st.button("🗑️  Clear conversation"):
            st.session_state.messages = []
            agent.reset()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Pipeline":
    subject = get_env("SUBJECT", "your subject")
    st.markdown(f'<div class="page-title">⚙️ Pipeline</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Build the knowledge base for <strong>{subject}</strong>.</div>', unsafe_allow_html=True)

    if not configured:
        st.error("Go to **Setup** first.")
        st.stop()

    # Status bar
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Raw Documents", f"{raw_n:,}", delta="fetched")
    with c2: st.metric("Processed", f"{proc_n:,}", delta="structured")
    stats = kb_stats()
    with c3: st.metric("KB Chunks", f"{sum(stats.values()):,}", delta="indexed")

    st.markdown("---")

    # ── One-click ────────────────────────────────────────────────────────────
    with st.expander("🚀  **Build Everything** (recommended for first run)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            max_docs  = st.number_input("Max documents to fetch", 10, 1000, 100, step=10, key="be_max")
            n_queries = st.number_input("AI search queries", 5, 30, 10, key="be_q")
        with col2:
            fast_mode = st.checkbox("Fast mode (skip LLM structuring)", key="be_fast")
            rebuild   = st.checkbox("Force full rebuild of KB", key="be_rebuild")

        if st.button("🚀  Build Everything", type="primary", use_container_width=True, key="be_btn"):
            log_ph = st.empty()
            success = True

            st.markdown("**① Researching the web and Wikipedia...**")
            rc = stream_command(
                ["scripts/research.py", "--max", str(max_docs), "--queries", str(n_queries)],
                log_ph,
            )
            if rc != 0: success = False

            if success:
                st.markdown("**② Processing documents (clean → PII → structure)...**")
                cmd = ["scripts/process_data.py"]
                if fast_mode: cmd.append("--no-structure")
                rc = stream_command(cmd, log_ph)
                if rc != 0: success = False

            if success:
                st.markdown("**③ Building knowledge base...**")
                cmd = ["scripts/build_rag.py"]
                if rebuild: cmd.append("--rebuild")
                rc = stream_command(cmd, log_ph)
                if rc != 0: success = False

            if success:
                st.success("Done! Your expert is ready. Go to **Chat**.")
                st.balloons()
                st.cache_resource.clear()
            else:
                st.error("A step failed. Check the log above.")

    st.markdown("---")

    # ── Individual steps ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["① Research", "② Process", "③ Build KB"])

    with tab1:
        st.markdown("**AI-powered web research** — generates queries, searches the web, fetches Wikipedia.")
        c1, c2, c3 = st.columns(3)
        with c1: t1_max = st.number_input("Max docs", 10, 1000, 100, step=10)
        with c2: t1_q   = st.number_input("Queries", 5, 30, 10)
        with c3:
            t1_src = st.selectbox("Source", ["All", "Web (AI)", "Wikipedia"])

        if st.button("Run Research", type="primary"):
            cmd = ["scripts/research.py", "--max", str(t1_max), "--queries", str(t1_q)]
            if t1_src == "Web (AI)": cmd.append("--web")
            elif t1_src == "Wikipedia": cmd.append("--wikipedia")
            ph = st.empty()
            rc = stream_command(cmd, ph)
            (st.success if rc == 0 else st.error)(
                "Research complete!" if rc == 0 else "Research failed — see log."
            )

    with tab2:
        st.markdown("**Clean + anonymize PII + extract structured knowledge with LLM.**")
        c1, c2 = st.columns(2)
        with c1:
            t2_limit = st.number_input("Limit (0 = all)", 0, 10000, 0, step=50)
            t2_fast  = st.checkbox("Fast model (cheaper)")
        with c2:
            t2_skip  = st.checkbox("Skip already-structured")
            t2_nostr = st.checkbox("No LLM structuring (fastest)")

        if st.button("Run Processing", type="primary"):
            cmd = ["scripts/process_data.py"]
            if t2_limit: cmd += ["--limit", str(t2_limit)]
            if t2_fast:  cmd.append("--fast-model")
            if t2_skip:  cmd.append("--skip-structured")
            if t2_nostr: cmd.append("--no-structure")
            ph = st.empty()
            rc = stream_command(cmd, ph)
            (st.success if rc == 0 else st.error)(
                "Processing complete!" if rc == 0 else "Processing failed — see log."
            )

    with tab3:
        st.markdown("**Index processed documents into ChromaDB for semantic search.**")
        t3_rebuild = st.checkbox("Force full rebuild (delete existing index)")

        if st.button("Build Knowledge Base", type="primary"):
            cmd = ["scripts/build_rag.py"]
            if t3_rebuild: cmd.append("--rebuild")
            ph = st.empty()
            rc = stream_command(cmd, ph)
            if rc == 0:
                st.success("Knowledge base built!")
                st.cache_resource.clear()
            else:
                st.error("Build failed — see log.")


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Knowledge Base":
    st.markdown('<div class="page-title">🔍 Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Search and browse the indexed knowledge.</div>', unsafe_allow_html=True)

    stats = kb_stats()
    total = sum(stats.values())
    if total == 0:
        st.warning("Knowledge base is empty. Go to **Pipeline** to build it.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Documents",  f"{stats.get('documents',0):,} chunks")
    with c2: st.metric("Learnings",  f"{stats.get('learnings',0):,} chunks")
    with c3: st.metric("Summaries",  f"{stats.get('summaries',0):,} chunks")

    st.markdown("---")
    st.subheader("Search")

    @st.cache_resource(show_spinner=False)
    def get_store():
        from src.rag.vector_store import VectorStore
        return VectorStore()

    query = st.text_input("Search query", placeholder="Enter any question or topic...")
    col1, col2 = st.columns(2)
    with col1: k = st.slider("Results", 1, 20, 5)
    with col2: collection = st.selectbox("Collection", ["documents", "learnings", "summaries"])

    if query:
        store = get_store()
        with st.spinner("Searching..."):
            results = store.search_with_score(query, k=k, collection=collection)
        st.markdown(f"**{len(results)} results** in `{collection}`")
        for doc, score in results:
            meta = doc.metadata
            url   = meta.get("url", "")
            title = meta.get("title", "Unknown")
            with st.expander(f"{title}  ·  score {score:.3f}"):
                cols = st.columns(2)
                with cols[0]:
                    if url: st.markdown(f"[{meta.get('source_name', url)}]({url})")
                    st.caption(f"Date: {meta.get('date','—')}")
                with cols[1]:
                    topics = meta.get("topics", "")
                    if topics:
                        chips = "".join(f'<span class="source-chip">{t.strip()}</span>' for t in topics.split(",") if t.strip())
                        st.markdown(chips, unsafe_allow_html=True)
                st.text(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.markdown('<div class="page-title">📦 Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">View, download, and publish your cleaned dataset.</div>', unsafe_allow_html=True)

    PROC_DIR = ROOT / "data" / "processed"
    docs = []
    for path in sorted(PROC_DIR.rglob("*.json")):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

    if not docs:
        st.warning("No processed documents yet. Run the Pipeline first.")
        st.stop()

    st.metric("Total Documents", len(docs))

    # Filter
    search = st.text_input("Filter by title or topic", "")
    if search:
        docs = [d for d in docs if
                search.lower() in d.get("title", "").lower() or
                search.lower() in ",".join(d.get("topics", [])).lower()]
        st.caption(f"{len(docs)} matching")

    # Table
    import pandas as pd
    rows = [{"Title": d.get("title","")[:70], "Source": d.get("source_name",""),
             "Date": d.get("date",""), "Topics": ", ".join(d.get("topics",[]))[:50],
             "Structured": "✓" if d.get("structured") else "—"}
            for d in docs[:300]]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

    # Detail
    st.markdown("---")
    titles = [d.get("title", d.get("source_id","?"))[:80] for d in docs[:100]]
    sel = st.selectbox("View document", titles)
    if sel:
        doc = next((d for d in docs if d.get("title","").startswith(sel[:40])), None)
        if doc:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Source:** {doc.get('source_name','')}")
                st.write(f"**Date:** {doc.get('date','—')}")
                st.write(f"**Author:** {doc.get('author','—')}")
                if doc.get("url"):
                    st.write(f"**URL:** [{doc['url']}]({doc['url']})")
            with col2:
                chips = "".join(f'<span class="source-chip">{t}</span>' for t in doc.get("topics",[]))
                if chips: st.markdown(chips, unsafe_allow_html=True)
                st.write(f"**Structured:** {'Yes' if doc.get('structured') else 'No'}")
            for field, label in [("summary","Summary"), ("key_points","Key Points"), ("learnings","Learnings")]:
                if doc.get(field):
                    with st.expander(label):
                        st.write(doc[field])

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        jsonl = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs)
        subject_slug = get_env("SUBJECT", "dataset").replace(" ", "_")
        st.download_button("⬇️  Download JSONL", data=jsonl,
                           file_name=f"{subject_slug}_dataset.jsonl",
                           mime="application/json", use_container_width=True)
    with col2:
        hf_token = get_env("HF_TOKEN")
        if not hf_token:
            st.warning("Set HF_TOKEN in Setup to enable upload.")
        else:
            repo_id  = st.text_input("HuggingFace Repo ID",
                                     value=get_env("HF_REPO_ID", f"username/{subject_slug}-sme"))
            private  = st.checkbox("Private")
            if st.button("⬆️  Upload to HuggingFace", type="primary", use_container_width=True):
                ph = st.empty()
                cmd = ["scripts/upload_to_hf.py", "--repo-id", repo_id]
                if private: cmd.append("--private")
                rc = stream_command(cmd, ph)
                if rc == 0:
                    st.success(f"Uploaded! View at https://huggingface.co/datasets/{repo_id}")
