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
        get_env("GOOGLE_API_KEY") or
        get_env("GROQ_API_KEY")
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


# ── Language / UI translations ────────────────────────────────────────────────
ui_lang = get_env("LANGUAGE", "en")

_TEXTS = {
    "en": {
        "app_title": "AI SME",
        "subject_label": "Subject:",
        "raw_docs": "Raw docs:",
        "processed": "Processed:",
        "not_configured": "⚠️ Not configured",
        "nav_home": "Home", "nav_chat": "Chat", "nav_pipeline": "Pipeline",
        "nav_kb": "Knowledge Base", "nav_dataset": "Dataset",
        "nav_setup": "Setup", "nav_about": "About",
        "setup_title": "🛠️ Setup",
        "setup_sub": "Configure once — everything else happens inside the app.",
        "step1_title": "Step 1 — LLM API Key",
        "step1_desc": "The AI needs an LLM to generate search queries, extract knowledge, and answer questions.",
        "step2_title": "Step 2 — Choose a Subject",
        "step2_desc": "Click a category to select it, or drill down up to 3 levels for a specialty. You can save at any level.",
        "step3_title": "Step 3 — Research Region",
        "step3_desc": "Choose the geographic focus for research. Use the language buttons at the top of the sidebar to switch language instantly.",
        "lang_label": "Interface language",
        "lang_en": "English", "lang_es": "Español",
        "region_label": "Research region",
        "region_anywhere": "🌍 Anywhere (global)",
        "region_spain": "🇪🇸 Spain",
        "region_catalunya": "🏴 Catalunya",
        "region_europe": "🇪🇺 Europe",
        "save_btn": "💾  Save Configuration",
        "save_ok": "Configuration saved!",
        "save_no_key": "Please enter your API key.",
        "setup_done": "Configuration is complete. Go to **Pipeline** to start researching, or **Chat** to talk to your expert.",
        "subject_name": "Subject name", "description": "Description",
        "keywords": "Keywords (comma-separated)",
        "selected": "Selected:",
        "home_sub": "Build an AI expert on any topic.",
        "not_configured_warn": "Not configured yet. Go to **Setup** to add your API key and choose a subject.",
        "kb_ready": "Ready", "kb_empty": "Empty",
        "raw_documents": "Raw Documents", "processed_label": "Processed",
        "kb_chunks": "KB Chunks", "status": "Status",
        "quick_start": "Quick Start",
        "quick_start_desc": "Build your knowledge base in one click:",
        "max_docs": "Max documents to fetch",
        "fast_mode": "Fast mode (skip LLM structuring — much quicker)",
        "build_everything": "🚀  Build Everything",
        "step1_research": "**Step 1/3 — Researching the web...**",
        "step2_process": "**Step 2/3 — Processing documents...**",
        "step3_build": "**Step 3/3 — Building knowledge base...**",
        "research_failed": "Research step failed.",
        "process_failed": "Processing step failed.",
        "kb_ready_msg": "Knowledge base is ready! Go to Chat to start asking questions.",
        "how_it_works": "### How It Works",
        "step_research": "Research", "step_research_desc": "AI generates smart queries → searches DuckDuckGo/Tavily → fetches pages + Wikipedia",
        "step_process": "Process", "step_process_desc": "Clean text, remove PII (names/emails/phones), extract structured knowledge with LLM",
        "step_index": "Index", "step_index_desc": "Embed documents into ChromaDB with multilingual sentence transformers",
        "step_chat": "Chat", "step_chat_desc": "Ask anything — grounded answers with cited sources",
        "chat_title": "💬 Chat",
        "chat_sub": "Ask anything about {subject}.",
        "chat_no_config": "Go to **Setup** to configure your API key first.",
        "chat_no_kb": "Knowledge base is empty — the agent will use general knowledge only. Go to **Pipeline** to build it.",
        "thinking": "Thinking...",
        "sources": "{n} sources",
        "clear_conv": "🗑️  Clear conversation",
        "chat_input": "Ask about {subject}...",
        "pipeline_title": "⚙️ Pipeline",
        "pipeline_sub": "Build the knowledge base for <strong>{subject}</strong>.",
        "pipeline_no_config": "Go to **Setup** first.",
        "build_everything_exp": "🚀  **Build Everything** (recommended for first run)",
        "ai_queries": "AI search queries",
        "fast_mode2": "Fast mode (skip LLM structuring)",
        "reprocess": "Reprocess already-processed docs",
        "rebuild": "Force full rebuild of KB",
        "step_research_title": "**① Researching the web and Wikipedia...**",
        "step_process_title": "**② Processing documents (clean → PII → structure)...**",
        "step_build_title": "**③ Building knowledge base...**",
        "done_msg": "Done! Your expert is ready. Go to **Chat**.",
        "step_failed": "A step failed. Check the log above.",
        "tab_research": "① Research", "tab_process": "② Process", "tab_build": "③ Build KB",
        "research_desc": "**AI-powered web research** — generates queries, searches the web, fetches Wikipedia.",
        "run_research": "Run Research",
        "research_complete": "Research complete!",
        "research_fail": "Research failed — see log.",
        "process_desc": "**Clean + anonymize PII + extract structured knowledge with LLM.**",
        "limit": "Limit (0 = all)", "fast_model": "Fast model (cheaper)",
        "reprocess2": "Reprocess already-processed docs",
        "no_structure": "No LLM structuring (fastest)",
        "run_process": "Run Processing",
        "process_complete": "Processing complete!",
        "process_fail": "Processing failed — see log.",
        "build_desc": "**Index processed documents into ChromaDB for semantic search.**",
        "force_rebuild": "Force full rebuild (delete existing index)",
        "build_kb": "Build Knowledge Base",
        "build_complete": "Knowledge base built!",
        "build_fail": "Build failed — see log.",
        "source_select": "Source",
        "max_docs_short": "Max docs",
        "queries_label": "Queries",
        "kb_page_title": "🔍 Knowledge Base",
        "kb_page_sub": "Search and browse the indexed knowledge.",
        "kb_empty_warn": "Knowledge base is empty. Go to **Pipeline** to build it.",
        # KB search
        "kb_documents": "Documents", "kb_learnings": "Learnings", "kb_summaries": "Summaries",
        "kb_chunks": "chunks",
        "kb_search_header": "Search",
        "kb_search_input": "Search query",
        "kb_search_placeholder": "Enter any question or topic...",
        "kb_results_slider": "Results",
        "kb_collection": "Collection",
        "kb_searching": "Searching...",
        "kb_no_results": "No results in `{col}`. This collection may be empty or its index is missing — go to **Pipeline → Build KB** and enable **Force full rebuild**.",
        "kb_n_results": "**{n} results** in `{col}`",
        "kb_date": "Date:",
        "kb_search_error": "Could not search `{col}`: {err}\n\nGo to **Pipeline → Build KB** and enable **Force full rebuild** to fix the index.",
        # Home KB-ready section
        "kb_ready_goto": "Your expert is ready. Go to **Chat** to ask questions.",
        "kb_section_header": "### Knowledge Base",
        # Dataset
        "dataset_title": "📦 Dataset",
        "dataset_sub": "View, download, and publish your cleaned dataset.",
        "dataset_no_docs": "No processed documents yet. Run the Pipeline first.",
        "dataset_total": "Total Documents",
        "dataset_filter": "Filter by title or topic",
        "dataset_matching": "{n} matching",
        "dataset_col_title": "Title", "dataset_col_source": "Source",
        "dataset_col_date": "Date", "dataset_col_topics": "Topics",
        "dataset_col_structured": "Structured",
        "dataset_view_doc": "View document",
        "dataset_source": "**Source:**", "dataset_date": "**Date:**",
        "dataset_author": "**Author:**", "dataset_url": "**URL:**",
        "dataset_structured_lbl": "**Structured:**",
        "dataset_yes": "Yes", "dataset_no": "No",
        "dataset_summary": "Summary", "dataset_key_points": "Key Points",
        "dataset_learnings": "Learnings",
        "dataset_download": "⬇️  Download JSONL",
        "dataset_hf_warn": "Set HF_TOKEN in Setup to enable upload.",
        "dataset_hf_repo": "HuggingFace Repo ID",
        "dataset_private": "Private",
        "dataset_upload": "⬆️  Upload to HuggingFace",
        "dataset_upload_ok": "Uploaded! View at {url}",
        # About
        "about_title": "👤 About",
        "about_sub": "The person behind AI Subject Matter Expert.",
        "about_bio": "Builder of AI-powered tools that make expert knowledge accessible to everyone.",
        "about_project_desc": "AI Subject Matter Expert is an open-source project that lets you spin up a fully grounded, RAG-powered AI expert on <em>any</em> topic — from space exploration to family law — in minutes, using free or low-cost LLM providers.",
        "about_goal": "The goal is simple: democratise access to deep, reliable knowledge without needing a background in machine learning or paying enterprise prices.",
        "about_project_title": "### About This Project",
        "about_features": "- Researches any topic autonomously using web search + Wikipedia\n- Cleans, anonymises, and structures knowledge with LLMs\n- Stores everything in a local vector database (ChromaDB)\n- Answers questions with cited, grounded responses — no hallucination\n- Works with free providers (Groq, Google Gemini) or paid ones (Anthropic, OpenAI)\n- Exports your dataset to JSONL or HuggingFace",
        "about_contact_title": "### Get in Touch",
        "about_tagline": "Open-source · Self-hosted<br/>Any topic · Any LLM",
        "about_stack_title": "Stack",
        "about_belief": "Built with the belief that everyone deserves access to expert-level knowledge.",
        # Pipeline metrics & danger zone
        "pipeline_raw": "Raw Documents", "pipeline_processed": "Processed",
        "pipeline_fetched": "fetched", "pipeline_structured": "structured",
        "kb_chunks_label": "KB Chunks", "kb_indexed": "indexed",
        "danger_title": "🗑️  **Danger Zone — Reset All Data**",
        "danger_warn": "**This will permanently delete all collected, processed, and indexed data.**",
        "danger_sub": "Raw documents, processed JSONs, and the entire vector database will be wiped. You will need to run the full pipeline again from scratch.",
        "danger_confirm": "I understand this cannot be undone — delete everything",
        "danger_btn": "🗑️  Reset All Data",
        "danger_ok": "All data has been reset. Run the pipeline to start fresh.",
    },
    "es": {
        "app_title": "AI Experto",
        "subject_label": "Tema:",
        "raw_docs": "Docs brutos:",
        "processed": "Procesados:",
        "not_configured": "⚠️ Sin configurar",
        "nav_home": "Inicio", "nav_chat": "Chat", "nav_pipeline": "Pipeline",
        "nav_kb": "Base de Conocimiento", "nav_dataset": "Dataset",
        "nav_setup": "Configuración", "nav_about": "Acerca de",
        "setup_title": "🛠️ Configuración",
        "setup_sub": "Configura una vez — todo lo demás ocurre dentro de la app.",
        "step1_title": "Paso 1 — Clave API del LLM",
        "step1_desc": "La IA necesita un LLM para generar consultas, extraer conocimiento y responder preguntas.",
        "step2_title": "Paso 2 — Elige un Tema",
        "step2_desc": "Haz clic en una categoría o navega hasta 3 niveles para una especialidad. Puedes guardar en cualquier nivel.",
        "step3_title": "Paso 3 — Región de Investigación",
        "step3_desc": "Elige el foco geográfico para la investigación. Usa los botones de idioma en la parte superior de la barra lateral para cambiar el idioma al instante.",
        "lang_label": "Idioma de la interfaz",
        "lang_en": "English", "lang_es": "Español",
        "region_label": "Región de investigación",
        "region_anywhere": "🌍 En cualquier lugar (global)",
        "region_spain": "🇪🇸 España",
        "region_catalunya": "🏴 Catalunya",
        "region_europe": "🇪🇺 Europa",
        "save_btn": "💾  Guardar Configuración",
        "save_ok": "¡Configuración guardada!",
        "save_no_key": "Por favor introduce tu clave API.",
        "setup_done": "Configuración completa. Ve a **Pipeline** para investigar, o a **Chat** para hablar con tu experto.",
        "subject_name": "Nombre del tema", "description": "Descripción",
        "keywords": "Palabras clave (separadas por comas)",
        "selected": "Seleccionado:",
        "home_sub": "Crea un experto en IA sobre cualquier tema.",
        "not_configured_warn": "Aún no configurado. Ve a **Configuración** para añadir tu clave API y elegir un tema.",
        "kb_ready": "Listo", "kb_empty": "Vacío",
        "raw_documents": "Documentos Brutos", "processed_label": "Procesados",
        "kb_chunks": "Fragmentos KB", "status": "Estado",
        "quick_start": "Inicio Rápido",
        "quick_start_desc": "Construye tu base de conocimiento en un clic:",
        "max_docs": "Máximo de documentos a obtener",
        "fast_mode": "Modo rápido (omitir estructuración LLM — mucho más rápido)",
        "build_everything": "🚀  Construir Todo",
        "step1_research": "**Paso 1/3 — Investigando la web...**",
        "step2_process": "**Paso 2/3 — Procesando documentos...**",
        "step3_build": "**Paso 3/3 — Construyendo base de conocimiento...**",
        "research_failed": "El paso de investigación falló.",
        "process_failed": "El paso de procesamiento falló.",
        "kb_ready_msg": "¡La base de conocimiento está lista! Ve a Chat para hacer preguntas.",
        "how_it_works": "### Cómo Funciona",
        "step_research": "Investigar", "step_research_desc": "La IA genera consultas → busca en DuckDuckGo/Tavily → obtiene páginas + Wikipedia",
        "step_process": "Procesar", "step_process_desc": "Limpiar texto, eliminar PII, extraer conocimiento estructurado con LLM",
        "step_index": "Indexar", "step_index_desc": "Incrusta documentos en ChromaDB con sentence transformers multilingüe",
        "step_chat": "Chat", "step_chat_desc": "Pregunta lo que quieras — respuestas fundamentadas con fuentes citadas",
        "chat_title": "💬 Chat",
        "chat_sub": "Pregunta cualquier cosa sobre {subject}.",
        "chat_no_config": "Ve a **Configuración** para configurar tu clave API primero.",
        "chat_no_kb": "La base de conocimiento está vacía — el agente usará solo conocimiento general. Ve a **Pipeline** para construirla.",
        "thinking": "Pensando...",
        "sources": "{n} fuentes",
        "clear_conv": "🗑️  Limpiar conversación",
        "chat_input": "Pregunta sobre {subject}...",
        "pipeline_title": "⚙️ Pipeline",
        "pipeline_sub": "Construye la base de conocimiento para <strong>{subject}</strong>.",
        "pipeline_no_config": "Ve a **Configuración** primero.",
        "build_everything_exp": "🚀  **Construir Todo** (recomendado para la primera vez)",
        "ai_queries": "Consultas de búsqueda de IA",
        "fast_mode2": "Modo rápido (omitir estructuración LLM)",
        "reprocess": "Reprocesar documentos ya procesados",
        "rebuild": "Forzar reconstrucción completa de KB",
        "step_research_title": "**① Investigando la web y Wikipedia...**",
        "step_process_title": "**② Procesando documentos (limpiar → PII → estructura)...**",
        "step_build_title": "**③ Construyendo base de conocimiento...**",
        "done_msg": "¡Listo! Tu experto está preparado. Ve a **Chat**.",
        "step_failed": "Un paso falló. Revisa el log anterior.",
        "tab_research": "① Investigar", "tab_process": "② Procesar", "tab_build": "③ Construir KB",
        "research_desc": "**Investigación web con IA** — genera consultas, busca en la web, obtiene Wikipedia.",
        "run_research": "Iniciar Investigación",
        "research_complete": "¡Investigación completa!",
        "research_fail": "Investigación fallida — ver log.",
        "process_desc": "**Limpiar + anonimizar PII + extraer conocimiento estructurado con LLM.**",
        "limit": "Límite (0 = todos)", "fast_model": "Modelo rápido (más económico)",
        "reprocess2": "Reprocesar documentos ya procesados",
        "no_structure": "Sin estructuración LLM (más rápido)",
        "run_process": "Iniciar Procesamiento",
        "process_complete": "¡Procesamiento completo!",
        "process_fail": "Procesamiento fallido — ver log.",
        "build_desc": "**Indexar documentos procesados en ChromaDB para búsqueda semántica.**",
        "force_rebuild": "Forzar reconstrucción completa (eliminar índice existente)",
        "build_kb": "Construir Base de Conocimiento",
        "build_complete": "¡Base de conocimiento construida!",
        "build_fail": "Construcción fallida — ver log.",
        "source_select": "Fuente",
        "max_docs_short": "Máx docs",
        "queries_label": "Consultas",
        "kb_page_title": "🔍 Base de Conocimiento",
        "kb_page_sub": "Busca y explora el conocimiento indexado.",
        "kb_empty_warn": "La base de conocimiento está vacía. Ve a **Pipeline** para construirla.",
        # KB search
        "kb_documents": "Documentos", "kb_learnings": "Conocimientos", "kb_summaries": "Resúmenes",
        "kb_chunks": "fragmentos",
        "kb_search_header": "Buscar",
        "kb_search_input": "Consulta de búsqueda",
        "kb_search_placeholder": "Introduce cualquier pregunta o tema...",
        "kb_results_slider": "Resultados",
        "kb_collection": "Colección",
        "kb_searching": "Buscando...",
        "kb_no_results": "Sin resultados en `{col}`. Esta colección puede estar vacía o falta su índice — ve a **Pipeline → Construir KB** y activa **Forzar reconstrucción completa**.",
        "kb_n_results": "**{n} resultados** en `{col}`",
        "kb_date": "Fecha:",
        "kb_search_error": "No se pudo buscar en `{col}`: {err}\n\nVe a **Pipeline → Construir KB** y activa **Forzar reconstrucción completa** para reparar el índice.",
        # Home KB-ready section
        "kb_ready_goto": "Tu experto está listo. Ve a **Chat** para hacer preguntas.",
        "kb_section_header": "### Base de Conocimiento",
        # Dataset
        "dataset_title": "📦 Conjunto de Datos",
        "dataset_sub": "Visualiza, descarga y publica tu conjunto de datos limpio.",
        "dataset_no_docs": "Aún no hay documentos procesados. Ejecuta primero el Pipeline.",
        "dataset_total": "Total de Documentos",
        "dataset_filter": "Filtrar por título o tema",
        "dataset_matching": "{n} coincidencias",
        "dataset_col_title": "Título", "dataset_col_source": "Fuente",
        "dataset_col_date": "Fecha", "dataset_col_topics": "Temas",
        "dataset_col_structured": "Estructurado",
        "dataset_view_doc": "Ver documento",
        "dataset_source": "**Fuente:**", "dataset_date": "**Fecha:**",
        "dataset_author": "**Autor:**", "dataset_url": "**URL:**",
        "dataset_structured_lbl": "**Estructurado:**",
        "dataset_yes": "Sí", "dataset_no": "No",
        "dataset_summary": "Resumen", "dataset_key_points": "Puntos Clave",
        "dataset_learnings": "Aprendizajes",
        "dataset_download": "⬇️  Descargar JSONL",
        "dataset_hf_warn": "Configura HF_TOKEN en Configuración para habilitar la subida.",
        "dataset_hf_repo": "ID del Repositorio HuggingFace",
        "dataset_private": "Privado",
        "dataset_upload": "⬆️  Subir a HuggingFace",
        "dataset_upload_ok": "¡Subido! Ver en {url}",
        # About
        "about_title": "👤 Acerca de",
        "about_sub": "La persona detrás del Experto en Materia IA.",
        "about_bio": "Constructor de herramientas con IA que hacen el conocimiento experto accesible para todos.",
        "about_project_desc": "El Experto en Materia IA es un proyecto de código abierto que te permite crear un experto en IA completamente fundamentado con RAG sobre <em>cualquier</em> tema — desde exploración espacial hasta derecho de familia — en minutos, usando proveedores de LLM gratuitos o de bajo coste.",
        "about_goal": "El objetivo es sencillo: democratizar el acceso al conocimiento profundo y fiable sin necesitar conocimientos en machine learning ni pagar precios empresariales.",
        "about_project_title": "### Sobre Este Proyecto",
        "about_features": "- Investiga cualquier tema de forma autónoma mediante búsqueda web + Wikipedia\n- Limpia, anonimiza y estructura el conocimiento con LLMs\n- Almacena todo en una base de datos vectorial local (ChromaDB)\n- Responde preguntas con respuestas fundamentadas y citadas — sin alucinaciones\n- Funciona con proveedores gratuitos (Groq, Google Gemini) o de pago (Anthropic, OpenAI)\n- Exporta tu conjunto de datos a JSONL o HuggingFace",
        "about_contact_title": "### Contacto",
        "about_tagline": "Código abierto · Autoalojado<br/>Cualquier tema · Cualquier LLM",
        "about_stack_title": "Stack",
        "about_belief": "Construido con la convicción de que todo el mundo merece acceso al conocimiento de nivel experto.",
        # Pipeline metrics & danger zone
        "pipeline_raw": "Documentos Brutos", "pipeline_processed": "Procesados",
        "pipeline_fetched": "obtenidos", "pipeline_structured": "estructurados",
        "kb_chunks_label": "Fragmentos KB", "kb_indexed": "indexados",
        "danger_title": "🗑️  **Zona de Peligro — Eliminar Todos los Datos**",
        "danger_warn": "**Esto eliminará permanentemente todos los datos recopilados, procesados e indexados.**",
        "danger_sub": "Los documentos brutos, los JSON procesados y toda la base de datos vectorial serán borrados. Tendrás que volver a ejecutar el pipeline completo desde cero.",
        "danger_confirm": "Entiendo que esto no se puede deshacer — eliminar todo",
        "danger_btn": "🗑️  Eliminar Todos los Datos",
        "danger_ok": "Todos los datos han sido eliminados. Ejecuta el pipeline para empezar de nuevo.",
    },
}

def t(key: str) -> str:
    """Return translated UI string for the current language."""
    return _TEXTS.get(ui_lang, _TEXTS["en"]).get(key, _TEXTS["en"].get(key, key))


# ── Preset category labels (Spanish) ──────────────────────────────────────────
# Maps the English key used in PRESETS → Spanish display label for buttons.
# Keys are used as stable session-state identifiers; only display changes.
_PRESET_LABELS_ES: dict = {
    # ── Top level ──────────────────────────────────────────
    "🎨 Custom":                       "🎨 Personalizado",
    "🤖 Artificial Intelligence":      "🤖 Inteligencia Artificial",
    "💻 Cybersecurity":                "💻 Ciberseguridad",
    "📊 Data Science":                 "📊 Ciencia de Datos",
    "⚖️ Law":                          "⚖️ Derecho",
    "🌍 Climate Change":               "🌍 Cambio Climático",
    "💰 Personal Finance":             "💰 Finanzas Personales",
    "⚕️ Nutrition & Health":           "⚕️ Nutrición y Salud",
    "🏋️ Fitness & Exercise":           "🏋️ Fitness y Ejercicio",
    "🧘 Mental Health":                "🧘 Salud Mental",
    "🏥 Medicine & Healthcare":        "🏥 Medicina y Sanidad",
    "🚀 Space Exploration":            "🚀 Exploración Espacial",
    "🧬 Genetics & Genomics":          "🧬 Genética y Genómica",
    "🏛️ History":                      "🏛️ Historia",
    "🎮 Game Development":             "🎮 Desarrollo de Videojuegos",
    "🍳 Cooking & Culinary Arts":      "🍳 Cocina y Artes Culinarias",
    "🚗 Electric Vehicles & Automotive": "🚗 Vehículos Eléctricos y Automoción",
    "🌿 Ecology & Environment":        "🌿 Ecología y Medio Ambiente",
    "🔐 Blockchain & Crypto":          "🔐 Blockchain y Criptomonedas",
    "🎵 Music Theory & Production":    "🎵 Teoría Musical y Producción",
    "📱 Mobile App Development":       "📱 Desarrollo de Apps Móviles",
    "🏗️ Architecture & Engineering":   "🏗️ Arquitectura e Ingeniería",
    # ── AI sub-categories ──────────────────────────────────
    "Machine Learning":                "Aprendizaje Automático",
    "Natural Language Processing":     "Procesamiento del Lenguaje Natural",
    "Computer Vision":                 "Visión Artificial",
    "AI Ethics & Safety":              "Ética y Seguridad en IA",
    "Deep Learning":                   "Aprendizaje Profundo",
    "Reinforcement Learning":          "Aprendizaje por Refuerzo",
    "MLOps":                           "MLOps",
    "Large Language Models":           "Modelos de Lenguaje (LLM)",
    "Text Classification & NER":       "Clasificación de Texto y NER",
    "Speech & Audio NLP":              "NLP de Voz y Audio",
    "Object Detection":                "Detección de Objetos",
    "Image Generation":                "Generación de Imágenes",
    "Medical Imaging":                 "Imágenes Médicas con IA",
    "Bias & Fairness":                 "Sesgo y Equidad",
    "AI Alignment":                    "Alineación de IA",
    "AI Regulation":                   "Regulación de IA",
    # ── Cybersecurity ──────────────────────────────────────
    "Offensive Security":              "Seguridad Ofensiva",
    "Defensive Security":              "Seguridad Defensiva",
    "Cryptography":                    "Criptografía",
    "Penetration Testing":             "Pruebas de Penetración",
    "Social Engineering":              "Ingeniería Social",
    "Malware Analysis":                "Análisis de Malware",
    "Incident Response":               "Respuesta a Incidentes",
    "Threat Intelligence":             "Inteligencia de Amenazas",
    "Cloud Security":                  "Seguridad en la Nube",
    "Symmetric & Asymmetric Encryption": "Cifrado Simétrico y Asimétrico",
    "Post-Quantum Cryptography":       "Criptografía Poscuántica",
    # ── Data Science ───────────────────────────────────────
    "Statistics & Mathematics":        "Estadística y Matemáticas",
    "Data Engineering":                "Ingeniería de Datos",
    "Analytics & BI":                  "Analítica y BI",
    "Bayesian Statistics":             "Estadística Bayesiana",
    "A/B Testing":                     "Pruebas A/B",
    "ETL & Pipelines":                 "ETL y Pipelines",
    "Data Warehousing":                "Data Warehousing",
    "Streaming & Real-Time":           "Streaming en Tiempo Real",
    "Data Visualization":              "Visualización de Datos",
    "Product Analytics":               "Analítica de Producto",
    # ── Law ────────────────────────────────────────────────
    "Family Law":                      "Derecho de Familia",
    "Criminal Law":                    "Derecho Penal",
    "Employment & Labour Law":         "Derecho Laboral",
    "Corporate & Commercial Law":      "Derecho Mercantil",
    "Immigration Law":                 "Derecho de Extranjería",
    "Divorce":                         "Divorcio",
    "Child Custody & Support":         "Custodia y Manutención Infantil",
    "Adoption & Surrogacy":            "Adopción y Gestación Subrogada",
    "Criminal Defence":                "Defensa Penal",
    "White Collar Crime":              "Delitos Económicos",
    "Sentencing & Parole":             "Sentencias y Libertad Condicional",
    "Unfair Dismissal & Redundancy":   "Despido Improcedente y ERE",
    "Workplace Discrimination":        "Discriminación Laboral",
    "Trade Union & Collective":        "Sindicatos y Negociación Colectiva",
    "Mergers & Acquisitions":          "Fusiones y Adquisiciones",
    "Intellectual Property":           "Propiedad Intelectual",
    "Contract Law":                    "Derecho Contractual",
    "Asylum & Refugee Law":            "Derecho de Asilo y Refugio",
    "Work & Skilled Visas":            "Visados de Trabajo",
    "Citizenship & Naturalisation":    "Ciudadanía y Naturalización",
    # ── Climate Change ─────────────────────────────────────
    "Climate Science":                 "Ciencia Climática",
    "Energy Transition":               "Transición Energética",
    "Climate Policy":                  "Política Climática",
    "Atmospheric Science":             "Ciencia Atmosférica",
    "Ocean & Ice Science":             "Ciencia Oceánica y del Hielo",
    "Extreme Weather":                 "Fenómenos Meteorológicos Extremos",
    "Solar Energy":                    "Energía Solar",
    "Wind Energy":                     "Energía Eólica",
    "Energy Storage":                  "Almacenamiento de Energía",
    "Carbon Markets":                  "Mercados de Carbono",
    "International Agreements":        "Acuerdos Internacionales",
    # ── Personal Finance ───────────────────────────────────
    "Investing":                       "Inversión",
    "Budgeting & Debt":                "Presupuesto y Deuda",
    "Retirement Planning":             "Planificación de la Jubilación",
    "Stock Market":                    "Mercado de Valores",
    "Real Estate Investing":           "Inversión Inmobiliaria",
    "Index Funds & ETFs":              "Fondos Indexados y ETFs",
    "Debt Payoff":                     "Liquidación de Deudas",
    "FIRE Movement":                   "Movimiento FIRE",
    "401k & IRA":                      "Planes de Pensiones (EE.UU.)",
    "Pension & UK Retirement":         "Pensión y Jubilación",
    # ── Nutrition & Health ─────────────────────────────────
    "Diets & Eating Patterns":         "Dietas y Patrones Alimentarios",
    "Sports Nutrition":                "Nutrición Deportiva",
    "Gut Health & Microbiome":         "Salud Intestinal y Microbioma",
    "Ketogenic Diet":                  "Dieta Cetogénica",
    "Mediterranean Diet":              "Dieta Mediterránea",
    "Intermittent Fasting":            "Ayuno Intermitente",
    "Protein & Muscle":                "Proteína y Músculo",
    "Endurance Nutrition":             "Nutrición para Resistencia",
    "Probiotics & Prebiotics":         "Probióticos y Prebióticos",
    "IBS & Digestive Disorders":       "SII y Trastornos Digestivos",
    # ── Fitness & Exercise ─────────────────────────────────
    "Strength Training":               "Entrenamiento de Fuerza",
    "Cardio & Endurance":              "Cardio y Resistencia",
    "Recovery & Flexibility":          "Recuperación y Flexibilidad",
    "Powerlifting":                    "Powerlifting",
    "Bodybuilding":                    "Culturismo",
    "Calisthenics":                    "Calistenia",
    "Running":                         "Running",
    "Cycling":                         "Ciclismo",
    "HIIT":                            "HIIT",
    "Mobility & Stretching":           "Movilidad y Estiramientos",
    "Sleep & Recovery Science":        "Ciencia del Sueño y Recuperación",
    # ── Mental Health ──────────────────────────────────────
    "Therapy Approaches":              "Enfoques Terapéuticos",
    "Mental Health Conditions":        "Condiciones de Salud Mental",
    "Mindfulness & Stress":            "Mindfulness y Estrés",
    "CBT":                             "TCC (Terapia Cognitivo-Conductual)",
    "DBT":                             "TDC (Terapia Dialéctica Conductual)",
    "ACT":                             "ACT (Terapia de Aceptación y Compromiso)",
    "Anxiety Disorders":               "Trastornos de Ansiedad",
    "Depression":                      "Depresión",
    "ADHD":                            "TDAH",
    "Meditation":                      "Meditación",
    "Burnout":                         "Burnout",
    # ── Medicine & Healthcare ──────────────────────────────
    "Internal Medicine":               "Medicina Interna",
    "Pharmacology":                    "Farmacología",
    "Public Health":                   "Salud Pública",
    "Cardiology":                      "Cardiología",
    "Endocrinology":                   "Endocrinología",
    "Pulmonology":                     "Neumología",
    "Antibiotics & Antimicrobials":    "Antibióticos y Antimicrobianos",
    "Drug Development":                "Desarrollo de Fármacos",
    "Epidemiology":                    "Epidemiología",
    "Vaccines & Immunology":           "Vacunas e Inmunología",
    "Global Health":                   "Salud Global",
    # ── Space Exploration ──────────────────────────────────
    "Solar System":                    "Sistema Solar",
    "Deep Space & Astrophysics":       "Espacio Profundo y Astrofísica",
    "Space Technology":                "Tecnología Espacial",
    "Mars Exploration":                "Exploración de Marte",
    "Moon & Lunar Missions":           "Misiones Lunares",
    "Asteroid & Comet Science":        "Asteroides y Cometas",
    "Black Holes":                     "Agujeros Negros",
    "Exoplanets":                      "Exoplanetas",
    "Rocket Propulsion":               "Propulsión de Cohetes",
    "Satellites & Constellations":     "Satélites y Constelaciones",
    # ── Genetics & Genomics ────────────────────────────────
    "Molecular Biology":               "Biología Molecular",
    "Genomics & Sequencing":           "Genómica y Secuenciación",
    "Genetic Engineering":             "Ingeniería Genética",
    "Gene Expression":                 "Expresión Génica",
    "Protein Structure":               "Estructura de Proteínas",
    "Bioinformatics":                  "Bioinformática",
    "Cancer Genomics":                 "Genómica del Cáncer",
    "CRISPR":                          "CRISPR",
    "Gene Therapy":                    "Terapia Génica",
    # ── History ────────────────────────────────────────────
    "Ancient History":                 "Historia Antigua",
    "Medieval History":                "Historia Medieval",
    "Modern History":                  "Historia Moderna",
    "Ancient Greece":                  "Antigua Grecia",
    "Ancient Rome":                    "Roma Antigua",
    "Ancient Egypt":                   "Antiguo Egipto",
    "Crusades":                        "Las Cruzadas",
    "Islamic Golden Age":              "Época Dorada Islámica",
    "World War II":                    "Segunda Guerra Mundial",
    "Cold War":                        "Guerra Fría",
    "Decolonisation":                  "Descolonización",
    # ── Game Development ───────────────────────────────────
    "Game Engines":                    "Motores de Juego",
    "Game Design":                     "Diseño de Juegos",
    "Unity":                           "Unity",
    "Unreal Engine":                   "Unreal Engine",
    "Godot":                           "Godot",
    # ── Cooking & Culinary Arts ────────────────────────────
    "Cooking Techniques":              "Técnicas Culinarias",
    "Baking & Pastry":                 "Repostería y Pastelería",
    "World Cuisines":                  "Cocinas del Mundo",
    "French Techniques":               "Técnicas Francesas",
    "Fermentation":                    "Fermentación",
    "Molecular Gastronomy":            "Gastronomía Molecular",
    "Bread Baking":                    "Panadería",
    "Chocolate & Confectionery":       "Chocolate y Confitería",
    "Japanese Cuisine":                "Cocina Japonesa",
    "Indian Cuisine":                  "Cocina India",
    # ── EV & Automotive ────────────────────────────────────
    "EV Technology":                   "Tecnología de VE",
    "Autonomous Driving":              "Conducción Autónoma",
    "Battery Systems":                 "Sistemas de Baterías",
    "Charging Infrastructure":         "Infraestructura de Carga",
    "Sensor Systems":                  "Sistemas de Sensores",
    "AI & Path Planning":              "IA y Planificación de Rutas",
    # ── Ecology & Environment ──────────────────────────────
    "Biodiversity & Conservation":     "Biodiversidad y Conservación",
    "Pollution & Waste":               "Contaminación y Residuos",
    "Rewilding":                       "Rewilding",
    "Marine Conservation":             "Conservación Marina",
    "Plastic Pollution":               "Contaminación Plástica",
    "Air Quality":                     "Calidad del Aire",
    # ── Blockchain & Crypto ────────────────────────────────
    "Cryptocurrencies":                "Criptomonedas",
    "DeFi":                            "DeFi",
    "Bitcoin":                         "Bitcoin",
    "Ethereum & Smart Contracts":      "Ethereum y Contratos Inteligentes",
    "Lending & Borrowing":             "Préstamos y Créditos DeFi",
    "DEX & AMM":                       "DEX y AMM",
    # ── Music Theory & Production ──────────────────────────
    "Music Theory":                    "Teoría Musical",
    "Audio Production":                "Producción de Audio",
    "Harmony & Chords":                "Armonía y Acordes",
    "Counterpoint":                    "Contrapunto",
    "Ear Training":                    "Entrenamiento Auditivo",
    "Mixing":                          "Mezcla y Masterización",
    "Sound Synthesis":                 "Síntesis de Sonido",
    # ── Mobile App Development ─────────────────────────────
    "iOS Development":                 "Desarrollo iOS",
    "Android Development":             "Desarrollo Android",
    "Cross-Platform":                  "Multiplataforma",
    "SwiftUI":                         "SwiftUI",
    "Core Data & Persistence":         "Core Data y Persistencia",
    "Jetpack Compose":                 "Jetpack Compose",
    "Kotlin Coroutines":               "Coroutines de Kotlin",
    "React Native":                    "React Native",
    "Flutter":                         "Flutter",
    # ── Architecture & Engineering ─────────────────────────
    "Structural Engineering":          "Ingeniería Estructural",
    "Sustainable Design":              "Diseño Sostenible",
    "Urban Planning":                  "Urbanismo",
    "Seismic Design":                  "Diseño Sísmico",
    "Materials Science":               "Ciencia de Materiales",
    "Passive House":                   "Casa Pasiva (Passivhaus)",
    "Net Zero Buildings":              "Edificios de Emisiones Netas Cero",
    "Smart Cities":                    "Ciudades Inteligentes",
    "Housing & Density":               "Vivienda y Densidad Urbana",
}

# Spanish subject / desc / kws for preset nodes.
# Used instead of English values when LANGUAGE=es and user picks a preset.
_PRESET_META_ES: dict = {
    "🎨 Custom":                    {"subject": "", "desc": "", "kws": ""},
    "🤖 Artificial Intelligence":   {"subject": "inteligencia artificial", "desc": "inteligencia artificial, aprendizaje automático y deep learning", "kws": "inteligencia artificial,aprendizaje automático,deep learning,redes neuronales,LLM,transformers,GPT"},
    "Machine Learning":             {"subject": "aprendizaje automático", "desc": "algoritmos de aprendizaje automático, entrenamiento y evaluación de modelos", "kws": "aprendizaje automático,aprendizaje supervisado,no supervisado,gradiente descendente,sobreajuste,validación cruzada"},
    "Deep Learning":                {"subject": "aprendizaje profundo", "desc": "redes neuronales profundas, CNN, RNN y transformers", "kws": "aprendizaje profundo,red neuronal,CNN,RNN,transformer,retropropagación,GPU,PyTorch,TensorFlow"},
    "Reinforcement Learning":       {"subject": "aprendizaje por refuerzo", "desc": "aprendizaje por refuerzo, agentes y optimización de recompensas", "kws": "aprendizaje por refuerzo,Q-learning,gradiente de política,recompensa,agente,entorno,PPO,DQN"},
    "Natural Language Processing":  {"subject": "procesamiento del lenguaje natural", "desc": "PLN, análisis de texto y modelos de lenguaje", "kws": "PLN,clasificación de texto,reconocimiento de entidades,análisis de sentimientos,tokenización,BERT,GPT,modelo de lenguaje"},
    "Large Language Models":        {"subject": "modelos de lenguaje de gran escala", "desc": "LLMs, ingeniería de prompts y ajuste fino", "kws": "LLM,GPT-4,Claude,Llama,ingeniería de prompts,ajuste fino,RAG,RLHF,ajuste por instrucciones"},
    "Computer Vision":              {"subject": "visión artificial", "desc": "visión por computador, reconocimiento de imágenes e IA visual", "kws": "visión artificial,clasificación de imágenes,detección de objetos,segmentación,CNN,YOLO,OpenCV,generación de imágenes"},
    "AI Ethics & Safety":           {"subject": "ética y seguridad en IA", "desc": "ética en IA, sesgo, equidad e IA responsable", "kws": "ética IA,sesgo,equidad,explicabilidad,responsabilidad,transparencia,seguridad IA,alineación,regulación"},
    "💻 Cybersecurity":             {"subject": "ciberseguridad", "desc": "ciberseguridad, hacking ético y defensa digital", "kws": "ciberseguridad,hacking,malware,firewall,cifrado,phishing,zero-day,pentesting,SIEM"},
    "Penetration Testing":          {"subject": "pruebas de penetración", "desc": "pentesting web, red y aplicaciones", "kws": "pentesting,hacking ético,aplicación web,inyección SQL,XSS,OWASP,Burp Suite,escalada de privilegios"},
    "📊 Data Science":              {"subject": "ciencia de datos", "desc": "ciencia de datos, estadística y pipelines de machine learning", "kws": "ciencia de datos,pandas,estadística,regresión,clasificación,ingeniería de características,EDA,Jupyter,SQL"},
    "Statistics & Mathematics":     {"subject": "estadística para ciencia de datos", "desc": "métodos estadísticos, probabilidad y fundamentos matemáticos", "kws": "estadística,probabilidad,prueba de hipótesis,bayesiano,regresión,distribución,intervalo de confianza,prueba A/B"},
    "⚖️ Law":                       {"subject": "derecho", "desc": "principios jurídicos, jurisprudencia y práctica legal", "kws": "derecho,legal,tribunal,jurisdicción,ley,precedente,litigio,abogado,derechos"},
    "Family Law":                   {"subject": "derecho de familia", "desc": "derecho de familia, relaciones familiares y bienestar infantil", "kws": "derecho de familia,divorcio,custodia,manutención infantil,adopción,violencia doméstica,tutela,bienes gananciales"},
    "Divorce":                      {"subject": "divorcio", "desc": "procedimientos de divorcio, causas y acuerdos económicos", "kws": "divorcio,separación,causas del divorcio,liquidación económica,bienes gananciales,acuerdo de separación,pensión compensatoria,convenio regulador"},
    "Child Custody & Support":      {"subject": "custodia y manutención infantil", "desc": "régimen de custodia, planes de parentalidad y pensión alimenticia", "kws": "custodia infantil,guarda y custodia,régimen de visitas,pensión alimenticia,interés superior del menor,custodia compartida,patria potestad,mediación familiar"},
    "Adoption & Surrogacy":         {"subject": "adopción y gestación subrogada", "desc": "procedimientos de adopción, gestación subrogada y derechos parentales", "kws": "adopción,gestación subrogada,maternidad subrogada,derechos parentales,acogimiento,adopción internacional,filiación"},
    "Criminal Law":                 {"subject": "derecho penal", "desc": "delitos, procedimiento penal y defensa", "kws": "derecho penal,delito,acusación,defensa,sentencia,prisión preventiva,juicio oral,jurado,más allá de toda duda razonable"},
    "Employment & Labour Law":      {"subject": "derecho laboral", "desc": "derechos laborales, conflictos laborales y relaciones colectivas", "kws": "derecho laboral,despido improcedente,ERE,discriminación,tribunal laboral,convenio colectivo,sindicato,salario,contrato"},
    "Corporate & Commercial Law":   {"subject": "derecho mercantil", "desc": "derecho societario, contratos y transacciones comerciales", "kws": "derecho mercantil,sociedad,administrador,accionista,contrato,fusiones y adquisiciones,propiedad intelectual,due diligence,estatutos"},
    "Immigration Law":              {"subject": "derecho de extranjería", "desc": "visados, asilo, ciudadanía y procedimiento de extranjería", "kws": "extranjería,visado,asilo,refugiado,deportación,ciudadanía,naturalización,permiso de residencia,extranjero"},
    "Intellectual Property":        {"subject": "propiedad intelectual", "desc": "patentes, marcas, derechos de autor y protección de la PI", "kws": "propiedad intelectual,patente,marca registrada,derechos de autor,secreto comercial,licencia,infracción,OEPM,diseño industrial"},
    "Contract Law":                 {"subject": "derecho contractual", "desc": "formación de contratos, incumplimiento y remedios", "kws": "contrato,oferta,aceptación,causa,incumplimiento,daños y perjuicios,cumplimiento específico,resolución,vicios del consentimiento"},
    "🌍 Climate Change":            {"subject": "cambio climático", "desc": "ciencia climática, calentamiento global y política ambiental", "kws": "cambio climático,calentamiento global,CO2,gas de efecto invernadero,IPCC,carbono,energías renovables"},
    "Solar Energy":                 {"subject": "energía solar", "desc": "fotovoltaica, instalaciones solares y tecnología solar", "kws": "solar,fotovoltaico,panel solar,instalación solar,inversor,LCOE,autoconsumo,energía solar concentrada,perovskita"},
    "Wind Energy":                  {"subject": "energía eólica", "desc": "energía eólica terrestre y marina", "kws": "energía eólica,aerogenerador,eólica marina,eólica terrestre,factor de capacidad,parque eólico,LCOE,pala"},
    "💰 Personal Finance":          {"subject": "finanzas personales", "desc": "finanzas personales, inversión, presupuesto y gestión patrimonial", "kws": "inversión,acciones,bonos,ETF,presupuesto,ahorro,jubilación,interés compuesto"},
    "Investing":                    {"subject": "inversión", "desc": "bolsa de valores, bonos y estrategias de inversión", "kws": "inversión,acciones,bonos,ETF,fondo indexado,cartera,diversificación,asignación de activos,rentabilidad"},
    "Mediterranean Diet":           {"subject": "dieta mediterránea", "desc": "alimentación mediterránea, aceite de oliva e investigación sobre longevidad", "kws": "dieta mediterránea,aceite de oliva,pescado,legumbres,cereales integrales,antioxidantes,longevidad,salud cardiovascular"},
    "⚕️ Nutrition & Health":        {"subject": "nutrición y salud", "desc": "ciencia nutricional, alimentación saludable y salud preventiva", "kws": "nutrición,dieta,vitaminas,proteínas,carbohidratos,metabolismo,salud intestinal,suplementos"},
    "🏋️ Fitness & Exercise":        {"subject": "fitness y ejercicio", "desc": "fitness, ciencia del ejercicio y rendimiento deportivo", "kws": "ejercicio,entrenamiento de fuerza,cardio,HIIT,músculo,recuperación,VO2 máximo,flexibilidad,rendimiento deportivo"},
    "Strength Training":            {"subject": "entrenamiento de fuerza", "desc": "entrenamiento de resistencia, sobrecarga progresiva e hipertrofia", "kws": "entrenamiento de fuerza,sobrecarga progresiva,powerlifting,hipertrofia,1RM,ejercicios compuestos,periodización"},
    "🧘 Mental Health":             {"subject": "salud mental", "desc": "salud mental, psicología y bienestar", "kws": "salud mental,ansiedad,depresión,terapia,TCC,mindfulness,estrés,psiquiatría,resiliencia"},
    "CBT":                          {"subject": "terapia cognitivo-conductual", "desc": "técnicas de TCC, registros de pensamiento y activación conductual", "kws": "TCC,distorsiones cognitivas,registro de pensamiento,activación conductual,exposición,formulación,pensamientos automáticos"},
    "Anxiety Disorders":            {"subject": "trastornos de ansiedad", "desc": "ansiedad generalizada, trastorno de pánico, fobias y tratamiento", "kws": "ansiedad,trastorno de pánico,TAG,ansiedad social,fobia,agorafobia,terapia de exposición,preocupación"},
    "Depression":                   {"subject": "depresión", "desc": "depresión mayor, distimia, causas y tratamiento", "kws": "depresión,TDM,antidepresivo,ISRS,anhedonia,estado de ánimo bajo,terapia cognitiva,biológico,psicosocial"},
    "🏥 Medicine & Healthcare":     {"subject": "medicina y sanidad", "desc": "medicina, práctica clínica y sistemas sanitarios", "kws": "medicina,diagnóstico,tratamiento,ensayo clínico,farmacología,cirugía,epidemiología,sistema sanitario,OMS"},
    "Public Health":                {"subject": "salud pública", "desc": "epidemiología, vacunación y salud global", "kws": "salud pública,epidemiología,vacuna,pandemia,vigilancia epidemiológica,incidencia,prevalencia,determinantes sociales,OMS"},
    "🚀 Space Exploration":         {"subject": "exploración espacial", "desc": "ciencia espacial, astronomía y misiones espaciales", "kws": "NASA,SpaceX,Marte,luna,cohete,satélite,asteroide,telescopio,cosmología"},
    "Mars Exploration":             {"subject": "exploración de Marte", "desc": "misiones a Marte, geología marciana y futuros asentamientos", "kws": "Marte,rover,Perseverance,Curiosity,terraformación,muestra de Marte,habitabilidad,metano"},
    "🧬 Genetics & Genomics":       {"subject": "genética y genómica", "desc": "genética, genómica y biología molecular", "kws": "ADN,gen,genoma,mutación,CRISPR,herencia,cromosoma,proteína,ARN"},
    "CRISPR":                       {"subject": "edición génica CRISPR", "desc": "mecanismos CRISPR-Cas9, entrega y aplicaciones", "kws": "CRISPR,Cas9,ARN guía,HDR,NHEJ,editor de bases,edición primaria,fuera de objetivo,entrega,in vivo"},
    "🏛️ History":                   {"subject": "historia", "desc": "historia mundial, civilizaciones y eventos históricos", "kws": "historia,civilización,guerra,imperio,revolución,arqueología,antigüedad,medieval,historia moderna"},
    "Ancient Greece":               {"subject": "Antigua Grecia", "desc": "ciudades-estado griegas, filosofía, democracia y cultura", "kws": "Grecia,Atenas,Esparta,democracia,filosofía,Sócrates,Platón,Aristóteles,Guerra del Peloponeso,helenístico"},
    "World War II":                 {"subject": "Segunda Guerra Mundial", "desc": "causas de la Segunda Guerra Mundial, batallas principales y consecuencias", "kws": "Segunda Guerra Mundial,Hitler,Churchill,Día D,Holocausto,Guerra del Pacífico,bomba atómica,Normandía,frente oriental"},
    "🎮 Game Development":          {"subject": "desarrollo de videojuegos", "desc": "desarrollo de videojuegos, diseño de juegos y motores gráficos", "kws": "Unity,Unreal Engine,diseño de juegos,mecánicas,shader,motor de física,generación procedural,juego indie"},
    "🍳 Cooking & Culinary Arts":   {"subject": "cocina y artes culinarias", "desc": "técnicas culinarias, recetas y ciencia de los alimentos", "kws": "cocina,receta,repostería,fermentación,habilidades con el cuchillo,reacción de Maillard,pastelería,cuisine,maridaje de sabores"},
    "Mediterranean Diet":           {"subject": "dieta mediterránea", "desc": "cocina mediterránea, ingredientes y técnicas regionales", "kws": "cocina mediterránea,aceite de oliva,pescado,legumbres,cereales,especias,longevidad,dieta española"},
    "🚗 Electric Vehicles & Automotive": {"subject": "vehículos eléctricos y automoción", "desc": "vehículos eléctricos, tecnología del automóvil y transporte", "kws": "vehículo eléctrico,VE,batería,Tesla,carga,autonomía,conducción autónoma,litio,par motor"},
    "🌿 Ecology & Environment":     {"subject": "ecología y medio ambiente", "desc": "ecología, biodiversidad y ciencias medioambientales", "kws": "ecología,biodiversidad,ecosistema,conservación,deforestación,especies,hábitat,contaminación,rewilding"},
    "🔐 Blockchain & Crypto":       {"subject": "blockchain y criptomonedas", "desc": "tecnología blockchain, criptomonedas y finanzas descentralizadas", "kws": "blockchain,Bitcoin,Ethereum,DeFi,contrato inteligente,NFT,consenso,monedero,Web3"},
    "Bitcoin":                      {"subject": "Bitcoin", "desc": "protocolo Bitcoin, minería y reserva de valor", "kws": "Bitcoin,BTC,minería,SHA-256,halving,Lightning Network,UTXO,mempool,nodo completo,Satoshi Nakamoto"},
    "🎵 Music Theory & Production": {"subject": "teoría musical y producción", "desc": "teoría musical, composición y producción de audio", "kws": "teoría musical,armonía,progresión de acordes,mezcla,masterización,DAW,síntesis,arreglos,entrenamiento auditivo"},
    "📱 Mobile App Development":    {"subject": "desarrollo de aplicaciones móviles", "desc": "desarrollo de apps iOS y Android, React Native y Flutter", "kws": "iOS,Android,React Native,Flutter,Swift,Kotlin,UI móvil,App Store,notificaciones push"},
    "🏗️ Architecture & Engineering": {"subject": "arquitectura e ingeniería", "desc": "arquitectura, ingeniería estructural y construcción", "kws": "arquitectura,ingeniería estructural,BIM,AutoCAD,estructura portante,cimentación,diseño sostenible,HVAC"},
    "Smart Cities":                 {"subject": "ciudades inteligentes", "desc": "IoT, datos y tecnología para la gestión urbana", "kws": "ciudad inteligente,IoT,sensores,plataforma de datos,movilidad,gemelo digital,gestión del tráfico,red energética,ciudadano"},
    "Passive House":                {"subject": "casa pasiva (Passivhaus)", "desc": "estándar Passivhaus, envolvente térmica y eficiencia energética", "kws": "casa pasiva,Passivhaus,PHPP,estanqueidad al aire,puente térmico,MVHR,triple acristalamiento,demanda de calefacción"},
}

def preset_label(key: str) -> str:
    """Return display label for a preset key, translated when UI is Spanish."""
    if ui_lang == "es":
        return _PRESET_LABELS_ES.get(key, key)
    return key

def get_preset_meta(node: dict, key: str = "") -> dict:
    """Return subject/desc/kws for the active language."""
    if ui_lang == "es" and key in _PRESET_META_ES:
        return _PRESET_META_ES[key]
    return {
        "subject": node.get("_subject", ""),
        "desc":    node.get("_desc", ""),
        "kws":     node.get("_kws", ""),
    }


# ── Navigation ────────────────────────────────────────────────────────────────
configured = is_configured()
raw_n   = count_json(ROOT / "data" / "raw")
proc_n  = count_json(ROOT / "data" / "processed")

# Auto-land on Setup if not configured
default_page = "Setup" if not configured else "Home"

with st.sidebar:
    # ── Language toggle (top, immediate effect) ───────────────────────────────
    _lang_col1, _lang_col2 = st.columns(2)
    with _lang_col1:
        if st.button("🇬🇧 English", use_container_width=True,
                     type="primary" if ui_lang == "en" else "secondary"):
            if ui_lang != "en":
                write_env({"LANGUAGE": "en"})
                st.rerun()
    with _lang_col2:
        if st.button("🇪🇸 Español", use_container_width=True,
                     type="primary" if ui_lang == "es" else "secondary"):
            if ui_lang != "es":
                write_env({"LANGUAGE": "es"})
                st.rerun()
    st.markdown("---")

    st.markdown(f"## 🧠 {t('app_title')}")
    subject = get_env("SUBJECT", "—")
    st.caption(f"{t('subject_label')} **{subject}**")
    st.markdown("---")

    page_keys = ["Home", "Chat", "Pipeline", "Knowledge Base", "Dataset", "Setup", "About"]
    page_labels = [
        t("nav_home"), t("nav_chat"), t("nav_pipeline"),
        t("nav_kb"), t("nav_dataset"), t("nav_setup"), t("nav_about"),
    ]
    icons  = ["🏠", "💬", "⚙️", "🔍", "📦", "🛠️", "👤"]
    labels = [f"{i}  {p}" for i, p in zip(icons, page_labels)]

    default_idx = page_keys.index(default_page)
    choice = st.radio("", labels, index=default_idx, label_visibility="collapsed")
    page = page_keys[labels.index(choice)]

    st.markdown("---")
    st.caption(f"{t('raw_docs')} **{raw_n:,}**")
    st.caption(f"{t('processed')} **{proc_n:,}**")
    if configured:
        provider = (
            "Anthropic" if get_env("ANTHROPIC_API_KEY") else
            "Google"    if get_env("GOOGLE_API_KEY")    else
            "Groq"      if get_env("GROQ_API_KEY")      else "OpenAI"
        )
        st.caption(f"LLM: **{provider}**")
    else:
        st.caption(t("not_configured"))


# ══════════════════════════════════════════════════════════════════════════════
# SETUP PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Setup":
    st.markdown(f'<div class="page-title">{t("setup_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("setup_sub")}</div>', unsafe_allow_html=True)

    # ── Step 1: LLM ──────────────────────────────────────────────────────────
    st.markdown(f"### {t('step1_title')}")
    st.markdown(t("step1_desc"))

    provider = st.radio(
        "Provider",
        ["Groq (FREE — Llama 3.3)", "Google (Gemini — free tier)", "Anthropic (Claude)", "OpenAI (GPT-4)"],
        horizontal=True,
        index=0,
    )

    FREE_LINKS = {
        "Groq (FREE — Llama 3.3)":        "https://console.groq.com/keys",
        "Google (Gemini — free tier)":     "https://aistudio.google.com/apikey",
    }
    if provider in FREE_LINKS:
        st.caption(f"Get a free API key → [{FREE_LINKS[provider]}]({FREE_LINKS[provider]})")

    key_map = {
        "Groq (FREE — Llama 3.3)":        ("GROQ_API_KEY",       "gsk_..."),
        "Google (Gemini — free tier)":     ("GOOGLE_API_KEY",     "AIza..."),
        "Anthropic (Claude)":              ("ANTHROPIC_API_KEY",  "sk-ant-..."),
        "OpenAI (GPT-4)":                  ("OPENAI_API_KEY",     "sk-..."),
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
    st.markdown(f"### {t('step2_title')}")
    st.markdown(t("step2_desc"))

    # Each node: _subject/_desc/_kws = this level's config; other keys = subcategories
    PRESETS = {
        "🎨 Custom": {
            "_subject": "", "_desc": "", "_kws": "",
        },
        "🤖 Artificial Intelligence": {
            "_subject": "artificial intelligence",
            "_desc": "artificial intelligence, machine learning, and deep learning",
            "_kws": "artificial intelligence,machine learning,deep learning,neural networks,LLM,transformers,GPT",
            "Machine Learning": {
                "_subject": "machine learning",
                "_desc": "machine learning algorithms, model training, and evaluation",
                "_kws": "machine learning,supervised learning,unsupervised learning,gradient descent,overfitting,cross-validation",
                "Deep Learning": {"_subject": "deep learning", "_desc": "deep neural networks, CNNs, RNNs, and transformers", "_kws": "deep learning,neural network,CNN,RNN,transformer,backpropagation,GPU,PyTorch,TensorFlow"},
                "Reinforcement Learning": {"_subject": "reinforcement learning", "_desc": "reinforcement learning, agents, and reward optimization", "_kws": "reinforcement learning,Q-learning,policy gradient,reward,agent,environment,OpenAI Gym,PPO,DQN"},
                "MLOps": {"_subject": "MLOps", "_desc": "machine learning operations, deployment, and monitoring", "_kws": "MLOps,model deployment,feature store,monitoring,drift,CI/CD,Kubeflow,MLflow,Docker"},
            },
            "Natural Language Processing": {
                "_subject": "natural language processing",
                "_desc": "NLP, text analysis, and language models",
                "_kws": "NLP,text classification,named entity recognition,sentiment analysis,tokenization,BERT,GPT,language model",
                "Large Language Models": {"_subject": "large language models", "_desc": "LLMs, prompt engineering, and fine-tuning", "_kws": "LLM,GPT-4,Claude,Llama,prompt engineering,fine-tuning,RAG,RLHF,instruction tuning"},
                "Text Classification & NER": {"_subject": "text classification and NER", "_desc": "text classification, named entity recognition, and information extraction", "_kws": "text classification,NER,information extraction,span,sequence labeling,spaCy,BERT,zero-shot"},
                "Speech & Audio NLP": {"_subject": "speech and audio NLP", "_desc": "speech recognition, text-to-speech, and audio processing", "_kws": "speech recognition,ASR,TTS,Whisper,voice,phoneme,prosody,audio,wav2vec"},
            },
            "Computer Vision": {
                "_subject": "computer vision",
                "_desc": "computer vision, image recognition, and visual AI",
                "_kws": "computer vision,image classification,object detection,segmentation,CNN,YOLO,OpenCV,image generation",
                "Object Detection": {"_subject": "object detection", "_desc": "object detection, bounding boxes, and real-time vision", "_kws": "object detection,YOLO,Faster R-CNN,anchor box,bounding box,mAP,real-time detection,COCO"},
                "Image Generation": {"_subject": "AI image generation", "_desc": "diffusion models, GANs, and generative visual AI", "_kws": "image generation,diffusion model,GAN,Stable Diffusion,DALL-E,Midjourney,latent space,VAE"},
                "Medical Imaging": {"_subject": "medical imaging AI", "_desc": "AI for radiology, pathology, and clinical imaging", "_kws": "medical imaging,radiology,CT scan,MRI,X-ray,segmentation,DICOM,diagnostic AI,pathology"},
            },
            "AI Ethics & Safety": {
                "_subject": "AI ethics and safety",
                "_desc": "AI ethics, bias, fairness, and responsible AI",
                "_kws": "AI ethics,bias,fairness,explainability,accountability,transparency,AI safety,alignment,regulation",
                "Bias & Fairness": {"_subject": "AI bias and fairness", "_desc": "algorithmic bias, fairness metrics, and equitable AI", "_kws": "algorithmic bias,fairness,disparate impact,demographic parity,equalized odds,debiasing,representation"},
                "AI Alignment": {"_subject": "AI alignment", "_desc": "AI alignment, value learning, and existential risk", "_kws": "AI alignment,value learning,reward hacking,interpretability,corrigibility,RLHF,existential risk,AGI safety"},
                "AI Regulation": {"_subject": "AI regulation and policy", "_desc": "AI laws, governance frameworks, and policy", "_kws": "AI Act,EU regulation,AI governance,GDPR,liability,auditing,standards,policy,compliance"},
            },
        },
        "💻 Cybersecurity": {
            "_subject": "cybersecurity",
            "_desc": "cybersecurity, ethical hacking, and digital defense",
            "_kws": "cybersecurity,hacking,malware,firewall,encryption,phishing,zero-day,penetration testing,SIEM",
            "Offensive Security": {
                "_subject": "offensive security",
                "_desc": "penetration testing, ethical hacking, and red teaming",
                "_kws": "penetration testing,ethical hacking,red team,exploit,vulnerability,Metasploit,CTF,OSCP,bug bounty",
                "Penetration Testing": {"_subject": "penetration testing", "_desc": "web, network, and application penetration testing", "_kws": "pentest,web application,SQL injection,XSS,OWASP,Burp Suite,network scan,privilege escalation"},
                "Social Engineering": {"_subject": "social engineering", "_desc": "phishing, pretexting, and human-factor attacks", "_kws": "social engineering,phishing,spear phishing,vishing,pretexting,human factor,awareness training"},
                "Malware Analysis": {"_subject": "malware analysis", "_desc": "reverse engineering, malware families, and threat analysis", "_kws": "malware,reverse engineering,ransomware,trojan,rootkit,sandbox,static analysis,dynamic analysis,IDA Pro"},
            },
            "Defensive Security": {
                "_subject": "defensive security",
                "_desc": "incident response, threat detection, and blue team operations",
                "_kws": "incident response,SIEM,blue team,threat hunting,SOC,forensics,log analysis,EDR,firewall",
                "Incident Response": {"_subject": "incident response", "_desc": "IR playbooks, forensics, and breach containment", "_kws": "incident response,forensics,containment,eradication,recovery,playbook,chain of custody,DFIR"},
                "Threat Intelligence": {"_subject": "cyber threat intelligence", "_desc": "CTI, threat actors, and intelligence frameworks", "_kws": "threat intelligence,CTI,MITRE ATT&CK,IOC,TTPs,threat actor,APT,dark web,intelligence sharing"},
                "Cloud Security": {"_subject": "cloud security", "_desc": "AWS/Azure/GCP security, IAM, and zero trust", "_kws": "cloud security,AWS,Azure,GCP,IAM,zero trust,misconfiguration,CSPM,DevSecOps,S3 bucket"},
            },
            "Cryptography": {
                "_subject": "cryptography",
                "_desc": "encryption, cryptographic protocols, and key management",
                "_kws": "cryptography,encryption,AES,RSA,TLS,PKI,hash,digital signature,key exchange,elliptic curve",
                "Symmetric & Asymmetric Encryption": {"_subject": "encryption algorithms", "_desc": "symmetric, asymmetric, and hybrid encryption", "_kws": "AES,RSA,Diffie-Hellman,elliptic curve,block cipher,stream cipher,key exchange,hybrid encryption"},
                "Post-Quantum Cryptography": {"_subject": "post-quantum cryptography", "_desc": "quantum-resistant algorithms and NIST PQC standards", "_kws": "post-quantum,quantum computer,lattice-based,CRYSTALS-Kyber,CRYSTALS-Dilithium,NIST PQC,Shor algorithm"},
            },
        },
        "📊 Data Science": {
            "_subject": "data science",
            "_desc": "data science, statistics, and machine learning pipelines",
            "_kws": "data science,pandas,statistics,regression,classification,feature engineering,EDA,Jupyter,SQL",
            "Statistics & Mathematics": {
                "_subject": "statistics for data science",
                "_desc": "statistical methods, probability, and mathematical foundations",
                "_kws": "statistics,probability,hypothesis testing,Bayesian,regression,distribution,confidence interval,A/B test",
                "Bayesian Statistics": {"_subject": "Bayesian statistics", "_desc": "Bayesian inference, priors, and probabilistic models", "_kws": "Bayesian,prior,posterior,MCMC,PyMC,Stan,credible interval,likelihood,conjugate prior"},
                "A/B Testing": {"_subject": "A/B testing and experimentation", "_desc": "experiment design, statistical significance, and causal inference", "_kws": "A/B test,experiment,p-value,power,sample size,significance,conversion rate,causal inference,uplift"},
            },
            "Data Engineering": {
                "_subject": "data engineering",
                "_desc": "ETL pipelines, data warehousing, and big data",
                "_kws": "ETL,pipeline,Spark,Airflow,dbt,data warehouse,Kafka,Snowflake,BigQuery,Redshift",
                "ETL & Pipelines": {"_subject": "ETL pipelines", "_desc": "data ingestion, transformation, and orchestration", "_kws": "ETL,ELT,Airflow,dbt,data pipeline,ingestion,Fivetran,orchestration,DAG,data quality"},
                "Data Warehousing": {"_subject": "data warehousing", "_desc": "cloud data warehouses, dimensional modeling, and analytics", "_kws": "data warehouse,Snowflake,BigQuery,Redshift,dimensional modeling,star schema,OLAP,partitioning,query optimization"},
                "Streaming & Real-Time": {"_subject": "streaming data", "_desc": "real-time data processing with Kafka and Flink", "_kws": "Kafka,Flink,Spark Streaming,real-time,event-driven,message queue,CDC,watermark,window function"},
            },
            "Analytics & BI": {
                "_subject": "business intelligence and analytics",
                "_desc": "dashboards, reporting, and data visualization",
                "_kws": "BI,Tableau,Power BI,Looker,dashboard,KPI,data visualization,SQL,reporting",
                "Data Visualization": {"_subject": "data visualization", "_desc": "charts, dashboards, and visual storytelling with data", "_kws": "data visualization,Tableau,Plotly,D3.js,Power BI,Looker,chart,heatmap,geospatial,storytelling"},
                "Product Analytics": {"_subject": "product analytics", "_desc": "user behavior, funnels, and product metrics", "_kws": "product analytics,funnel,retention,DAU,MAU,cohort,Mixpanel,Amplitude,event tracking,user journey"},
            },
        },
        "⚖️ Law": {
            "_subject": "law",
            "_desc": "legal principles, case law, and legal practice",
            "_kws": "law,legal,court,jurisdiction,statute,precedent,litigation,counsel,rights",
            "Family Law": {
                "_subject": "family law",
                "_desc": "family law, domestic relations, and child welfare",
                "_kws": "family law,divorce,custody,child support,adoption,domestic violence,guardianship,marital property",
                "Divorce": {"_subject": "divorce law", "_desc": "divorce proceedings, grounds, and financial settlement", "_kws": "divorce,dissolution,grounds for divorce,financial settlement,ancillary relief,marital assets,separation agreement,decree nisi"},
                "Child Custody & Support": {"_subject": "child custody and support", "_desc": "custody arrangements, parenting plans, and child support calculations", "_kws": "child custody,parenting plan,visitation,child support,best interests,joint custody,sole custody,CAFCASS"},
                "Adoption & Surrogacy": {"_subject": "adoption and surrogacy law", "_desc": "adoption procedures, surrogacy agreements, and parental rights", "_kws": "adoption,surrogacy,parental order,home study,placement,foster,birth parent,international adoption"},
            },
            "Criminal Law": {
                "_subject": "criminal law",
                "_desc": "criminal offences, procedure, and defence",
                "_kws": "criminal law,offence,prosecution,defence,sentence,bail,crown court,magistrate,beyond reasonable doubt",
                "Criminal Defence": {"_subject": "criminal defence", "_desc": "defence strategies, rights of the accused, and criminal procedure", "_kws": "criminal defence,plea,acquittal,alibi,mens rea,actus reus,jury,cross-examination,reasonable doubt"},
                "White Collar Crime": {"_subject": "white collar crime", "_desc": "fraud, money laundering, and corporate crime", "_kws": "white collar crime,fraud,embezzlement,money laundering,insider trading,bribery,corporate crime,SFO,FCA"},
                "Sentencing & Parole": {"_subject": "sentencing and parole", "_desc": "sentencing guidelines, parole boards, and rehabilitation", "_kws": "sentencing,guidelines,parole,probation,rehabilitation,custodial,community order,tariff,Parole Board"},
            },
            "Employment & Labour Law": {
                "_subject": "employment and labour law",
                "_desc": "employment rights, workplace disputes, and labour relations",
                "_kws": "employment law,unfair dismissal,redundancy,discrimination,tribunal,TUPE,trade union,wages,contract",
                "Unfair Dismissal & Redundancy": {"_subject": "unfair dismissal and redundancy", "_desc": "dismissal rights, redundancy pay, and tribunal claims", "_kws": "unfair dismissal,redundancy,employment tribunal,settlement agreement,compromise,notice period,constructive dismissal"},
                "Workplace Discrimination": {"_subject": "workplace discrimination law", "_desc": "protected characteristics, harassment, and equality law", "_kws": "discrimination,Equality Act,protected characteristic,harassment,victimisation,reasonable adjustments,equal pay,ACAS"},
                "Trade Union & Collective": {"_subject": "trade union and collective labour law", "_desc": "unions, collective bargaining, and industrial action", "_kws": "trade union,collective bargaining,strike,industrial action,recognition,shop steward,TULRCA,picketing"},
            },
            "Corporate & Commercial Law": {
                "_subject": "corporate and commercial law",
                "_desc": "company law, contracts, and commercial transactions",
                "_kws": "corporate law,company,director,shareholder,contract,M&A,intellectual property,due diligence,articles",
                "Mergers & Acquisitions": {"_subject": "mergers and acquisitions law", "_desc": "M&A transactions, due diligence, and deal structures", "_kws": "M&A,merger,acquisition,due diligence,SPA,warranty,indemnity,completion,regulatory approval"},
                "Intellectual Property": {"_subject": "intellectual property law", "_desc": "patents, trademarks, copyright, and IP enforcement", "_kws": "intellectual property,patent,trademark,copyright,trade secret,licensing,infringement,WIPO,design right"},
                "Contract Law": {"_subject": "contract law", "_desc": "contract formation, breach, and remedies", "_kws": "contract,offer,acceptance,consideration,breach,damages,specific performance,frustration,misrepresentation"},
            },
            "Immigration Law": {
                "_subject": "immigration law",
                "_desc": "visas, asylum, citizenship, and immigration procedure",
                "_kws": "immigration,visa,asylum,refugee,deportation,citizenship,naturalisation,leave to remain,Home Office",
                "Asylum & Refugee Law": {"_subject": "asylum and refugee law", "_desc": "asylum claims, refugee status, and international protection", "_kws": "asylum,refugee,persecution,Convention refugee,credibility,country guidance,UNHCR,safe third country,humanitarian protection"},
                "Work & Skilled Visas": {"_subject": "work visas and skilled migration", "_desc": "skilled worker visas, sponsorship, and points-based systems", "_kws": "skilled worker,Tier 2,sponsorship,points-based,SOC code,salary threshold,certificate of sponsorship,global talent visa"},
                "Citizenship & Naturalisation": {"_subject": "citizenship and naturalisation", "_desc": "acquiring citizenship, naturalisation requirements, and nationality law", "_kws": "citizenship,naturalisation,British National,right of abode,stateless,dual nationality,good character,residency requirement"},
            },
        },
        "🌍 Climate Change": {
            "_subject": "climate change",
            "_desc": "climate science, global warming, and environmental policy",
            "_kws": "climate change,global warming,CO2,greenhouse gas,IPCC,carbon,renewable energy",
            "Climate Science": {
                "_subject": "climate science",
                "_desc": "atmospheric science, climate models, and climate data",
                "_kws": "climate science,atmosphere,CO2,radiative forcing,tipping points,IPCC,paleoclimate,climate model,feedback",
                "Atmospheric Science": {"_subject": "atmospheric science", "_desc": "atmospheric composition, circulation, and climate feedbacks", "_kws": "atmosphere,greenhouse effect,ozone,aerosol,jet stream,ENSO,troposphere,stratosphere,radiative forcing"},
                "Ocean & Ice Science": {"_subject": "ocean and cryosphere science", "_desc": "ocean heat, sea level rise, and ice sheet dynamics", "_kws": "ocean warming,sea level rise,ice sheet,glacier,thermohaline circulation,Arctic,Antarctic,albedo,permafrost"},
                "Extreme Weather": {"_subject": "extreme weather and climate", "_desc": "heatwaves, floods, hurricanes, and climate attribution", "_kws": "extreme weather,heatwave,flood,hurricane,drought,wildfire,attribution science,compound event,weather pattern"},
            },
            "Energy Transition": {
                "_subject": "energy transition",
                "_desc": "renewable energy, decarbonization, and clean technology",
                "_kws": "renewable energy,solar,wind,energy storage,grid,hydrogen,decarbonization,net zero,clean tech",
                "Solar Energy": {"_subject": "solar energy", "_desc": "photovoltaics, solar farms, and solar technology", "_kws": "solar,photovoltaic,PV panel,solar farm,inverter,LCOE,rooftop solar,concentrated solar,perovskite"},
                "Wind Energy": {"_subject": "wind energy", "_desc": "onshore and offshore wind power", "_kws": "wind energy,wind turbine,offshore wind,onshore wind,capacity factor,wake effect,wind farm,LCOE,blade"},
                "Energy Storage": {"_subject": "energy storage", "_desc": "batteries, grid storage, and long-duration solutions", "_kws": "energy storage,battery,lithium-ion,grid storage,pumped hydro,flow battery,hydrogen,duration,cycle life"},
            },
            "Climate Policy": {
                "_subject": "climate policy and economics",
                "_desc": "carbon markets, international agreements, and climate finance",
                "_kws": "Paris Agreement,carbon price,emissions trading,ESG,climate finance,net zero,NDC,carbon offset",
                "Carbon Markets": {"_subject": "carbon markets", "_desc": "emissions trading, carbon credits, and offset standards", "_kws": "carbon market,ETS,carbon credit,offset,voluntary market,compliance,additionality,permanence,CORSIA"},
                "International Agreements": {"_subject": "international climate agreements", "_desc": "Paris Agreement, COP, and global climate commitments", "_kws": "Paris Agreement,COP,UNFCCC,NDC,net zero,1.5 degrees,loss and damage,climate finance,adaptation fund"},
            },
        },
        "💰 Personal Finance": {
            "_subject": "personal finance",
            "_desc": "personal finance, investing, budgeting, and wealth management",
            "_kws": "investing,stocks,bonds,ETF,budgeting,savings,retirement,401k,compound interest",
            "Investing": {
                "_subject": "investing",
                "_desc": "stock market, bonds, and investment strategies",
                "_kws": "investing,stocks,bonds,ETF,index fund,portfolio,diversification,asset allocation,returns",
                "Stock Market": {"_subject": "stock market investing", "_desc": "equities, valuation, and stock analysis", "_kws": "stocks,equities,P/E ratio,valuation,dividend,earnings,growth stock,value investing,market cap"},
                "Real Estate Investing": {"_subject": "real estate investing", "_desc": "property investment, REITs, and rental income", "_kws": "real estate,rental property,REIT,buy-to-let,cap rate,cash flow,appreciation,leverage,mortgage"},
                "Index Funds & ETFs": {"_subject": "index funds and ETFs", "_desc": "passive investing, ETF selection, and cost efficiency", "_kws": "index fund,ETF,S&P 500,passive investing,expense ratio,Vanguard,Fidelity,total market,factor investing"},
            },
            "Budgeting & Debt": {
                "_subject": "budgeting and debt management",
                "_desc": "budgeting strategies, debt payoff, and saving",
                "_kws": "budgeting,debt,savings,emergency fund,50/30/20,debt snowball,debt avalanche,frugality,FIRE",
                "Debt Payoff": {"_subject": "debt payoff strategies", "_desc": "debt snowball, avalanche, and debt consolidation", "_kws": "debt payoff,debt snowball,debt avalanche,consolidation,interest rate,minimum payment,credit card debt"},
                "FIRE Movement": {"_subject": "FIRE financial independence", "_desc": "financial independence, early retirement, and safe withdrawal rates", "_kws": "FIRE,financial independence,early retirement,4% rule,safe withdrawal rate,leanFIRE,fatFIRE,coast FIRE"},
            },
            "Retirement Planning": {
                "_subject": "retirement planning",
                "_desc": "retirement accounts, pension planning, and withdrawal strategies",
                "_kws": "retirement,401k,IRA,pension,Roth,Social Security,safe withdrawal,annuity,required minimum distribution",
                "401k & IRA": {"_subject": "401k and IRA retirement accounts", "_desc": "US retirement accounts, contribution limits, and tax advantages", "_kws": "401k,IRA,Roth IRA,traditional IRA,contribution limit,employer match,rollover,backdoor Roth"},
                "Pension & UK Retirement": {"_subject": "pension and UK retirement", "_desc": "UK pensions, State Pension, and retirement planning", "_kws": "pension,State Pension,SIPP,ISA,lifetime ISA,auto-enrolment,defined benefit,annuity,drawdown"},
            },
        },
        "⚕️ Nutrition & Health": {
            "_subject": "nutrition and health",
            "_desc": "nutrition science, healthy eating, and preventive health",
            "_kws": "nutrition,diet,vitamins,protein,carbohydrates,metabolism,gut health,supplements",
            "Diets & Eating Patterns": {
                "_subject": "diets and eating patterns",
                "_desc": "dietary approaches, meal planning, and nutritional evidence",
                "_kws": "diet,keto,Mediterranean,vegan,intermittent fasting,paleo,carnivore,whole foods,caloric deficit",
                "Ketogenic Diet": {"_subject": "ketogenic diet", "_desc": "ketosis, macros, and low-carb nutrition science", "_kws": "keto,ketosis,low carb,fat adaptation,BHB,MCT,net carbs,ketoacidosis,metabolic flexibility"},
                "Mediterranean Diet": {"_subject": "Mediterranean diet", "_desc": "Mediterranean eating, olive oil, and longevity research", "_kws": "Mediterranean diet,olive oil,fish,legumes,whole grains,antioxidants,longevity,PREDIMED,cardiovascular"},
                "Intermittent Fasting": {"_subject": "intermittent fasting", "_desc": "IF protocols, autophagy, and metabolic effects", "_kws": "intermittent fasting,16:8,5:2,OMAD,autophagy,insulin,time-restricted eating,circadian,metabolic health"},
            },
            "Sports Nutrition": {
                "_subject": "sports nutrition",
                "_desc": "performance nutrition, supplements, and recovery",
                "_kws": "sports nutrition,protein,creatine,pre-workout,recovery,hydration,carb loading,electrolytes,amino acids",
                "Protein & Muscle": {"_subject": "protein and muscle building nutrition", "_desc": "protein requirements, leucine threshold, and hypertrophy nutrition", "_kws": "protein,muscle protein synthesis,leucine,whey,casein,hypertrophy,anabolic window,BCAAs,protein timing"},
                "Endurance Nutrition": {"_subject": "endurance sports nutrition", "_desc": "carbohydrate strategies, fueling, and long-distance nutrition", "_kws": "endurance nutrition,carb loading,gels,electrolytes,bonking,glycogen,VO2 max,race day fueling,hydration"},
            },
            "Gut Health & Microbiome": {
                "_subject": "gut health and microbiome",
                "_desc": "gut microbiome, probiotics, and digestive health",
                "_kws": "gut health,microbiome,probiotic,prebiotic,IBS,dysbiosis,leaky gut,fermented food,16S rRNA",
                "Probiotics & Prebiotics": {"_subject": "probiotics and prebiotics", "_desc": "probiotic strains, prebiotic fibres, and clinical evidence", "_kws": "probiotics,prebiotics,Lactobacillus,Bifidobacterium,inulin,FOS,gut flora,strain,CFU"},
                "IBS & Digestive Disorders": {"_subject": "IBS and digestive disorders", "_desc": "IBS, IBD, SIBO, and digestive symptom management", "_kws": "IBS,Crohn's,ulcerative colitis,SIBO,low FODMAP,bloating,leaky gut,gut-brain axis,gastroenterology"},
            },
        },
        "🏋️ Fitness & Exercise": {
            "_subject": "fitness and exercise",
            "_desc": "physical fitness, exercise science, and sports performance",
            "_kws": "exercise,strength training,cardio,HIIT,muscle,recovery,VO2 max,flexibility,sports performance",
            "Strength Training": {
                "_subject": "strength training",
                "_desc": "resistance training, progressive overload, and hypertrophy",
                "_kws": "strength training,progressive overload,powerlifting,hypertrophy,1RM,compound lift,periodisation",
                "Powerlifting": {"_subject": "powerlifting", "_desc": "squat, bench, deadlift, and competitive powerlifting", "_kws": "powerlifting,squat,bench press,deadlift,1RM,IPF,competition,peaking,conjugate,RPE"},
                "Bodybuilding": {"_subject": "bodybuilding", "_desc": "muscle hypertrophy, aesthetics, and bodybuilding programming", "_kws": "bodybuilding,hypertrophy,volume,isolation,muscle growth,contest prep,cutting,bulking,pump"},
                "Calisthenics": {"_subject": "calisthenics", "_desc": "bodyweight training, skill progressions, and street workout", "_kws": "calisthenics,bodyweight,pull-up,dip,handstand,planche,front lever,muscle-up,skill progression"},
            },
            "Cardio & Endurance": {
                "_subject": "cardio and endurance training",
                "_desc": "cardiovascular training, running, and endurance sports",
                "_kws": "cardio,running,cycling,VO2 max,aerobic,lactate threshold,zone 2,endurance,heart rate",
                "Running": {"_subject": "running training", "_desc": "running plans, form, and race preparation", "_kws": "running,marathon,5k,10k,training plan,tempo run,long run,pace,stride,injury prevention"},
                "Cycling": {"_subject": "cycling training", "_desc": "cycling fitness, FTP, and performance on the bike", "_kws": "cycling,FTP,power meter,VO2 max,Zwift,cadence,interval,gran fondo,peloton,time trial"},
                "HIIT": {"_subject": "HIIT training", "_desc": "high-intensity interval training, Tabata, and metabolic conditioning", "_kws": "HIIT,interval training,Tabata,Fartlek,sprint,work-to-rest,afterburn,EPOC,metabolic conditioning"},
            },
            "Recovery & Flexibility": {
                "_subject": "recovery and flexibility",
                "_desc": "sleep, mobility, stretching, and injury prevention",
                "_kws": "recovery,sleep,mobility,stretching,foam rolling,yoga,injury prevention,HRV,active recovery",
                "Mobility & Stretching": {"_subject": "mobility and stretching", "_desc": "joint mobility, flexibility, and movement quality", "_kws": "mobility,flexibility,stretching,ROM,PNF,dynamic warm-up,hip flexor,thoracic,movement quality"},
                "Sleep & Recovery Science": {"_subject": "sleep and exercise recovery", "_desc": "sleep quality, HRV, and recovery optimisation", "_kws": "sleep,HRV,heart rate variability,deep sleep,REM,recovery,cortisol,melatonin,overtraining"},
            },
        },
        "🧘 Mental Health": {
            "_subject": "mental health",
            "_desc": "mental health, psychology, and well-being",
            "_kws": "mental health,anxiety,depression,therapy,CBT,mindfulness,stress,psychiatry,resilience",
            "Therapy Approaches": {
                "_subject": "psychotherapy approaches",
                "_desc": "evidence-based therapies, modalities, and treatment frameworks",
                "_kws": "CBT,DBT,ACT,psychoanalysis,therapy,psychotherapy,EMDR,schema therapy,humanistic",
                "CBT": {"_subject": "cognitive behavioural therapy", "_desc": "CBT techniques, thought records, and behavioural activation", "_kws": "CBT,cognitive distortions,thought record,behavioural activation,exposure,formulation,automatic thoughts,schema"},
                "DBT": {"_subject": "dialectical behaviour therapy", "_desc": "DBT skills, distress tolerance, and emotional regulation", "_kws": "DBT,dialectical,distress tolerance,emotional regulation,mindfulness,interpersonal effectiveness,TIPP,DEAR MAN"},
                "ACT": {"_subject": "acceptance and commitment therapy", "_desc": "ACT, psychological flexibility, and values-based living", "_kws": "ACT,acceptance,defusion,values,committed action,psychological flexibility,mindfulness,RFT,hexaflex"},
            },
            "Mental Health Conditions": {
                "_subject": "mental health conditions",
                "_desc": "psychiatric diagnoses, symptoms, and evidence-based treatment",
                "_kws": "anxiety,depression,PTSD,ADHD,bipolar,OCD,schizophrenia,eating disorder,DSM,diagnosis",
                "Anxiety Disorders": {"_subject": "anxiety disorders", "_desc": "generalised anxiety, panic disorder, phobias, and treatment", "_kws": "anxiety,panic disorder,GAD,social anxiety,phobia,agoraphobia,exposure therapy,worry,autonomic arousal"},
                "Depression": {"_subject": "depression", "_desc": "major depression, dysthymia, causes, and treatment", "_kws": "depression,MDD,antidepressant,SSRI,anhedonia,low mood,cognitive,biological,psychosocial,suicidality"},
                "ADHD": {"_subject": "ADHD", "_desc": "attention deficit hyperactivity disorder, diagnosis, and management", "_kws": "ADHD,attention,hyperactivity,impulsivity,executive function,methylphenidate,Ritalin,Adderall,coaching,diagnosis"},
            },
            "Mindfulness & Stress": {
                "_subject": "mindfulness and stress management",
                "_desc": "meditation, stress reduction, and mindfulness-based interventions",
                "_kws": "mindfulness,meditation,stress,MBSR,breathing,body scan,resilience,burnout,relaxation response",
                "Meditation": {"_subject": "meditation practices", "_desc": "meditation techniques, traditions, and neuroscience", "_kws": "meditation,mindfulness,Vipassana,loving kindness,transcendental,breath awareness,neuroscience,default mode network"},
                "Burnout": {"_subject": "burnout and occupational stress", "_desc": "workplace burnout, recovery, and prevention", "_kws": "burnout,occupational stress,exhaustion,cynicism,efficacy,Maslach,recovery,boundaries,work-life balance"},
            },
        },
        "🏥 Medicine & Healthcare": {
            "_subject": "medicine and healthcare",
            "_desc": "medicine, clinical practice, and healthcare systems",
            "_kws": "medicine,diagnosis,treatment,clinical trials,pharmacology,surgery,epidemiology,NHS,WHO",
            "Internal Medicine": {
                "_subject": "internal medicine",
                "_desc": "internal medicine, chronic disease, and clinical management",
                "_kws": "internal medicine,cardiology,endocrinology,pulmonology,nephrology,gastroenterology,diagnosis,management",
                "Cardiology": {"_subject": "cardiology", "_desc": "heart disease, ECG, and cardiovascular management", "_kws": "cardiology,heart failure,arrhythmia,ECG,coronary artery disease,statin,hypertension,atrial fibrillation,percutaneous"},
                "Endocrinology": {"_subject": "endocrinology", "_desc": "diabetes, thyroid, hormones, and metabolic disorders", "_kws": "endocrinology,diabetes,thyroid,insulin,HbA1c,hypothyroidism,adrenal,pituitary,hormones,metabolic syndrome"},
                "Pulmonology": {"_subject": "pulmonology", "_desc": "lung disease, COPD, asthma, and respiratory medicine", "_kws": "pulmonology,COPD,asthma,spirometry,pulmonary fibrosis,lung cancer,ventilation,bronchodilator,oxygen therapy"},
            },
            "Pharmacology": {
                "_subject": "pharmacology",
                "_desc": "drug mechanisms, pharmacokinetics, and clinical pharmacology",
                "_kws": "pharmacology,drug,mechanism,receptor,pharmacokinetics,half-life,side effect,interaction,dose-response",
                "Antibiotics & Antimicrobials": {"_subject": "antibiotics and antimicrobial resistance", "_desc": "antibiotic classes, resistance mechanisms, and stewardship", "_kws": "antibiotic,resistance,AMR,beta-lactam,penicillin,MRSA,stewardship,minimum inhibitory concentration,bactericidal"},
                "Drug Development": {"_subject": "drug development and clinical trials", "_desc": "drug discovery, phase trials, and regulatory approval", "_kws": "drug development,clinical trial,Phase I,Phase II,Phase III,FDA,EMA,IND,efficacy,safety,randomised controlled trial"},
            },
            "Public Health": {
                "_subject": "public health",
                "_desc": "epidemiology, vaccination, and global health",
                "_kws": "public health,epidemiology,vaccine,pandemic,surveillance,incidence,prevalence,social determinants,WHO",
                "Epidemiology": {"_subject": "epidemiology", "_desc": "disease distribution, risk factors, and epidemiological methods", "_kws": "epidemiology,incidence,prevalence,cohort study,case-control,RCT,bias,confounding,Mendelian randomisation"},
                "Vaccines & Immunology": {"_subject": "vaccines and immunology", "_desc": "vaccine types, immune response, and immunisation programmes", "_kws": "vaccine,immunisation,mRNA vaccine,adjuvant,herd immunity,T cell,B cell,antibody,antigen,immunogenicity"},
                "Global Health": {"_subject": "global health", "_desc": "health inequalities, tropical diseases, and global health systems", "_kws": "global health,low-income,malaria,tuberculosis,HIV,neglected tropical disease,LMIC,health equity,SDGs"},
            },
        },
        "🚀 Space Exploration": {
            "_subject": "space exploration",
            "_desc": "space science, astronomy, and space missions",
            "_kws": "NASA,SpaceX,Mars,moon,rocket,satellite,asteroid,telescope,cosmology",
            "Solar System": {
                "_subject": "solar system exploration",
                "_desc": "planets, moons, and missions within our solar system",
                "_kws": "solar system,Mars,moon,Jupiter,asteroid,comet,planetary science,lander,rover",
                "Mars Exploration": {"_subject": "Mars exploration", "_desc": "Mars missions, geology, and future human settlement", "_kws": "Mars,rover,Perseverance,Curiosity,terraforming,Mars sample return,Olympus Mons,habitability,methane"},
                "Moon & Lunar Missions": {"_subject": "lunar exploration", "_desc": "Moon missions, Artemis, and lunar resources", "_kws": "Moon,Artemis,lunar,Moonbase,Gateway,Helium-3,regolith,crater,Apollo,lunar south pole"},
                "Asteroid & Comet Science": {"_subject": "asteroid and comet science", "_desc": "near-Earth objects, asteroid mining, and planetary defence", "_kws": "asteroid,comet,near-Earth object,planetary defence,DART,Hayabusa,mining,carbonaceous chondrite,Kuiper Belt"},
            },
            "Deep Space & Astrophysics": {
                "_subject": "astrophysics and deep space",
                "_desc": "stars, galaxies, dark matter, and cosmology",
                "_kws": "astrophysics,galaxy,dark matter,dark energy,black hole,neutron star,cosmology,James Webb,Hubble",
                "Black Holes": {"_subject": "black holes", "_desc": "black hole physics, event horizons, and observations", "_kws": "black hole,event horizon,Hawking radiation,singularity,accretion disk,gravitational wave,M87,Sagittarius A*"},
                "Exoplanets": {"_subject": "exoplanets", "_desc": "exoplanet detection, atmospheres, and habitability", "_kws": "exoplanet,transit,radial velocity,habitable zone,JWST,TESS,Kepler,biosignature,super Earth,hot Jupiter"},
            },
            "Space Technology": {
                "_subject": "space technology",
                "_desc": "rockets, spacecraft, and space infrastructure",
                "_kws": "rocket,spacecraft,propulsion,satellite,ISS,SpaceX,Starship,launch vehicle,reusability,orbital mechanics",
                "Rocket Propulsion": {"_subject": "rocket propulsion", "_desc": "rocket engines, propellants, and propulsion systems", "_kws": "rocket engine,Merlin,Raptor,ion thruster,specific impulse,thrust,propellant,LOX,methane,nuclear propulsion"},
                "Satellites & Constellations": {"_subject": "satellites and mega-constellations", "_desc": "satellite technology, Starlink, and orbital operations", "_kws": "satellite,Starlink,OneWeb,LEO,GEO,communication satellite,earth observation,CubeSat,debris,Kessler syndrome"},
            },
        },
        "🧬 Genetics & Genomics": {
            "_subject": "genetics and genomics",
            "_desc": "genetics, genomics, and molecular biology",
            "_kws": "DNA,gene,genome,mutation,CRISPR,heredity,chromosome,protein,RNA",
            "Molecular Biology": {
                "_subject": "molecular biology",
                "_desc": "DNA, RNA, proteins, and cellular mechanisms",
                "_kws": "DNA,RNA,protein synthesis,transcription,translation,replication,PCR,gel electrophoresis,restriction enzyme",
                "Gene Expression": {"_subject": "gene expression and regulation", "_desc": "transcription factors, epigenetics, and gene regulation", "_kws": "gene expression,transcription factor,promoter,enhancer,epigenetics,methylation,histone,chromatin,RNA polymerase"},
                "Protein Structure": {"_subject": "protein structure and function", "_desc": "protein folding, domains, and structural biology", "_kws": "protein structure,AlphaFold,folding,domain,active site,post-translational modification,cryo-EM,X-ray crystallography"},
            },
            "Genomics & Sequencing": {
                "_subject": "genomics and DNA sequencing",
                "_desc": "genome sequencing, bioinformatics, and genetic variation",
                "_kws": "genomics,sequencing,NGS,SNP,GWAS,bioinformatics,variant calling,reference genome,annotation",
                "Bioinformatics": {"_subject": "bioinformatics", "_desc": "computational genomics, pipelines, and sequence analysis", "_kws": "bioinformatics,BLAST,alignment,Python,R,genome assembly,pipeline,variant calling,RNA-seq,FASTQ"},
                "Cancer Genomics": {"_subject": "cancer genomics", "_desc": "somatic mutations, oncogenes, and precision oncology", "_kws": "cancer genomics,somatic mutation,oncogene,tumour suppressor,copy number,liquid biopsy,precision oncology,TMB,MSI"},
            },
            "Genetic Engineering": {
                "_subject": "genetic engineering",
                "_desc": "CRISPR, gene therapy, and synthetic biology",
                "_kws": "CRISPR,Cas9,gene therapy,base editing,prime editing,synthetic biology,AAV,plasmid,off-target",
                "CRISPR": {"_subject": "CRISPR gene editing", "_desc": "CRISPR-Cas9 mechanisms, delivery, and applications", "_kws": "CRISPR,Cas9,guide RNA,HDR,NHEJ,base editor,prime editor,off-target,delivery,in vivo"},
                "Gene Therapy": {"_subject": "gene therapy", "_desc": "viral vectors, gene delivery, and clinical gene therapy", "_kws": "gene therapy,AAV,lentivirus,ex vivo,in vivo,vector,clinical trial,Luxturna,Zolgensma,haemophilia"},
            },
        },
        "🏛️ History": {
            "_subject": "history",
            "_desc": "world history, civilizations, and historical events",
            "_kws": "history,civilization,war,empire,revolution,archaeology,ancient,medieval,modern history",
            "Ancient History": {
                "_subject": "ancient history",
                "_desc": "ancient civilizations, empires, and archaeology",
                "_kws": "ancient,Greece,Rome,Egypt,Mesopotamia,Persia,archaeology,classical,Bronze Age,Iron Age",
                "Ancient Greece": {"_subject": "ancient Greece", "_desc": "Greek city-states, philosophy, democracy, and culture", "_kws": "Greece,Athens,Sparta,democracy,philosophy,Socrates,Plato,Aristotle,Peloponnesian War,Hellenistic"},
                "Ancient Rome": {"_subject": "ancient Rome", "_desc": "Roman Republic, Empire, law, and legacy", "_kws": "Rome,Republic,Empire,Caesar,Augustus,Senate,legion,Roman law,aqueduct,Constantine,fall of Rome"},
                "Ancient Egypt": {"_subject": "ancient Egypt", "_desc": "pharaohs, pyramids, and Egyptian civilization", "_kws": "Egypt,pharaoh,pyramid,hieroglyphics,mummy,Nile,Cleopatra,Tutankhamun,New Kingdom,Rosetta Stone"},
            },
            "Medieval History": {
                "_subject": "medieval history",
                "_desc": "medieval Europe, Islam, and the feudal world",
                "_kws": "medieval,feudalism,crusades,Black Death,Byzantine,Islamic Golden Age,Viking,Norman,Magna Carta",
                "Crusades": {"_subject": "the Crusades", "_desc": "crusader states, religious war, and medieval geopolitics", "_kws": "Crusades,Jerusalem,Saladin,Richard I,crusader state,papal authority,Islamic response,military orders,reconquista"},
                "Islamic Golden Age": {"_subject": "Islamic Golden Age", "_desc": "medieval Islamic science, culture, and civilisation", "_kws": "Islamic Golden Age,Baghdad,House of Wisdom,Ibn Sina,Al-Khwarizmi,algebra,astronomy,medicine,translation movement"},
            },
            "Modern History": {
                "_subject": "modern history",
                "_desc": "20th and 21st century history, wars, and revolutions",
                "_kws": "WWI,WWII,Cold War,decolonisation,Holocaust,revolution,nuclear,United Nations,globalisation",
                "World War II": {"_subject": "World War II", "_desc": "WWII causes, major battles, and aftermath", "_kws": "WWII,Hitler,Churchill,D-Day,Holocaust,Pacific War,atomic bomb,Normandy,Eastern Front,Nazi Germany"},
                "Cold War": {"_subject": "the Cold War", "_desc": "US-Soviet rivalry, proxy wars, and arms race", "_kws": "Cold War,USSR,USA,nuclear arms race,Cuban Missile Crisis,Berlin Wall,NATO,Warsaw Pact,proxy war,detente"},
                "Decolonisation": {"_subject": "decolonisation", "_desc": "independence movements, post-colonial states, and legacies", "_kws": "decolonisation,independence,Africa,India,partition,anti-colonialism,non-alignment,Bandung,post-colonial"},
            },
        },
        "🎮 Game Development": {
            "_subject": "game development",
            "_desc": "video game development, game design, and game engines",
            "_kws": "Unity,Unreal Engine,game design,game mechanics,shader,physics engine,procedural generation,indie game",
            "Game Engines": {
                "_subject": "game engines",
                "_desc": "Unity, Unreal Engine, Godot, and engine architecture",
                "_kws": "Unity,Unreal Engine,Godot,game engine,rendering,physics,scripting,ECS,prefab,blueprint",
                "Unity": {"_subject": "Unity game development", "_desc": "Unity engine, C# scripting, and Unity best practices", "_kws": "Unity,C#,MonoBehaviour,physics,animator,prefab,ScriptableObject,URP,HDRP,Unity ECS"},
                "Unreal Engine": {"_subject": "Unreal Engine development", "_desc": "Unreal Engine, Blueprint, and C++ game development", "_kws": "Unreal Engine,UE5,Blueprint,C++,Nanite,Lumen,actor,component,material,animation tree"},
                "Godot": {"_subject": "Godot game development", "_desc": "Godot engine, GDScript, and open-source game dev", "_kws": "Godot,GDScript,node,scene,signal,open source,2D,3D,shader,export"},
            },
            "Game Design": {
                "_subject": "game design",
                "_desc": "game mechanics, level design, and player experience",
                "_kws": "game design,mechanics,level design,player experience,game feel,balance,progression,narrative,UX",
                "Level Design": {"_subject": "level design", "_desc": "level layout, player flow, and environmental storytelling", "_kws": "level design,player flow,waypointing,environmental storytelling,encounter design,pacing,blockout,grayboxing"},
                "Game Narrative": {"_subject": "game narrative design", "_desc": "interactive storytelling, branching dialogue, and ludonarrative", "_kws": "narrative design,branching,dialogue,ludonarrative,environmental storytelling,world building,Twine,choice consequence"},
            },
            "Graphics Programming": {
                "_subject": "graphics programming",
                "_desc": "shaders, rendering pipelines, and 3D mathematics",
                "_kws": "shader,HLSL,GLSL,rendering,rasterization,ray tracing,GPU,PBR,normal map,lighting",
                "Shaders & HLSL": {"_subject": "shader programming", "_desc": "HLSL, GLSL, shader graphs, and visual effects", "_kws": "HLSL,GLSL,vertex shader,fragment shader,compute shader,shader graph,VFX,post-processing,bloom"},
                "Ray Tracing": {"_subject": "real-time ray tracing", "_desc": "ray tracing, path tracing, and global illumination", "_kws": "ray tracing,path tracing,global illumination,DXR,Vulkan RT,denoising,DLSS,reflection,ambient occlusion"},
            },
        },
        "🍳 Cooking & Culinary Arts": {
            "_subject": "cooking and culinary arts",
            "_desc": "cooking techniques, recipes, and culinary science",
            "_kws": "cooking,recipe,baking,fermentation,knife skills,Maillard reaction,pastry,cuisine,flavor pairing",
            "Cooking Techniques": {
                "_subject": "cooking techniques",
                "_desc": "fundamental cooking methods, heat transfer, and flavour development",
                "_kws": "sauté,braise,roast,sear,steam,poach,Maillard reaction,caramelisation,umami,knife skills",
                "French Techniques": {"_subject": "French culinary techniques", "_desc": "classical French cooking, sauces, and knife skills", "_kws": "French cuisine,sauce,roux,beurre blanc,mise en place,julienne,brunoise,fond,reduction,meunière"},
                "Fermentation": {"_subject": "food fermentation", "_desc": "fermented foods, microbiology, and preservation", "_kws": "fermentation,lacto-fermentation,kimchi,sourdough,kefir,kombucha,brine,lactic acid bacteria,wild yeast"},
                "Molecular Gastronomy": {"_subject": "molecular gastronomy", "_desc": "food science, spherification, and modernist cooking", "_kws": "molecular gastronomy,spherification,sous vide,agar,xanthan,emulsification,flavour pairing,Ferran Adrià,Heston Blumenthal"},
            },
            "Baking & Pastry": {
                "_subject": "baking and pastry",
                "_desc": "bread, pastry, and confectionery science",
                "_kws": "baking,bread,pastry,gluten,yeast,proofing,croissant,soufflé,tempering chocolate,lamination",
                "Bread Baking": {"_subject": "bread baking", "_desc": "sourdough, yeasted breads, and gluten development", "_kws": "sourdough,levain,autolyse,bulk fermentation,shaping,scoring,crust,crumb,hydration,baguette"},
                "Chocolate & Confectionery": {"_subject": "chocolate and confectionery", "_desc": "chocolate tempering, pralines, and sugar work", "_kws": "chocolate,tempering,ganache,praline,sugar work,bonbon,cocoa butter,crystallisation,caramel,nougat"},
            },
            "World Cuisines": {
                "_subject": "world cuisines",
                "_desc": "international cooking traditions, ingredients, and flavour profiles",
                "_kws": "cuisine,Asian,Mediterranean,Latin American,Middle Eastern,Indian,Japanese,spice,regional cooking",
                "Japanese Cuisine": {"_subject": "Japanese cuisine", "_desc": "Japanese cooking techniques, ingredients, and culture", "_kws": "Japanese,sushi,ramen,dashi,miso,umami,kaiseki,tempura,katsu,izakaya,washoku"},
                "Indian Cuisine": {"_subject": "Indian cuisine", "_desc": "Indian spices, regional dishes, and cooking methods", "_kws": "Indian cuisine,curry,masala,tandoor,dal,biryani,spice blend,ghee,chutney,regional Indian cooking"},
            },
        },
        "🚗 Electric Vehicles & Automotive": {
            "_subject": "electric vehicles and automotive",
            "_desc": "electric vehicles, automotive technology, and transportation",
            "_kws": "electric vehicle,EV,battery,Tesla,charging,range anxiety,autonomous driving,lithium-ion,torque",
            "EV Technology": {
                "_subject": "electric vehicle technology",
                "_desc": "EV powertrains, batteries, and charging systems",
                "_kws": "EV,battery,BMS,motor,inverter,charging,range,regenerative braking,thermal management,pack",
                "Battery Systems": {"_subject": "EV battery systems", "_desc": "lithium-ion cells, BMS, and battery chemistry", "_kws": "lithium-ion,NMC,LFP,solid-state battery,BMS,cell,pack,energy density,degradation,thermal runaway"},
                "Charging Infrastructure": {"_subject": "EV charging infrastructure", "_desc": "charging standards, fast charging, and grid integration", "_kws": "charging,fast charging,DC fast charger,CCS,CHAdeMO,V2G,Supercharger,OCPP,kWh,charging network"},
            },
            "Autonomous Driving": {
                "_subject": "autonomous driving",
                "_desc": "self-driving technology, sensors, and AI for vehicles",
                "_kws": "autonomous driving,LIDAR,radar,camera,sensor fusion,path planning,SLAM,Waymo,Tesla FSD,SAE levels",
                "Sensor Systems": {"_subject": "autonomous vehicle sensors", "_desc": "LiDAR, radar, camera, and sensor fusion for AVs", "_kws": "LiDAR,radar,camera,ultrasonic,sensor fusion,point cloud,object detection,perception,depth estimation"},
                "AI & Path Planning": {"_subject": "autonomous driving AI", "_desc": "path planning, decision-making, and simulation for self-driving", "_kws": "path planning,motion planning,ROS,simulation,CARLA,behaviour planning,deep learning,end-to-end,safety case"},
            },
        },
        "🌿 Ecology & Environment": {
            "_subject": "ecology and environment",
            "_desc": "ecology, biodiversity, and environmental science",
            "_kws": "ecology,biodiversity,ecosystem,conservation,deforestation,species,habitat,pollution,rewilding",
            "Biodiversity & Conservation": {
                "_subject": "biodiversity and conservation",
                "_desc": "species conservation, habitat loss, and extinction",
                "_kws": "biodiversity,conservation,extinction,endangered species,habitat loss,rewilding,IUCN,protected area,wildlife",
                "Rewilding": {"_subject": "rewilding", "_desc": "rewilding projects, keystone species, and ecological restoration", "_kws": "rewilding,keystone species,trophic cascade,wolf reintroduction,Knepp,ecological restoration,apex predator,Pleistocene"},
                "Marine Conservation": {"_subject": "marine conservation", "_desc": "ocean biodiversity, coral reefs, and marine protected areas", "_kws": "marine conservation,coral reef,ocean,overfishing,MPA,whale,shark,plastic pollution,blue carbon,kelp forest"},
            },
            "Pollution & Waste": {
                "_subject": "pollution and waste management",
                "_desc": "air, water, soil pollution, and circular economy",
                "_kws": "pollution,plastic,microplastic,air quality,water pollution,soil contamination,circular economy,waste,recycling",
                "Plastic Pollution": {"_subject": "plastic pollution", "_desc": "plastic waste, microplastics, and ocean pollution", "_kws": "plastic pollution,microplastic,single-use plastic,ocean plastic,nanoplastic,ingestion,biodegradable,policy"},
                "Air Quality": {"_subject": "air quality and pollution", "_desc": "air pollutants, health effects, and monitoring", "_kws": "air quality,PM2.5,NOx,ozone,particulate matter,smog,AQI,emissions,health impact,clean air"},
            },
        },
        "🔐 Blockchain & Crypto": {
            "_subject": "blockchain and cryptocurrency",
            "_desc": "blockchain technology, cryptocurrencies, and decentralized finance",
            "_kws": "blockchain,Bitcoin,Ethereum,DeFi,smart contract,NFT,consensus,wallet,Web3",
            "Cryptocurrencies": {
                "_subject": "cryptocurrencies",
                "_desc": "Bitcoin, Ethereum, altcoins, and crypto markets",
                "_kws": "Bitcoin,Ethereum,altcoin,market cap,trading,wallet,private key,exchange,crypto,price",
                "Bitcoin": {"_subject": "Bitcoin", "_desc": "Bitcoin protocol, mining, and store of value thesis", "_kws": "Bitcoin,BTC,mining,SHA-256,halving,Lightning Network,UTXO,mempool,full node,Satoshi Nakamoto"},
                "Ethereum & Smart Contracts": {"_subject": "Ethereum and smart contracts", "_desc": "Ethereum, Solidity, and dApp development", "_kws": "Ethereum,Solidity,smart contract,EVM,gas,ERC-20,ERC-721,dApp,Remix,Hardhat,Foundry"},
            },
            "DeFi": {
                "_subject": "decentralized finance",
                "_desc": "DeFi protocols, yield farming, and liquidity",
                "_kws": "DeFi,liquidity pool,AMM,yield farming,lending,borrowing,Uniswap,Aave,Compound,impermanent loss",
                "Lending & Borrowing": {"_subject": "DeFi lending and borrowing", "_desc": "collateralised lending, liquidations, and money markets", "_kws": "DeFi lending,Aave,Compound,collateral,liquidation,over-collateralised,interest rate,flash loan,MakerDAO"},
                "DEX & AMM": {"_subject": "decentralized exchanges and AMMs", "_desc": "AMM mechanics, liquidity provision, and DEX protocols", "_kws": "DEX,AMM,Uniswap,Curve,constant product,liquidity pool,impermanent loss,slippage,concentrated liquidity"},
            },
        },
        "🎵 Music Theory & Production": {
            "_subject": "music theory and production",
            "_desc": "music theory, composition, and audio production",
            "_kws": "music theory,harmony,chord progression,mixing,mastering,DAW,synthesis,arrangement,ear training",
            "Music Theory": {
                "_subject": "music theory",
                "_desc": "harmony, melody, rhythm, and music analysis",
                "_kws": "music theory,harmony,melody,rhythm,chord,scale,interval,cadence,counterpoint,Roman numeral",
                "Harmony & Chords": {"_subject": "harmony and chord theory", "_desc": "chord construction, progressions, and harmonic analysis", "_kws": "chord,harmony,progression,ii-V-I,tritone substitution,modal harmony,voice leading,Roman numeral,reharmonisation"},
                "Counterpoint": {"_subject": "counterpoint", "_desc": "species counterpoint, voice leading, and polyphony", "_kws": "counterpoint,species,voice leading,Fux,consonance,dissonance,fugue,canon,imitation,Bach"},
                "Ear Training": {"_subject": "ear training and aural skills", "_desc": "interval recognition, chord identification, and dictation", "_kws": "ear training,interval,solfege,dictation,relative pitch,perfect pitch,melodic dictation,chord recognition"},
            },
            "Audio Production": {
                "_subject": "audio production",
                "_desc": "mixing, mastering, and sound engineering",
                "_kws": "mixing,mastering,EQ,compression,DAW,Logic Pro,Ableton,Pro Tools,reverb,delay,gain staging",
                "Mixing": {"_subject": "audio mixing", "_desc": "mixing techniques, EQ, compression, and spatial processing", "_kws": "mixing,EQ,compression,reverb,delay,stereo field,gain staging,bus,send,automation,balance"},
                "Sound Synthesis": {"_subject": "sound synthesis", "_desc": "subtractive, additive, FM, and wavetable synthesis", "_kws": "synthesis,subtractive,FM synthesis,wavetable,granular,oscillator,filter,envelope,LFO,modular,Moog"},
            },
        },
        "📱 Mobile App Development": {
            "_subject": "mobile app development",
            "_desc": "iOS and Android app development, React Native, and Flutter",
            "_kws": "iOS,Android,React Native,Flutter,Swift,Kotlin,mobile UI,app store,push notifications",
            "iOS Development": {
                "_subject": "iOS development",
                "_desc": "Swift, SwiftUI, UIKit, and App Store publishing",
                "_kws": "iOS,Swift,SwiftUI,UIKit,Xcode,App Store,Core Data,Combine,MVVM,TestFlight",
                "SwiftUI": {"_subject": "SwiftUI development", "_desc": "SwiftUI views, state management, and declarative UI", "_kws": "SwiftUI,View,State,Binding,ObservableObject,NavigationStack,modifier,animation,preview,environment"},
                "Core Data & Persistence": {"_subject": "iOS Core Data and persistence", "_desc": "Core Data, CloudKit, and local storage on iOS", "_kws": "Core Data,NSManagedObject,persistent container,CloudKit,UserDefaults,Keychain,SQLite,migration,fetch request"},
            },
            "Android Development": {
                "_subject": "Android development",
                "_desc": "Kotlin, Jetpack Compose, and Google Play publishing",
                "_kws": "Android,Kotlin,Jetpack Compose,Android Studio,Play Store,Room,Hilt,Coroutines,ViewModel,MVVM",
                "Jetpack Compose": {"_subject": "Jetpack Compose", "_desc": "Jetpack Compose, composables, and modern Android UI", "_kws": "Jetpack Compose,Composable,state,recomposition,LazyColumn,Material3,ViewModel,navigation,animation"},
                "Kotlin Coroutines": {"_subject": "Kotlin coroutines and async", "_desc": "coroutines, Flow, and async programming on Android", "_kws": "coroutines,Flow,suspend,launch,async,StateFlow,SharedFlow,Dispatchers,structured concurrency,Retrofit"},
            },
            "Cross-Platform": {
                "_subject": "cross-platform mobile development",
                "_desc": "React Native, Flutter, and cross-platform strategies",
                "_kws": "React Native,Flutter,Expo,Dart,cross-platform,code sharing,native modules,performance,bridge",
                "React Native": {"_subject": "React Native development", "_desc": "React Native components, navigation, and native integration", "_kws": "React Native,Expo,JavaScript,TypeScript,navigation,FlatList,native module,bridge,New Architecture,Hermes"},
                "Flutter": {"_subject": "Flutter development", "_desc": "Flutter widgets, Dart, and cross-platform UI", "_kws": "Flutter,Dart,widget,StatefulWidget,Provider,Riverpod,BLoC,pub.dev,platform channel,Skia"},
            },
        },
        "🏗️ Architecture & Engineering": {
            "_subject": "architecture and engineering",
            "_desc": "architecture, structural engineering, and construction",
            "_kws": "architecture,structural engineering,BIM,AutoCAD,load bearing,foundation,sustainable design,HVAC",
            "Structural Engineering": {
                "_subject": "structural engineering",
                "_desc": "structural analysis, materials, and building systems",
                "_kws": "structural engineering,load,beam,column,foundation,steel,concrete,seismic,Eurocodes,finite element",
                "Seismic Design": {"_subject": "seismic engineering", "_desc": "earthquake-resistant design and structural dynamics", "_kws": "seismic,earthquake,base isolation,ductility,damper,response spectrum,PGA,Eurocode 8,shear wall"},
                "Materials Science": {"_subject": "construction materials", "_desc": "concrete, steel, timber, and advanced materials", "_kws": "concrete,steel,timber,carbon fibre,composite,prestressed,reinforcement,mix design,durability,UHPC"},
            },
            "Sustainable Design": {
                "_subject": "sustainable architecture",
                "_desc": "green buildings, passive design, and net zero construction",
                "_kws": "sustainable design,LEED,BREEAM,passive house,net zero,embodied carbon,thermal mass,natural ventilation",
                "Passive House": {"_subject": "Passive House design", "_desc": "Passivhaus standard, thermal envelope, and energy efficiency", "_kws": "Passive House,Passivhaus,PHPP,airtightness,thermal bridge,MVHR,triple glazing,heating demand,EnerPHit"},
                "Net Zero Buildings": {"_subject": "net zero buildings", "_desc": "net zero energy design, whole-life carbon, and embodied carbon", "_kws": "net zero,embodied carbon,operational carbon,RIBA 2030,lifecycle,whole-life carbon,carbon offsetting,energy model"},
            },
            "Urban Planning": {
                "_subject": "urban planning and design",
                "_desc": "urban development, smart cities, and planning policy",
                "_kws": "urban planning,zoning,masterplan,smart city,transport,housing,density,green space,15-minute city",
                "Smart Cities": {"_subject": "smart cities", "_desc": "IoT, data, and technology for urban management", "_kws": "smart city,IoT,sensors,data platform,mobility,digital twin,traffic management,energy grid,citizen"},
                "Housing & Density": {"_subject": "housing and urban density", "_desc": "housing policy, density, and affordable housing", "_kws": "housing,density,affordable,social housing,planning permission,zoning,mixed-use,transit-oriented development"},
            },
        },
    }

    def _get_subcats(node: dict) -> dict:
        return {k: v for k, v in node.items() if not k.startswith("_")}

    # ── Level 1 ───────────────────────────────────────────────────────────────
    l1 = st.session_state.get("preset_l1")
    l2 = st.session_state.get("preset_l2")
    l3 = st.session_state.get("preset_l3")

    cols = st.columns(5)
    for i, key in enumerate(PRESETS):
        with cols[i % 5]:
            btn_type = "primary" if l1 == key else "secondary"
            if st.button(preset_label(key), key=f"l1_{key}", use_container_width=True, type=btn_type):
                st.session_state["preset_l1"] = key
                st.session_state.pop("preset_l2", None)
                st.session_state.pop("preset_l3", None)
                st.rerun()

    # ── Level 2 ───────────────────────────────────────────────────────────────
    if l1 and l1 in PRESETS:
        subcats_l2 = _get_subcats(PRESETS[l1])
        if subcats_l2:
            _drill = t("step2_desc").split(".")[0]  # short hint
            st.markdown(f"**{preset_label(l1)}** — {t('step2_desc').split('.')[0].lower()}:")
            cols2 = st.columns(min(5, len(subcats_l2)))
            for i, key in enumerate(subcats_l2):
                with cols2[i % 5]:
                    btn_type = "primary" if l2 == key else "secondary"
                    if st.button(preset_label(key), key=f"l2_{key}", use_container_width=True, type=btn_type):
                        st.session_state["preset_l2"] = key
                        st.session_state.pop("preset_l3", None)
                        st.rerun()

    # ── Level 3 ───────────────────────────────────────────────────────────────
    if l1 and l2 and l1 in PRESETS:
        subcats_l2 = _get_subcats(PRESETS[l1])
        if l2 in subcats_l2:
            subcats_l3 = _get_subcats(subcats_l2[l2])
            if subcats_l3:
                st.markdown(f"**{preset_label(l1)} → {preset_label(l2)}**:")
                cols3 = st.columns(min(5, len(subcats_l3)))
                for i, key in enumerate(subcats_l3):
                    with cols3[i % 5]:
                        btn_type = "primary" if l3 == key else "secondary"
                        if st.button(preset_label(key), key=f"l3_{key}", use_container_width=True, type=btn_type):
                            st.session_state["preset_l3"] = key
                            st.rerun()

    # ── Resolve active preset ─────────────────────────────────────────────────
    active_data = {"subject": "", "desc": "", "kws": ""}
    if l1 and l1 in PRESETS:
        active_data = get_preset_meta(PRESETS[l1], l1)
        subcats_l2 = _get_subcats(PRESETS[l1])
        if l2 and l2 in subcats_l2:
            active_data = get_preset_meta(subcats_l2[l2], l2)
            subcats_l3 = _get_subcats(subcats_l2[l2])
            if l3 and l3 in subcats_l3:
                active_data = get_preset_meta(subcats_l3[l3], l3)

    # Breadcrumb — show translated labels
    breadcrumb = " → ".join(preset_label(p) for p in [l1, l2, l3] if p)
    if breadcrumb:
        st.markdown(f"**{t('selected')}** {breadcrumb}")
    st.markdown("---")

    cur_subject = get_env("SUBJECT", active_data["subject"])
    cur_desc    = get_env("SUBJECT_DESCRIPTION", active_data["desc"])
    cur_kws     = get_env("SUBJECT_KEYWORDS", active_data["kws"])

    subject_val = st.text_input(t("subject_name"), value=active_data["subject"] or cur_subject)
    desc_val    = st.text_input(t("description"),  value=active_data["desc"] or cur_desc)
    kws_val     = st.text_input(t("keywords"), value=active_data["kws"] or cur_kws)

    st.markdown("---")

    # ── Step 3: Region ────────────────────────────────────────────────────────
    st.markdown(f"### {t('step3_title')}")
    st.markdown(t("step3_desc"))

    region_options = [
        t("region_anywhere"),
        t("region_spain"),
        t("region_catalunya"),
        t("region_europe"),
    ]
    region_values = ["anywhere", "spain", "catalunya", "europe"]
    cur_region = get_env("REGION", "anywhere")
    region_idx = region_values.index(cur_region) if cur_region in region_values else 0
    region_sel = st.selectbox(
        t("region_label"),
        region_options,
        index=region_idx,
    )
    region_val = region_values[region_options.index(region_sel)]

    st.markdown("---")

    # ── Save ──────────────────────────────────────────────────────────────────
    if st.button(t("save_btn"), type="primary", use_container_width=True):
        updates = {}
        if api_key:
            updates[env_key] = api_key
        if subject_val:
            updates["SUBJECT"] = subject_val
        if desc_val:
            updates["SUBJECT_DESCRIPTION"] = desc_val
        if kws_val:
            updates["SUBJECT_KEYWORDS"] = kws_val
        updates["REGION"] = region_val

        if not updates.get(env_key) and not get_env(env_key):
            st.error(t("save_no_key"))
        else:
            write_env(updates)
            st.success(t("save_ok"))
            time.sleep(0.8)
            st.rerun()

    if is_configured():
        st.info(t("setup_done"))


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Home":
    subject = get_env("SUBJECT", "your subject")
    st.markdown(f'<div class="page-title">🧠 AI Subject Matter Expert</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{get_env("SUBJECT_DESCRIPTION", subject)}</div>', unsafe_allow_html=True)

    if not configured:
        st.warning(t("not_configured_warn"))

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
    metric(c1, f"{raw_n:,}",     t("raw_documents"))
    metric(c2, f"{proc_n:,}",    t("processed_label"))
    metric(c3, f"{kb_total:,}",  t("kb_chunks"))
    metric(c4, t("kb_ready") if kb_total > 0 else t("kb_empty"), t("status"),
           "#16a34a" if kb_total > 0 else "#dc2626")

    st.markdown("---")

    # ── Quick start ───────────────────────────────────────────────────────────
    if kb_total == 0 and configured:
        st.markdown(f"### {t('quick_start')}")
        st.markdown(t("quick_start_desc"))

        col_left, col_right = st.columns([2, 1])
        with col_left:
            qs_max   = st.number_input(t("max_docs"), 10, 500, 50, step=10)
            qs_fast  = st.checkbox(t("fast_mode"), value=False)
        with col_right:
            st.markdown("<br/>", unsafe_allow_html=True)
            run_all = st.button(t("build_everything"), type="primary", use_container_width=True)

        if run_all:
            _lang = get_env("LANGUAGE", "en")
            _region = get_env("REGION", "anywhere")
            log_ph = st.empty()
            with st.spinner(""):
                st.markdown(t("step1_research"))
                rc = stream_command(
                    ["scripts/research.py", "--max", str(qs_max),
                     "--language", _lang, "--region", _region], log_ph
                )
                if rc != 0:
                    st.error(t("research_failed"))
                    st.stop()

                st.markdown(t("step2_process"))
                cmd = ["scripts/process_data.py"]
                if qs_fast:
                    cmd.append("--no-structure")
                rc = stream_command(cmd, log_ph)
                if rc != 0:
                    st.error(t("process_failed"))
                    st.stop()

                st.markdown(t("step3_build"))
                rc = stream_command(["scripts/build_rag.py"], log_ph)

            if rc == 0:
                st.success(t("kb_ready_msg"))
                st.balloons()
                st.cache_resource.clear()

    elif kb_total > 0:
        st.markdown(t("kb_section_header"))
        c1, c2, c3 = st.columns(3)
        with c1: st.metric(t("kb_documents"), f"{stats.get('documents',0):,} {t('kb_chunks')}")
        with c2: st.metric(t("kb_learnings"),  f"{stats.get('learnings',0):,} {t('kb_chunks')}")
        with c3: st.metric(t("kb_summaries"),  f"{stats.get('summaries',0):,} {t('kb_chunks')}")
        st.markdown(t("kb_ready_goto"))

    st.markdown("---")
    st.markdown(t("how_it_works"))
    steps = [
        ("1", t("step_research"), t("step_research_desc")),
        ("2", t("step_process"),  t("step_process_desc")),
        ("3", t("step_index"),    t("step_index_desc")),
        ("4", t("step_chat"),     t("step_chat_desc")),
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
    st.markdown(f'<div class="page-title">{t("chat_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("chat_sub").format(subject=subject)}</div>', unsafe_allow_html=True)

    if not configured:
        st.error(t("chat_no_config"))
        st.stop()

    # Load agent (cached)
    @st.cache_resource(show_spinner=True)
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
        st.info(t("chat_no_kb"))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {t('sources').format(n=len(msg['sources']))}"):
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

    if prompt := st.chat_input(t("chat_input").format(subject=subject)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(t("thinking")):
                try:
                    answer, sources = agent.chat_with_sources(prompt)
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"📚 {t('sources').format(n=len(sources))}"):
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
        if st.button(t("clear_conv")):
            st.session_state.messages = []
            agent.reset()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Pipeline":
    subject = get_env("SUBJECT", "your subject")
    st.markdown(f'<div class="page-title">{t("pipeline_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("pipeline_sub").format(subject=subject)}</div>', unsafe_allow_html=True)

    if not configured:
        st.error(t("pipeline_no_config"))
        st.stop()

    # Status bar
    c1, c2, c3 = st.columns(3)
    with c1: st.metric(t("pipeline_raw"), f"{raw_n:,}", delta=t("pipeline_fetched"))
    with c2: st.metric(t("pipeline_processed"), f"{proc_n:,}", delta=t("pipeline_structured"))
    stats = kb_stats()
    with c3: st.metric(t("kb_chunks_label"), f"{sum(stats.values()):,}", delta=t("kb_indexed"))

    st.markdown("---")

    # ── One-click ────────────────────────────────────────────────────────────
    with st.expander(t("build_everything_exp"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            max_docs  = st.number_input(t("max_docs"), 10, 1000, 100, step=10, key="be_max")
            n_queries = st.number_input(t("ai_queries"), 5, 30, 10, key="be_q")
        with col2:
            fast_mode  = st.checkbox(t("fast_mode2"), key="be_fast")
            reprocess  = st.checkbox(t("reprocess"), key="be_reprocess")
            rebuild    = st.checkbox(t("rebuild"), key="be_rebuild")

        if st.button(t("build_everything"), type="primary", use_container_width=True, key="be_btn"):
            log_ph = st.empty()
            success = True
            _lang = get_env("LANGUAGE", "en")
            _region = get_env("REGION", "anywhere")

            st.markdown(t("step_research_title"))
            rc = stream_command(
                ["scripts/research.py", "--max", str(max_docs), "--queries", str(n_queries),
                 "--language", _lang, "--region", _region],
                log_ph,
            )
            if rc != 0: success = False

            if success:
                st.markdown(t("step_process_title"))
                cmd = ["scripts/process_data.py"]
                if fast_mode:  cmd.append("--no-structure")
                if reprocess:  cmd.append("--reprocess")
                rc = stream_command(cmd, log_ph)
                if rc != 0: success = False

            if success:
                st.markdown(t("step_build_title"))
                cmd = ["scripts/build_rag.py"]
                if rebuild: cmd.append("--rebuild")
                rc = stream_command(cmd, log_ph)
                if rc != 0: success = False

            if success:
                st.success(t("done_msg"))
                st.balloons()
                st.cache_resource.clear()
            else:
                st.error(t("step_failed"))

    st.markdown("---")

    # ── Individual steps ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([t("tab_research"), t("tab_process"), t("tab_build")])

    with tab1:
        st.markdown(t("research_desc"))
        c1, c2, c3 = st.columns(3)
        with c1: t1_max = st.number_input(t("max_docs_short"), 10, 1000, 100, step=10)
        with c2: t1_q   = st.number_input(t("queries_label"), 5, 30, 10)
        with c3:
            t1_src = st.selectbox(t("source_select"), ["All", "Web (AI)", "Wikipedia"])

        if st.button(t("run_research"), type="primary"):
            _lang = get_env("LANGUAGE", "en")
            _region = get_env("REGION", "anywhere")
            cmd = ["scripts/research.py", "--max", str(t1_max), "--queries", str(t1_q),
                   "--language", _lang, "--region", _region]
            if t1_src == "Web (AI)": cmd.append("--web")
            elif t1_src == "Wikipedia": cmd.append("--wikipedia")
            ph = st.empty()
            rc = stream_command(cmd, ph)
            (st.success if rc == 0 else st.error)(
                t("research_complete") if rc == 0 else t("research_fail")
            )

    with tab2:
        st.markdown(t("process_desc"))
        c1, c2 = st.columns(2)
        with c1:
            t2_limit = st.number_input(t("limit"), 0, 10000, 0, step=50)
            t2_fast  = st.checkbox(t("fast_model"))
        with c2:
            t2_reprocess = st.checkbox(t("reprocess2"))
            t2_nostr     = st.checkbox(t("no_structure"))

        if st.button(t("run_process"), type="primary"):
            cmd = ["scripts/process_data.py"]
            if t2_limit:      cmd += ["--limit", str(t2_limit)]
            if t2_fast:       cmd.append("--fast-model")
            if t2_reprocess:  cmd.append("--reprocess")
            if t2_nostr:      cmd.append("--no-structure")
            ph = st.empty()
            rc = stream_command(cmd, ph)
            (st.success if rc == 0 else st.error)(
                t("process_complete") if rc == 0 else t("process_fail")
            )

    with tab3:
        st.markdown(t("build_desc"))
        t3_rebuild = st.checkbox(t("force_rebuild"))

        if st.button(t("build_kb"), type="primary"):
            cmd = ["scripts/build_rag.py"]
            if t3_rebuild: cmd.append("--rebuild")
            ph = st.empty()
            rc = stream_command(cmd, ph)
            if rc == 0:
                st.success(t("build_complete"))
                st.cache_resource.clear()
            else:
                st.error(t("build_fail"))

    # ── Danger zone ───────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander(t("danger_title")):
        st.markdown(
            f'<div class="card card-red">'
            f"{t('danger_warn')}<br/>"
            f"<span style='color:#64748b;font-size:0.88rem'>{t('danger_sub')}</span></div>",
            unsafe_allow_html=True,
        )

        confirm = st.checkbox(t("danger_confirm"))
        if confirm:
            if st.button(t("danger_btn"), type="primary", use_container_width=True):
                import shutil
                deleted = []
                errors = []
                for folder, label in [
                    (ROOT / "data" / "raw",       t("pipeline_raw")),
                    (ROOT / "data" / "processed",  t("pipeline_processed")),
                    (ROOT / "data" / "vector_db",  "Vector DB"),
                ]:
                    try:
                        if folder.exists():
                            shutil.rmtree(folder)
                            folder.mkdir(parents=True, exist_ok=True)
                            deleted.append(label)
                    except Exception as e:
                        errors.append(f"{label}: {e}")

                st.cache_resource.clear()

                if errors:
                    st.error("\n".join(errors))
                else:
                    st.success(t("danger_ok"))
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Knowledge Base":
    st.markdown(f'<div class="page-title">{t("kb_page_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("kb_page_sub")}</div>', unsafe_allow_html=True)

    stats = kb_stats()
    total = sum(stats.values())
    if total == 0:
        st.warning(t("kb_empty_warn"))
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1: st.metric(t("kb_documents"), f"{stats.get('documents',0):,} {t('kb_chunks')}")
    with c2: st.metric(t("kb_learnings"), f"{stats.get('learnings',0):,} {t('kb_chunks')}")
    with c3: st.metric(t("kb_summaries"), f"{stats.get('summaries',0):,} {t('kb_chunks')}")

    st.markdown("---")
    st.subheader(t("kb_search_header"))

    @st.cache_resource(show_spinner=False)
    def get_store():
        from src.rag.vector_store import VectorStore
        return VectorStore()

    query = st.text_input(t("kb_search_input"), placeholder=t("kb_search_placeholder"))
    col1, col2 = st.columns(2)
    with col1: k = st.slider(t("kb_results_slider"), 1, 20, 5)
    with col2: collection = st.selectbox(t("kb_collection"), ["documents", "learnings", "summaries"])

    if query:
        store = get_store()
        with st.spinner(t("kb_searching")):
            try:
                results = store.search_with_score(query, k=k, collection=collection)
            except Exception as e:
                st.error(t("kb_search_error").format(col=collection, err=e))
                st.stop()
        if not results:
            st.info(t("kb_no_results").format(col=collection))
        else:
            st.markdown(t("kb_n_results").format(n=len(results), col=collection))
            for doc, score in results:
                meta = doc.metadata
                url   = meta.get("url", "")
                title = meta.get("title", "Unknown")
                with st.expander(f"{title}  ·  score {score:.3f}"):
                    cols = st.columns(2)
                    with cols[0]:
                        if url: st.markdown(f"[{meta.get('source_name', url)}]({url})")
                        st.caption(f"{t('kb_date')} {meta.get('date','—')}")
                    with cols[1]:
                        topics = meta.get("topics", "")
                        if topics:
                            chips = "".join(f'<span class="source-chip">{tg.strip()}</span>' for tg in topics.split(",") if tg.strip())
                            st.markdown(chips, unsafe_allow_html=True)
                    st.text(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.markdown(f'<div class="page-title">{t("dataset_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("dataset_sub")}</div>', unsafe_allow_html=True)

    PROC_DIR = ROOT / "data" / "processed"
    docs = []
    for path in sorted(PROC_DIR.rglob("*.json")):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

    if not docs:
        st.warning(t("dataset_no_docs"))
        st.stop()

    st.metric(t("dataset_total"), len(docs))

    # Filter
    search = st.text_input(t("dataset_filter"), "")
    if search:
        docs = [d for d in docs if
                search.lower() in d.get("title", "").lower() or
                search.lower() in ",".join(d.get("topics", [])).lower()]
        st.caption(t("dataset_matching").format(n=len(docs)))

    # Table
    import pandas as pd
    rows = [{
        t("dataset_col_title"):  d.get("title","")[:70],
        t("dataset_col_source"): d.get("source_name",""),
        t("dataset_col_date"):   d.get("date",""),
        t("dataset_col_topics"): ", ".join(d.get("topics",[]))[:50],
        t("dataset_col_structured"): "✓" if d.get("structured") else "—",
    } for d in docs[:300]]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

    # Detail
    st.markdown("---")
    titles = [d.get("title", d.get("source_id","?"))[:80] for d in docs[:100]]
    sel = st.selectbox(t("dataset_view_doc"), titles)
    if sel:
        doc = next((d for d in docs if d.get("title","").startswith(sel[:40])), None)
        if doc:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"{t('dataset_source')} {doc.get('source_name','')}")
                st.write(f"{t('dataset_date')} {doc.get('date','—')}")
                st.write(f"{t('dataset_author')} {doc.get('author','—')}")
                if doc.get("url"):
                    st.write(f"{t('dataset_url')} [{doc['url']}]({doc['url']})")
            with col2:
                chips = "".join(f'<span class="source-chip">{tg}</span>' for tg in doc.get("topics",[]))
                if chips: st.markdown(chips, unsafe_allow_html=True)
                st.write(f"{t('dataset_structured_lbl')} {t('dataset_yes') if doc.get('structured') else t('dataset_no')}")
            for field, lbl_key in [("summary","dataset_summary"), ("key_points","dataset_key_points"), ("learnings","dataset_learnings")]:
                if doc.get(field):
                    with st.expander(t(lbl_key)):
                        st.write(doc[field])

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        jsonl = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs)
        subject_slug = get_env("SUBJECT", "dataset").replace(" ", "_")
        st.download_button(t("dataset_download"), data=jsonl,
                           file_name=f"{subject_slug}_dataset.jsonl",
                           mime="application/json", use_container_width=True)
    with col2:
        hf_token = get_env("HF_TOKEN")
        if not hf_token:
            st.warning(t("dataset_hf_warn"))
        else:
            repo_id  = st.text_input(t("dataset_hf_repo"),
                                     value=get_env("HF_REPO_ID", f"username/{subject_slug}-sme"))
            private  = st.checkbox(t("dataset_private"))
            if st.button(t("dataset_upload"), type="primary", use_container_width=True):
                ph = st.empty()
                cmd = ["scripts/upload_to_hf.py", "--repo-id", repo_id]
                if private: cmd.append("--private")
                rc = stream_command(cmd, ph)
                if rc == 0:
                    st.success(t("dataset_upload_ok").format(url=f"https://huggingface.co/datasets/{repo_id}"))


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown(f'<div class="page-title">{t("about_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("about_sub")}</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(f"""
<div class="card card-blue">
<h2 style="margin-top:0">Albert G.D.</h2>
<p style="color:#475569;font-size:1rem">
{t("about_bio")}
</p>
<p>
{t("about_project_desc")}
</p>
<p>
{t("about_goal")}
</p>
</div>
""", unsafe_allow_html=True)

        st.markdown(t("about_project_title"))
        st.markdown(t("about_features"))

        st.markdown(t("about_contact_title"))
        st.markdown("""
<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.5rem">
  <a href="https://www.linkedin.com/in/albertgd" target="_blank"
     style="display:inline-flex;align-items:center;gap:8px;background:#0077b5;color:white;
            padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
    LinkedIn
  </a>
  <a href="https://twitter.com/albertgd" target="_blank"
     style="display:inline-flex;align-items:center;gap:8px;background:#000;color:white;
            padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.746l7.73-8.835L1.254 2.25H8.08l4.253 5.622zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
    </svg>
    X / Twitter
  </a>
  <a href="https://github.com/albertgd" target="_blank"
     style="display:inline-flex;align-items:center;gap:8px;background:#24292f;color:white;
            padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
    </svg>
    GitHub
  </a>
</div>
""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
<div class="card card-green" style="text-align:center">
  <div style="font-size:3rem">🧠</div>
  <div style="font-weight:700;font-size:1.1rem;margin:0.5rem 0">AI SME</div>
  <div style="color:#64748b;font-size:0.85rem">{t("about_tagline")}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="card card-amber" style="margin-top:0.8rem">
  <div style="font-weight:700;margin-bottom:0.4rem">{t("about_stack_title")}</div>
  <div style="font-size:0.83rem;color:#475569;line-height:1.8">
    Python · Streamlit<br/>
    ChromaDB · LangChain<br/>
    sentence-transformers<br/>
    DuckDuckGo · Wikipedia<br/>
    Groq · Gemini · Claude
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="card card-blue" style="margin-top:0.8rem;font-size:0.83rem;color:#475569">
  {t("about_belief")}
</div>
""", unsafe_allow_html=True)
