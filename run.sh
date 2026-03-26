#!/usr/bin/env bash
# AI Subject Matter Expert — runner script
set -e

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ] && [ -d "venv" ]; then
    source venv/bin/activate
fi

echo ""
echo "================================================="
echo "  AI Subject Matter Expert"
echo "================================================="
echo ""
echo "What would you like to do?"
echo ""
echo "  1) Launch the web app (Streamlit UI)"
echo "  2) Collect data from public sources"
echo "  3) Process + clean data (PII removal + structuring)"
echo "  4) Build/rebuild the RAG knowledge base"
echo "  5) Upload dataset to HuggingFace"
echo "  6) Run tests"
echo "  7) Interactive CLI chat"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo "Launching Streamlit app..."
        streamlit run src/app.py
        ;;
    2)
        echo "Collecting data from public sources..."
        python scripts/collect_data.py
        ;;
    3)
        echo "Processing and cleaning data..."
        python scripts/process_data.py
        ;;
    4)
        echo "Building RAG knowledge base..."
        python scripts/build_rag.py
        ;;
    5)
        echo "Uploading dataset to HuggingFace..."
        python scripts/upload_to_hf.py
        ;;
    6)
        echo "Running tests..."
        pytest tests/ -v
        ;;
    7)
        echo "Starting CLI chat..."
        python -m src.agents.sme_agent
        ;;
    *)
        echo "Invalid choice. Launching Streamlit app..."
        streamlit run src/app.py
        ;;
esac
