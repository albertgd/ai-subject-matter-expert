#!/usr/bin/env bash
# AI Subject Matter Expert — launch the web app

# Activate venv if needed
if [ -z "$VIRTUAL_ENV" ] && [ -d "venv" ]; then
    source venv/bin/activate
fi

# First-time check
if [ ! -f "venv/pyvenv.cfg" ]; then
    echo "Run ./setup.sh first."
    exit 1
fi

echo ""
echo "🧠  Starting AI Subject Matter Expert..."
echo "   Open http://localhost:8501 in your browser"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run src/app.py \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base light \
    --theme.primaryColor "#3b82f6"
