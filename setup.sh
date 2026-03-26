#!/usr/bin/env bash
# AI Subject Matter Expert — setup script
set -e

echo "================================================="
echo "  AI Subject Matter Expert — Setup"
echo "================================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/6] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

echo "[2/6] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[3/6] Installing dependencies..."
pip install -r requirements.txt --quiet

echo "[4/6] Installing spaCy English model (for PII removal)..."
python -m spacy download en_core_web_lg --quiet || \
    python -m spacy download en_core_web_sm --quiet || \
    echo "  WARNING: spaCy model not installed. PII removal will use regex only."

echo "[5/6] Installing Playwright browsers (optional — for JS-heavy scraping)..."
playwright install chromium --quiet || \
    echo "  WARNING: Playwright not installed. JS-heavy scraping will be skipped."

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "[6/6] Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "  ACTION REQUIRED: Edit .env and add your API keys."
else
    echo "[6/6] .env already exists, skipping."
fi

echo ""
echo "================================================="
echo "  Setup complete!"
echo "================================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)"
echo "  2. Run:  source venv/bin/activate"
echo "  3. Run:  ./run.sh"
echo ""
