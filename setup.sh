#!/usr/bin/env bash
# AI Subject Matter Expert — one-command setup
set -e

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo ""
echo -e "${BOLD}=================================================${NC}"
echo -e "${BOLD}  🧠  AI Subject Matter Expert — Setup${NC}"
echo -e "${BOLD}=================================================${NC}"
echo ""

# Python version check
PY=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
MAJOR=$(echo $PY | cut -d. -f1)
MINOR=$(echo $PY | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ( [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ] ); then
    echo -e "${RED}ERROR: Python 3.10+ required (found $PY)${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python $PY"

# Virtual environment
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment ready"

# Dependencies
echo "  Installing dependencies (this may take a minute)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓${NC} Dependencies installed"

# spaCy model (for PII removal)
echo "  Downloading spaCy language model..."
python -m spacy download en_core_web_lg --quiet 2>/dev/null || \
    python -m spacy download en_core_web_sm --quiet 2>/dev/null || \
    echo -e "${YELLOW}  ⚠ spaCy model not installed — PII removal will use regex only${NC}"
echo -e "${GREEN}✓${NC} Language model ready"

# .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env from template"
else
    echo -e "${GREEN}✓${NC} .env already exists"
fi

echo ""
echo -e "${BOLD}=================================================${NC}"
echo -e "${GREEN}${BOLD}  Setup complete!${NC}"
echo -e "${BOLD}=================================================${NC}"
echo ""
echo -e "  Start the app:  ${BOLD}./run.sh${NC}"
echo ""
echo -e "  The app will guide you through the rest:"
echo -e "    1. Enter your API key (Anthropic / OpenAI / Google)"
echo -e "    2. Choose a subject"
echo -e "    3. Click 'Build Everything'"
echo -e "    4. Chat with your expert"
echo ""
