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
# We use `pip install <wheel>` directly instead of `python -m spacy download`
# because spacy download searches for pip/uv on the system PATH and fails in
# many environments (conda, nix, some Linux distros). Using pip from the venv
# directly is always reliable.
echo "  Installing spaCy language model..."
SPACY_VER=$(python -c "import spacy; v=spacy.__version__; p=v.split('.'); print(f'{p[0]}.{p[1]}.0')" 2>/dev/null || echo "3.8.0")
SPACY_BASE="https://github.com/explosion/spacy-models/releases/download"

pip install \
    "en_core_web_lg @ ${SPACY_BASE}/en_core_web_lg-${SPACY_VER}/en_core_web_lg-${SPACY_VER}-py3-none-any.whl" \
    --quiet 2>/dev/null && echo -e "${GREEN}✓${NC} spaCy model ready (en_core_web_lg)" || \
pip install \
    "en_core_web_sm @ ${SPACY_BASE}/en_core_web_sm-${SPACY_VER}/en_core_web_sm-${SPACY_VER}-py3-none-any.whl" \
    --quiet 2>/dev/null && echo -e "${YELLOW}  ✓ spaCy sm model installed (lg unavailable)${NC}" || \
echo -e "${YELLOW}  ⚠ spaCy model not installed — PII removal will use regex only${NC}"

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
echo -e "    1. Enter your API key (Groq is free — get one at console.groq.com)"
echo -e "    2. Choose a subject"
echo -e "    3. Click 'Build Everything'"
echo -e "    4. Chat with your expert"
echo ""
