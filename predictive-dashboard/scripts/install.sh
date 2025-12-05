#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     📦 Installing Project Dependencies                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"

# Install Python dependencies
echo -e "${BLUE}🐍 Installing Python dependencies...${NC}"
if [[ -f "requirements.txt" ]]; then
    pip3 install -r requirements.txt --quiet
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    # Create requirements.txt if it doesn't exist
    echo -e "${YELLOW}⚠️  Creating requirements.txt...${NC}"
    cat > requirements.txt << 'EOF'
flask>=2.3.0
flask-cors>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
torch>=2.0.0
scikit-learn>=1.3.0
EOF
    pip3 install -r requirements.txt --quiet
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
fi

echo ""

# Install Node.js dependencies
echo -e "${BLUE}📦 Installing Node.js dependencies...${NC}"
if [[ -f "package.json" ]]; then
    npm install --silent 2>/dev/null
    echo -e "${GREEN}✅ Node.js dependencies installed${NC}"
else
    echo -e "${RED}❌ package.json not found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     ✅ All dependencies installed!                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
