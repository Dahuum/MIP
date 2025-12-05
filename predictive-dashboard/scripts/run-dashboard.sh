#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo ""
echo -e "${BLUE}⚛️  Starting Dashboard on port 3000...${NC}"
echo -e "${CYAN}   Press Ctrl+C to stop${NC}"
echo ""

# Check if .next exists (production build)
if [[ -d ".next" ]]; then
    npm run start
else
    echo -e "${CYAN}   Building first (this may take a moment)...${NC}"
    npm run build && npm run start
fi
