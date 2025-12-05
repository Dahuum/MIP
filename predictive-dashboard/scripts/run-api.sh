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
echo -e "${BLUE}🐍 Starting API Server on port 5000...${NC}"
echo -e "${CYAN}   Press Ctrl+C to stop${NC}"
echo ""

python3 -m uvicorn api_server:app --host 0.0.0.0 --port 5000 --reload
