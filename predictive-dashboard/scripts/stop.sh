#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo ""
echo -e "${YELLOW}🛑 Stopping all services...${NC}"
echo ""

# Stop API server
if [[ -f .pids/api.pid ]]; then
    API_PID=$(cat .pids/api.pid)
    if kill -0 $API_PID 2>/dev/null; then
        kill $API_PID 2>/dev/null
        echo -e "${GREEN}✅ API Server stopped (PID: $API_PID)${NC}"
    fi
    rm .pids/api.pid
else
    echo -e "${YELLOW}⚠️  API Server not running${NC}"
fi

# Stop Dashboard
if [[ -f .pids/dashboard.pid ]]; then
    DASHBOARD_PID=$(cat .pids/dashboard.pid)
    if kill -0 $DASHBOARD_PID 2>/dev/null; then
        kill $DASHBOARD_PID 2>/dev/null
        echo -e "${GREEN}✅ Dashboard stopped (PID: $DASHBOARD_PID)${NC}"
    fi
    rm .pids/dashboard.pid
else
    echo -e "${YELLOW}⚠️  Dashboard not running${NC}"
fi

# Also try to kill any orphaned processes
pkill -f "uvicorn api_server:app" 2>/dev/null
pkill -f "next-server" 2>/dev/null
pkill -f "next dev" 2>/dev/null

echo ""
echo -e "${GREEN}✅ All services stopped${NC}"
echo ""
