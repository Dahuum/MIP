#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Create PID directory
mkdir -p .pids

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🔧 Starting in Development Mode (Hot Reload)             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down services...${NC}"
    if [[ -f .pids/api.pid ]]; then
        kill $(cat .pids/api.pid) 2>/dev/null
        rm .pids/api.pid
    fi
    if [[ -f .pids/dashboard.pid ]]; then
        kill $(cat .pids/dashboard.pid) 2>/dev/null
        rm .pids/dashboard.pid
    fi
    echo -e "${GREEN}✅ Services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start API Server
echo -e "${BLUE}🐍 Starting API Server on port 5000...${NC}"
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 5000 --reload > .pids/api.log 2>&1 &
API_PID=$!
echo $API_PID > .pids/api.pid

# Wait for API to start
sleep 2

# Check if API is running
if kill -0 $API_PID 2>/dev/null; then
    echo -e "${GREEN}✅ API Server started (PID: $API_PID)${NC}"
else
    echo -e "${RED}❌ Failed to start API Server. Check .pids/api.log for errors${NC}"
    cat .pids/api.log
    exit 1
fi

echo ""

# Start Dashboard in dev mode
echo -e "${BLUE}⚛️  Starting Dashboard in development mode...${NC}"
npm run dev > .pids/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo $DASHBOARD_PID > .pids/dashboard.pid

# Wait for dashboard to start
sleep 5

# Check if dashboard is running
if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo -e "${GREEN}✅ Dashboard started (PID: $DASHBOARD_PID)${NC}"
else
    echo -e "${RED}❌ Failed to start Dashboard. Check .pids/dashboard.log for errors${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     🎉 Development mode is running!                          ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  📊 Dashboard:  ${CYAN}http://localhost:3000${GREEN}                       ║${NC}"
echo -e "${GREEN}║  🔌 API Server: ${CYAN}http://localhost:5000${GREEN}                       ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  ${YELLOW}Changes will auto-reload!${GREEN}                                  ║${NC}"
echo -e "${GREEN}║  Press Ctrl+C to stop all services                           ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Keep script running and wait for both processes
wait $API_PID $DASHBOARD_PID
