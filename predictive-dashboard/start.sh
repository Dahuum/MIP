#!/bin/bash

# OCP Predictive Maintenance Dashboard Launcher
# ==============================================
# This script starts both the Next.js frontend and FastAPI backend

echo "=================================================="
echo "🏭 OCP PREDICTIVE MAINTENANCE DASHBOARD"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the predictive-dashboard directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $API_PID 2>/dev/null
    kill $NEXT_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

echo -e "${BLUE}Starting API Server...${NC}"
python api_server.py &
API_PID=$!

# Wait for API to start
sleep 2

echo -e "${BLUE}Starting Next.js Dashboard...${NC}"
npm run dev &
NEXT_PID=$!

echo ""
echo -e "${GREEN}=================================================="
echo -e "✅ SERVERS STARTED SUCCESSFULLY"
echo -e "==================================================${NC}"
echo ""
echo -e "📊 Dashboard:  ${GREEN}http://localhost:3000${NC}"
echo -e "🔌 API:        ${GREEN}http://localhost:8000${NC}"
echo -e "📖 API Docs:   ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for both processes
wait
