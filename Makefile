# Predictive Maintenance Dashboard - Makefile
# Works on macOS and Linux
# Run from the MIP root directory

.PHONY: all install run stop clean help setup-venv check-python check-node install-deps start-services

# Project directory
DASHBOARD_DIR := predictive-dashboard
VENV_DIR := $(DASHBOARD_DIR)/venv
PYTHON := $(VENV_DIR)/bin/python3
PIP := $(VENV_DIR)/bin/pip

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
RED := \033[0;31m
NC := \033[0m

# Default target - run everything
all: run

# Full setup and run
run:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║     🏭 OCP Predictive Maintenance System                     ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────┐"
	@echo "│  STEP 1/5: Checking Python                                   │"
	@echo "└──────────────────────────────────────────────────────────────┘"
	@$(MAKE) -s check-python
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────┐"
	@echo "│  STEP 2/5: Checking Node.js                                  │"
	@echo "└──────────────────────────────────────────────────────────────┘"
	@$(MAKE) -s check-node
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────┐"
	@echo "│  STEP 3/5: Setting up Python Virtual Environment             │"
	@echo "└──────────────────────────────────────────────────────────────┘"
	@$(MAKE) -s setup-venv
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────┐"
	@echo "│  STEP 4/5: Installing Dependencies                           │"
	@echo "└──────────────────────────────────────────────────────────────┘"
	@$(MAKE) -s install-deps
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────┐"
	@echo "│  STEP 5/5: Starting Services                                 │"
	@echo "└──────────────────────────────────────────────────────────────┘"
	@$(MAKE) -s start-services

# Check Python
check-python:
	@printf "  🐍 Python........... "
	@if ! command -v python3 &> /dev/null; then \
		echo ""; \
		echo "  ⏳ Installing Python..."; \
		if [ "$$(uname)" = "Darwin" ]; then \
			if ! command -v brew &> /dev/null; then \
				echo "  ⏳ Installing Homebrew first..."; \
				/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
			fi; \
			brew install python3; \
		else \
			sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv; \
		fi; \
		echo "  ✅ Python installed"; \
	else \
		echo "✅ $$(python3 --version)"; \
	fi

# Check Node.js
check-node:
	@printf "  📦 Node.js.......... "
	@if ! command -v node &> /dev/null; then \
		echo ""; \
		echo "  ⏳ Installing Node.js..."; \
		if [ "$$(uname)" = "Darwin" ]; then \
			brew install node; \
		else \
			curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -; \
			sudo apt-get install -y nodejs; \
		fi; \
		echo "  ✅ Node.js installed"; \
	else \
		echo "✅ $$(node --version)"; \
	fi

# Setup Python virtual environment
setup-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		printf "  🔧 Creating venv..... "; \
		for i in 1 2 3 4 5; do printf "▓"; sleep 0.1; done; \
		python3 -m venv $(VENV_DIR); \
		for i in 1 2 3 4 5; do printf "▓"; sleep 0.1; done; \
		echo " ✅ Created"; \
	else \
		echo "  🔧 Virtual env....... ✅ Already exists"; \
	fi

# Install dependencies
install-deps:
	@printf "  🐍 Python packages... "
	@for i in 1 2 3; do printf "▓"; sleep 0.1; done
	@$(PIP) install --upgrade pip -q 2>/dev/null
	@for i in 1 2 3; do printf "▓"; sleep 0.1; done
	@$(PIP) install -r $(DASHBOARD_DIR)/requirements.txt -q 2>/dev/null
	@for i in 1 2 3 4; do printf "▓"; sleep 0.1; done
	@echo " ✅ Installed"
	@printf "  📦 Node modules...... "
	@for i in 1 2 3; do printf "▓"; sleep 0.1; done
	@cd $(DASHBOARD_DIR) && npm install --silent 2>/dev/null || npm install --silent
	@for i in 1 2 3 4 5 6 7; do printf "▓"; sleep 0.1; done
	@echo " ✅ Installed"

# Start both services
start-services:
	@mkdir -p $(DASHBOARD_DIR)/.pids
	@# Kill any existing processes
	@-lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@-lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@-pkill -f "uvicorn api_server:app" 2>/dev/null || true
	@-pkill -f "next dev" 2>/dev/null || true
	@sleep 1
	@printf "  🐍 API Server........ "
	@for i in 1 2 3; do printf "▓"; sleep 0.2; done
	@cd $(DASHBOARD_DIR) && nohup $(CURDIR)/$(VENV_DIR)/bin/python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000 > $(CURDIR)/$(DASHBOARD_DIR)/.pids/api.log 2>&1 & echo $$! > $(CURDIR)/$(DASHBOARD_DIR)/.pids/api.pid
	@for i in 1 2 3 4 5 6 7; do printf "▓"; sleep 0.3; done
	@if [ -f $(DASHBOARD_DIR)/.pids/api.pid ] && kill -0 $$(cat $(DASHBOARD_DIR)/.pids/api.pid) 2>/dev/null; then \
		echo " ✅ Running (PID: $$(cat $(DASHBOARD_DIR)/.pids/api.pid))"; \
	else \
		echo " ❌ Failed"; \
		echo ""; \
		echo "  Error log:"; \
		cat $(DASHBOARD_DIR)/.pids/api.log; \
		exit 1; \
	fi
	@echo ""
	@printf "  ⚛️  Dashboard......... "
	@for i in 1 2 3 4 5; do printf "▓"; sleep 0.1; done
	@echo " 🚀 Starting..."
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ ALL SYSTEMS GO!                                          ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║                                                              ║"
	@echo "║  📊 Dashboard:  http://localhost:3000                        ║"
	@echo "║  🔌 API Server: http://localhost:8000                        ║"
	@echo "║  📚 API Docs:   http://localhost:8000/docs                   ║"
	@echo "║                                                              ║"
	@echo "║  Press Ctrl+C to stop the dashboard                          ║"
	@echo "║  Run 'make stop' to stop API server                          ║"
	@echo "║                                                              ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@cd $(DASHBOARD_DIR) && npm run dev

# Stop all services
stop:
	@echo "🛑 Stopping services..."
	@-lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@-lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@-if [ -f $(DASHBOARD_DIR)/.pids/api.pid ]; then \
		kill $$(cat $(DASHBOARD_DIR)/.pids/api.pid) 2>/dev/null; \
		rm $(DASHBOARD_DIR)/.pids/api.pid; \
	fi
	@-pkill -f "uvicorn api_server:app" 2>/dev/null || true
	@-pkill -f "next dev" 2>/dev/null || true
	@echo "✅ All services stopped"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning..."
	@rm -rf $(DASHBOARD_DIR)/node_modules
	@rm -rf $(DASHBOARD_DIR)/.next
	@rm -rf $(DASHBOARD_DIR)/.pids
	@rm -rf $(VENV_DIR)
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"

# Help
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║     🏭 OCP Predictive Maintenance Dashboard                  ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage: make [command]"
	@echo ""
	@echo "Commands:"
	@echo "  make        - Install deps and run everything"
	@echo "  make run    - Install deps and run everything"  
	@echo "  make stop   - Stop all running services"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help"
	@echo ""
