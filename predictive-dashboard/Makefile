# Predictive Maintenance Dashboard - Makefile
# Works on macOS and Linux

.PHONY: all install run stop clean check-deps help api dashboard dev

# Default target
all: help

# Check and install all dependencies, then run
start: check-deps install run

# Check system dependencies
check-deps:
	@echo "🔍 Checking system dependencies..."
	@./scripts/check-deps.sh

# Install project dependencies
install:
	@echo "📦 Installing dependencies..."
	@./scripts/install.sh

# Run both API server and dashboard
run:
	@echo "🚀 Starting Predictive Maintenance System..."
	@./scripts/run.sh

# Run only the API server
api:
	@echo "🐍 Starting API Server..."
	@./scripts/run-api.sh

# Run only the dashboard
dashboard:
	@echo "⚛️  Starting Dashboard..."
	@./scripts/run-dashboard.sh

# Development mode (runs both with hot reload)
dev:
	@echo "🔧 Starting in development mode..."
	@./scripts/run-dev.sh

# Stop all running services
stop:
	@echo "🛑 Stopping all services..."
	@./scripts/stop.sh

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf node_modules
	@rm -rf .next
	@rm -rf __pycache__
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Show help
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║     🏭 OCP Predictive Maintenance Dashboard                  ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make start      - Check deps, install, and run everything"
	@echo "  make install    - Install all dependencies"
	@echo "  make run        - Run API server and dashboard"
	@echo "  make dev        - Run in development mode (hot reload)"
	@echo "  make api        - Run only the API server"
	@echo "  make dashboard  - Run only the dashboard"
	@echo "  make stop       - Stop all running services"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  make start"
	@echo ""
