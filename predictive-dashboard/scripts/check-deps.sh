#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🔍 Checking System Dependencies                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo -e "${BLUE}📱 Detected: macOS${NC}"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo -e "${BLUE}🐧 Detected: Linux${NC}"
else
    echo -e "${YELLOW}⚠️  Unknown OS: $OSTYPE${NC}"
fi

echo ""

# Check for Homebrew (macOS) or apt (Linux)
install_package_manager() {
    if [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}📦 Homebrew not found. Installing...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            
            # Add Homebrew to PATH for Apple Silicon Macs
            if [[ -f "/opt/homebrew/bin/brew" ]]; then
                eval "$(/opt/homebrew/bin/brew shellenv)"
            fi
        else
            echo -e "${GREEN}✅ Homebrew is installed${NC}"
        fi
    fi
}

# Check and install Python
check_python() {
    echo -e "${BLUE}🐍 Checking Python...${NC}"
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        echo -e "${GREEN}✅ Python $PYTHON_VERSION is installed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Python 3 not found. Installing...${NC}"
        if [[ "$OS" == "macos" ]]; then
            brew install python3
        elif [[ "$OS" == "linux" ]]; then
            sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
        fi
        return $?
    fi
}

# Check and install Node.js
check_node() {
    echo -e "${BLUE}📦 Checking Node.js...${NC}"
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        echo -e "${GREEN}✅ Node.js $NODE_VERSION is installed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Node.js not found. Installing...${NC}"
        if [[ "$OS" == "macos" ]]; then
            brew install node
        elif [[ "$OS" == "linux" ]]; then
            # Install Node.js via NodeSource
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt-get install -y nodejs
        fi
        return $?
    fi
}

# Check and install npm
check_npm() {
    echo -e "${BLUE}📦 Checking npm...${NC}"
    
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        echo -e "${GREEN}✅ npm $NPM_VERSION is installed${NC}"
        return 0
    else
        echo -e "${RED}❌ npm not found (should come with Node.js)${NC}"
        return 1
    fi
}

# Check pip
check_pip() {
    echo -e "${BLUE}📦 Checking pip...${NC}"
    
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
        echo -e "${GREEN}✅ pip $PIP_VERSION is installed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  pip not found. Installing...${NC}"
        python3 -m ensurepip --upgrade
        return $?
    fi
}

# Main execution
install_package_manager
echo ""
check_python
echo ""
check_pip
echo ""
check_node
echo ""
check_npm

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     ✅ Dependency check complete!                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
