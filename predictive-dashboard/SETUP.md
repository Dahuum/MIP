# 🏭 OCP Predictive Maintenance Dashboard

An AI-powered predictive maintenance system for industrial fan monitoring (Line 307 - Fan C07). Uses a Bidirectional LSTM model trained on real failure data to predict equipment failures up to 24 hours in advance.

![Dashboard Preview](./public/preview.png)

## 🚀 Quick Start

### One-Command Setup (Recommended)

```bash
make start
```

This will:
1. Check and install required dependencies (Python, Node.js)
2. Install project dependencies (pip, npm)
3. Start both the API server and Dashboard
4. Open the dashboard at http://localhost:3000

### Manual Setup

If you prefer to run commands manually:

```bash
# 1. Install dependencies
make install

# 2. Run the system
make run
```

## 📋 Requirements

The setup script will automatically install these if missing:

- **Python 3.8+** - For the ML API server
- **Node.js 18+** - For the Next.js dashboard
- **npm** - Node package manager

### Supported Operating Systems

- ✅ macOS (Intel & Apple Silicon)
- ✅ Linux (Ubuntu, Debian, etc.)

## 🎮 Available Commands

| Command | Description |
|---------|-------------|
| `make start` | Full setup: check deps, install, and run |
| `make install` | Install all dependencies |
| `make run` | Run API server + Dashboard (production) |
| `make dev` | Run in development mode (hot reload) |
| `make api` | Run only the API server |
| `make dashboard` | Run only the dashboard |
| `make stop` | Stop all running services |
| `make clean` | Remove build artifacts |
| `make help` | Show help message |

## 🌐 Access Points

Once running:

- **Dashboard**: http://localhost:3000
- **API Server**: http://localhost:5000
- **API Docs**: http://localhost:5000/docs

## 🧠 The AI Model

### Architecture
- **Type**: Bidirectional LSTM
- **Input**: 22 sensor features
- **Hidden Layers**: 128 → 64 → 32 units
- **Output**: Failure probability (0-1)

### Performance Metrics
- **Recall**: 85% (catches 85% of actual failures)
- **Precision**: 78%
- **F1 Score**: 81%
- **Imbalance Detection**: 92%

### Training Data
- **Period**: 2019 operational data
- **Failures**: 22 real failure events
- **Features**: Vibration, temperature, current, valve position, flow rates

## 📁 Project Structure

```
predictive-dashboard/
├── api_server.py          # FastAPI backend server
├── lstm_pytorch_model.pt  # Trained LSTM model
├── Makefile               # Build automation
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
├── scripts/               # Automation scripts
│   ├── check-deps.sh      # Dependency checker
│   ├── install.sh         # Installation script
│   ├── run.sh             # Production runner
│   ├── run-dev.sh         # Development runner
│   ├── run-api.sh         # API only runner
│   ├── run-dashboard.sh   # Dashboard only runner
│   └── stop.sh            # Service stopper
├── src/                   # Next.js source code
│   ├── app/               # App router pages
│   ├── components/        # React components
│   ├── lib/               # Utilities & i18n
│   └── types/             # TypeScript types
└── public/                # Static assets
```

## 🔧 Troubleshooting

### Port Already in Use

```bash
# Stop all services
make stop

# Or manually kill processes
lsof -ti:3000 | xargs kill -9
lsof -ti:5000 | xargs kill -9
```

### Dependencies Not Installing

```bash
# Check if Python and Node are installed
python3 --version
node --version
npm --version

# Manually install if needed
# macOS:
brew install python3 node

# Linux:
sudo apt-get install python3 python3-pip nodejs npm
```

### Model Not Loading

Ensure `lstm_pytorch_model.pt` is in the project root directory.

## 🌍 Language Support

The dashboard supports:
- 🇬🇧 English
- 🇫🇷 French

Switch languages using the dropdown in the header.

## 🎨 Themes

Toggle between:
- 🌙 Dark Mode
- ☀️ Light Mode

## 📊 Features

- **Real-time Risk Assessment**: Live monitoring with instant failure probability
- **Interactive Sensor Controls**: Adjust sensor values to simulate scenarios
- **Pre-configured Scenarios**: Test common failure patterns
- **Historical Analysis**: View past failure events
- **Feature Importance**: See which sensors contribute most to risk
- **Bilingual Interface**: English and French support

## 📜 License

© 2024 OCP Group - Predictive Maintenance Initiative

---

Made with ❤️ for industrial reliability
