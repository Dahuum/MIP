# OCP Predictive Maintenance Dashboard

A modern, interactive dashboard for the OCP Fan C07 predictive maintenance system, built with Next.js 14 and FastAPI.

![Dashboard Preview](./preview.png)

## 🚀 Features

### Live Simulation Mode
- **Interactive Sensor Controls**: Adjust vibration, temperature, and current values in real-time
- **Risk Gauge**: Visual representation of failure probability (0-100%)
- **Animated Fan Visualization**: Speed and color change based on risk level
- **Scenario Presets**: Pre-configured scenarios for demonstration:
  - Normal Operation
  - Dust Accumulation
  - Warning Signs
  - Critical Failure
  - High Temperature
  - Bearing Wear

### AI-Powered Predictions
- **LSTM Deep Learning Model**: Bidirectional LSTM trained on 22 historical failures
- **Root Cause Analysis**: Identifies dust (Balourd) as the primary cause of vibration-induced failures
- **24-Hour Prediction Window**: Alerts before failure occurs
- **Contributing Factors**: Shows which sensors are driving the risk score

### Analysis Dashboard
- **Historical Charts**: 24-hour sensor trends
- **Feature Importance**: What the AI learned from the data
- **ROI Calculator**: $200,000 saved per failure prevented

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Recharts** - Interactive charts
- **Lucide Icons** - Modern icons

### Backend
- **FastAPI** - High-performance Python API
- **PyTorch** - LSTM deep learning model
- **Pandas/NumPy** - Data processing
- **Scikit-learn** - Feature scaling

## 📦 Installation

### Prerequisites
- Node.js 18+
- Python 3.9+
- npm or yarn

### Frontend Setup

```bash
cd predictive-dashboard

# Install dependencies
npm install

# Run development server
npm run dev
```

The dashboard will be available at http://localhost:3000

### Backend Setup

```bash
# Install Python dependencies
pip install fastapi uvicorn pandas numpy torch joblib

# Run API server
python api_server.py
```

The API will be available at http://localhost:8000

### With Trained Model

If you have trained LSTM models, place these files in the `predictive-dashboard` folder:
- `lstm_optimized_model.pt` (or `lstm_pytorch_model.pt`)
- `lstm_optimized_scaler.pkl` (or `lstm_scaler.pkl`)
- `lstm_optimized_features.pkl` (or `lstm_features.pkl`)
- `lstm_optimized_config.pkl` (or `lstm_config.pkl`)

The API will automatically load the model on startup.

## 🎮 Usage

### For Jury Demonstration

1. **Start both servers** (frontend and backend)
2. **Navigate to Live Simulation** tab
3. **Use Quick Scenario buttons** to demonstrate different failure conditions:
   - Start with "Normal Operation" - low risk
   - Switch to "Dust Accumulation" - medium risk with dust detection
   - Show "Critical Failure" - immediate action required
4. **Adjust sliders manually** to show real-time prediction updates
5. **Switch to Analysis** tab for historical data and ROI
6. **Show Model Info** tab to explain the LSTM architecture

### Key Scenarios to Demonstrate

| Scenario | What Happens | Expected Risk |
|----------|--------------|---------------|
| Normal | All green, fan spinning slowly | < 30% |
| Dust Accumulation | Vibration rises, dust detected | 30-50% |
| Warning Signs | Multiple sensors elevated | 50-70% |
| Critical | Red alert, fan spinning fast | > 70% |

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Get prediction from sensor data |
| `/api/model-info` | GET | Model specifications |
| `/api/stats` | GET | Dashboard statistics |
| `/api/history` | GET | Historical sensor data |
| `/api/failures` | GET | Failure events |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "motor_current": 25.5,
    "temp_opposite": 72.0,
    "temp_motor": 85.0,
    "vib_opposite": 1.2,
    "vib_motor": 2.8,
    "valve_opening": 88.0
  }'
```

## 🎨 Customization

### OCP Branding
Colors are defined in `tailwind.config.ts`:
- `ocp-green`: #00843D (primary)
- `ocp-gold`: #C4A000 (accent)

### Sensor Thresholds
Edit `src/lib/constants.ts` to adjust:
- Warning/critical thresholds
- Scenario presets
- Model info

## 📱 Responsive Design

The dashboard is fully responsive:
- **Desktop**: Full sidebar with controls
- **Tablet**: Collapsible panels
- **Mobile**: Stacked layout with swipe navigation

## 🔒 Production Deployment

### Frontend (Vercel)
```bash
npm run build
vercel deploy
```

### Backend (Docker)
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install fastapi uvicorn pandas numpy torch joblib
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Recall | 85% |
| Precision | 78% |
| F1-Score | 81% |
| Dust Detection | 92% |
| Parameters | 158,721 |

## 💰 ROI Summary

- **Failures prevented**: 18/22 (85%)
- **Cost per failure**: $200,000
- **Total savings**: $3,600,000
- **Implementation cost**: $90,000
- **ROI**: 4,000%
- **Payback period**: < 1 month

## 👥 Team

Developed for OCP Hackathon 2024

---

**OCP Group** - Predictive Maintenance powered by LSTM Deep Learning
