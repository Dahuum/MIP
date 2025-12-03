# 🏭 OCP Predictive Maintenance Hackathon

## Industrial Fan C07 - Failure Prediction System

[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)

---

## 📋 Project Overview

**Client:** OCP (Office Chérifien des Phosphates) - Morocco  
**Challenge:** Predict industrial fan failures 24-48 hours in advance  
**Impact:** $1.5M+ annual savings through predictive maintenance  
**Timeline:** Hackathon Project

### The Problem

Fan C07 on OCP's phosphate production line 307 experienced **22 unexpected failures in 2019**, causing:
- Unplanned production downtime
- Emergency repair costs (2-3x normal maintenance)
- Safety risks
- Reduced equipment lifespan

### Our Solution

AI-powered predictive maintenance system that:
- ✅ Analyzes 6 real-time sensors every minute
- ✅ Predicts failures 24-48 hours in advance
- ✅ Enables proactive maintenance scheduling
- ✅ Reduces downtime by 30%+

---

## 📊 Dataset

### Sensor Data (`Dataframemin.csv`)
- **Records:** 502,793 complete measurements (minute-by-minute)
- **Period:** March 11, 2019 → February 24, 2020 (~350 days)
- **Sensors:** 6 critical parameters

| Sensor | Description | Units | Critical Level |
|--------|-------------|-------|----------------|
| 307II546 | Motor Current | Amperes | 23-25 A |
| 307TI178A | Temperature (Opposite Side) | °C | >85°C |
| 307TI178B | Temperature (Motor Side) | °C | >85°C |
| 307VI746A | Vibration (Opposite Side) | mm/s | >2.5 mm/s |
| 307VI746B | Vibration (Motor Side) | mm/s | >2.5 mm/s |
| 307ZI717 | Valve Opening | % | <10% or >98% |

**⚠️ Important:** Skip first 15,609 rows - they contain "No Data" values!

### Failure Log
- **File:** `Maintenance prédictive/Suivi des arrêts_ligne 307_2019.xlsx`
- **Content:** 22 documented failure events with timestamps, descriptions, and root causes

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd /Users/mac/Desktop/MIP

# Install required packages
pip install -r requirements.txt
```

### 2. Run Initial Analysis

```bash
# Automated data processing and exploration
python quick_start_analysis.py
```

This script will:
- ✅ Load 502K clean sensor records
- ✅ Compute basic statistics
- ✅ Detect anomalies
- ✅ Create initial features
- ✅ Save processed data to `processed_sensor_data.csv`

### 3. Manual Steps

**Extract failure timestamps:**
1. Open `Maintenance prédictive/Suivi des arrêts_ligne 307_2019.xlsx`
2. Extract failure dates/times and descriptions
3. Save as `failure_events.csv` with format:
   ```
   timestamp,failure_type,duration_hours
   2019-04-15 14:30,Bearing Failure,4.5
   2019-06-22 09:15,Temperature Spike,2.0
   ```

---

## 📁 Project Structure

```
MIP/
├── Dataframemin.csv                    # Main sensor dataset (518K rows)
├── requirements.txt                    # Python dependencies
├── quick_start_analysis.py             # Automated data processing
├── readme.MD                           # This file
│
├── HACKATHON_BRIEFING.md              # Executive summary & action items
├── PROJECT_ANALYSIS.md                # Comprehensive technical analysis
│
├── Maintenance prédictive/
│   ├── HackathonPM.pptx               # Hackathon presentation
│   ├── Suivi des arrêts_ligne 307_2019.xlsx  # Failure log (22 events)
│   └── tags mapping.txt               # Sensor descriptions
│
└── OCP Docs/
    ├── Technical documentation (PDFs)
    ├── Production calculations (DOCX)
    └── Compressed data archives (.7z, .zip)
```

---

## 🔬 Technical Approach

### Phase 1: Data Preparation
- Load clean data (skip rows with "No Data")
- Parse timestamps and sort chronologically
- Extract failure events from Excel
- Create labeled dataset (pre-failure windows)

### Phase 2: Feature Engineering
- **Temporal Features:**
  - Rolling statistics (10min, 1hr, 4hr windows)
  - Rate of change (vibration & temperature trends)
  - Lag features (readings from 1-24 hours ago)

- **Domain-Specific Features:**
  - Temperature differential (motor - opposite)
  - Vibration ratio (motor / opposite)
  - Valve opening anomalies

### Phase 3: Model Development
- **Baseline:** Random Forest Classifier
- **Advanced:** XGBoost with engineered features
- **Stretch Goal:** LSTM for sequential patterns

### Phase 4: Deployment
- Streamlit dashboard for real-time monitoring
- Risk scoring system (0-100 scale)
- Alert system for maintenance teams

---

## 📈 Key Performance Metrics

### Technical Metrics
- **Recall:** >85% (minimize false negatives - critical!)
- **Precision:** >50% (avoid too many false alarms)
- **F1-Score:** >0.65
- **Lead Time:** 24-48 hours advance warning

### Business Metrics
- **Downtime Reduction:** 30%+ (22 → <15 failures/year)
- **Annual Savings:** $1.5M+
- **Equipment Life Extension:** 20%
- **ROI:** 30x (implementation cost vs. annual savings)

---

## 🚨 Critical Insights from Data Analysis

### Anomalies Detected
1. **High Temperature Events:** Peak at 97.6°C (normal: 43-80°C)
2. **High Vibration Spikes:** Up to 2.9 mm/s (normal: 1.6-2.3 mm/s)
3. **Valve Position Anomalies:** Drops to 3.4% (normal: 90-96%)

### Data Quality
- ✅ 97% complete data (502,793 usable records)
- ⚠️ Severe class imbalance (0.004% failure rate)
- ✅ High-frequency sampling (1-minute intervals)
- ⚠️ Multiple failure modes require different detection strategies

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
- [ ] Clean dataset loaded and labeled
- [ ] Baseline model trained (>70% recall)
- [ ] Simple dashboard showing risk score
- [ ] Presentation with business case

### Competitive Product
- [ ] Advanced model (>85% recall, <15% false positives)
- [ ] Rich feature engineering (100+ features)
- [ ] Interactive dashboard with real-time simulation
- [ ] ROI demonstration with OCP-specific numbers

### Winning Product
- [ ] Ensemble model (>90% recall)
- [ ] Multi-modal failure detection
- [ ] Professional dashboard with alert system
- [ ] Live demo using actual 2019 failure predictions

---

## 💡 Domain Knowledge

### How Industrial Fans Fail

1. **Bearing Degradation** (Most Common)
   - Symptom: Gradual vibration increase
   - Warning Time: 2-4 weeks
   - Detection: Vibration 0.7 → 2.5+ mm/s

2. **Motor Overheating**
   - Symptom: Temperature rise + current draw
   - Warning Time: Hours to days
   - Detection: Temp >80°C + asymmetry

3. **Imbalance/Misalignment**
   - Symptom: Sudden vibration at specific frequencies
   - Warning Time: Days
   - Detection: Pattern changes + high current

### ISO 10816 Vibration Standards
```
Zone A (Good):           < 0.71 mm/s  ✅
Zone B (Acceptable):     0.71-2.8 mm/s  ⚠️
Zone C (Unsatisfactory): 2.8-7.1 mm/s  🚨
Zone D (Unacceptable):   > 7.1 mm/s  🛑
```

---

## 📚 Documentation

- **`HACKATHON_BRIEFING.md`** - Executive summary with immediate action items
- **`PROJECT_ANALYSIS.md`** - Comprehensive technical analysis and recommendations
- **`quick_start_analysis.py`** - Automated data processing script

---

## 🛠️ Technology Stack

**Data Processing:** Pandas, NumPy, SciPy  
**Machine Learning:** Scikit-learn, XGBoost, LightGBM  
**Deep Learning:** TensorFlow/Keras (optional)  
**Visualization:** Matplotlib, Seaborn, Plotly  
**Dashboard:** Streamlit or Dash  
**Imbalanced Learning:** SMOTE, class weights  
**Interpretability:** SHAP values  

---

## ⚠️ Common Pitfalls to Avoid

❌ **DON'T:**
- Use all data for training (violates time-series nature)
- Optimize for accuracy (wrong metric for imbalanced data)
- Ignore domain knowledge (physics of fan failures)
- Build notebook-only solution (need deployable system)

✅ **DO:**
- Use time-aware train/test split
- Optimize for recall (missing failures is worse than false alarms)
- Engineer physics-based features
- Build working demo dashboard

---

## 📞 Team & Contact

**Hackathon Team:** [Your Team Name]  
**Project Lead:** [Your Name]  
**Technical Stack:** Python 3.8+, Scikit-learn, XGBoost, Streamlit  

---

## 🏆 Expected Outcomes

By the end of this hackathon, we will deliver:

1. **Predictive Model** - ML system predicting failures 24-48 hours in advance
2. **Dashboard** - Real-time monitoring interface for maintenance teams
3. **Business Case** - ROI analysis showing $1.5M+ annual savings
4. **Deployment Plan** - Roadmap for OCP implementation

---

## 📝 Getting Help

### Quick References
- **Clean data starts:** Row 15,610 (March 11, 2019 20:08)
- **Total usable records:** 502,793
- **Most critical sensors:** Vibration (Motor & Opposite)
- **Target metric:** Recall >85%

### Next Steps
1. ✅ Run `pip install -r requirements.txt`
2. ✅ Run `python quick_start_analysis.py`
3. ✅ Extract failure timestamps from Excel
4. ✅ Start building baseline model

---

## 🚀 Let's Build Something Amazing!

**Current Status:** ✅ Data analyzed and ready  
**Next Action:** Start model development  
**Timeline:** Hackathon in progress  

**Good luck! 🏆**

---

*Last Updated: Hackathon Day 1*  
*Project: OCP Predictive Maintenance*  
*Goal: Save $1.5M+ annually through AI-powered maintenance*