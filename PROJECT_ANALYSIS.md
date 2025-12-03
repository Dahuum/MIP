# 🏭 OCP Predictive Maintenance Hackathon - Complete Project Analysis

## 📋 Executive Summary

**Project**: Predictive Maintenance for Industrial Fan C07 (Line 307)  
**Client**: OCP (Office Chérifien des Phosphates) - Morocco's phosphate processing company  
**Objective**: Build ML model to predict fan failures 24-48 hours in advance  
**Data Period**: March 11, 2019 20:08 → February 24, 2020 00:00  
**Total Records**: 502,793 complete sensor readings (minute-by-minute)

---

## 🎯 Core Problem Statement

### Current Situation
- **Equipment**: Industrial ventilation Fan C07 on production line 307
- **Industry**: Phosphate and phosphoric acid manufacturing
- **Failure Mode**: Reactive maintenance causing:
  - Unplanned downtime (22 documented failures in 2019)
  - High emergency repair costs
  - Safety risks
  - Reduced equipment lifespan

### Target Solution
Build predictive model to:
- Predict failures 24-48 hours in advance
- Enable proactive maintenance scheduling
- Reduce downtime by 30%+
- Cut maintenance costs by 25%+
- Extend fan lifespan by 20%+

---

## 📊 Data Inventory & Analysis

### 1. Sensor Data (`Dataframemin.csv`)

#### Dataset Statistics
```
Total Rows: 518,402
- Header: 1 row
- Incomplete data (with "No Data"): 15,608 rows (March 1-11, 2019)
- Complete usable data: 502,793 rows
- Data Quality: 97% complete

Time Range: March 11, 2019 20:08 → February 24, 2020 00:00
Duration: ~350 days
Frequency: 1-minute intervals
```

#### Column Mapping (Sensor Tags)

| CSV Column | Tag Code  | Sensor Description              | Units | Critical? |
|------------|-----------|----------------------------------|-------|-----------|
| Date       | -         | Timestamp                        | -     | ✅        |
| 307II546   | -         | **Ampérage du moteur** (Motor Current) | A | ✅ |
| 307TI178A  | -         | **Température côté opposé** (Temp Opposite) | °C | ✅ |
| 307TI178B  | -         | **Température côté moteur** (Temp Motor Side) | °C | ✅ |
| 307VI746A  | -         | **Vibration côté opposé** (Vibration Opposite) | mm/s | ✅✅ |
| 307VI746B  | -         | **Vibration côté moteur** (Vibration Motor Side) | mm/s | ✅✅ |
| 307ZI717   | -         | **Ouverture volets** (Valve Opening) | % | ⚠️ |

**Note**: Vibration sensors are MOST CRITICAL for failure prediction based on industrial standards.

#### Sample Data (Complete Records)
```csv
Date,Motor Current,Temp Opposite,Temp Motor,Vib Opposite,Vib Motor,Valve Opening
11/03/2019 20:08,23.31,50.55,55.84,0.66,2.30,90.29
11/03/2019 20:09,23.28,50.59,55.85,0.69,2.32,90.28
23/02/2020 23:59,23.72,43.98,34.57,0.77,1.74,96.23
24/02/2020 00:00,23.76,43.99,34.56,0.77,1.73,96.24
```

#### Data Quality Observations

**✅ Strengths:**
- High-frequency sampling (1-minute intervals)
- 11+ months of continuous data
- All critical parameters captured
- Minimal missing values in usable range

**⚠️ Challenges:**
- First 10 days contain "No Data" values (must skip rows 2-15609)
- Need to handle potential outliers
- Imbalanced dataset (22 failures vs 500K+ normal operations = 0.004%)
- Multiple failure modes require different detection strategies

#### Sensor Value Ranges (Observed)

```
Motor Current (307II546):     ~23-25 A (normal operation)
Temp Opposite (307TI178A):    ~43-93°C (wide range)
Temp Motor Side (307TI178B):  ~34-98°C (wide range)
Vibration Opposite (307VI746A): ~0.06-1.1 mm/s
Vibration Motor Side (307VI746B): ~0.13-2.9 mm/s (higher than opposite)
Valve Opening (307ZI717):     ~3.4-96% (mostly 90-96% in normal operation)
```

**🚨 Anomaly Indicators Identified:**
- Extreme valve positions (<10% or sudden drops)
- High vibration readings (>2.5 mm/s on motor side)
- Temperature spikes (>90°C)
- Temperature asymmetry (motor side >> opposite side)

---

### 2. Failure Log Data (`Suivi des arrêts_ligne 307_2019.xlsx`)

#### Contents
- **Purpose**: Historical record of all production line 307 stoppages in 2019
- **Expected Data**:
  - Failure timestamps
  - Failure descriptions
  - Downtime duration
  - Failure types (mechanical, electrical, vibration, temperature)
  - Root cause analysis

#### Key Statistics (from briefing)
- **Total Failures**: 22 documented events for Fan C07
- **Common Failure Modes**:
  1. Bearing failures (vibration-related)
  2. Temperature-related issues
  3. Motor electrical faults
  4. Mechanical wear

**⚠️ Note**: Excel file requires Python with openpyxl/xlrd to read programmatically.

---

### 3. Documentation Files (`OCP Docs/`)

#### Available Reference Materials

1. **`886801764347850_BECKER - Phosphates and Phosphoric Acid (1).pdf`**
   - Technical documentation on phosphate processing
   - Context for industrial environment
   - Equipment specifications

2. **`How Generative AI Improves Supply Chain Management 1 (1).pdf`**
   - AI/ML approaches for industrial optimization
   - Potential solution architectures

3. **`Use AI to Stress Test Your Supply Chain (1).PDF`**
   - Predictive analytics methodologies
   - Risk assessment frameworks

4. **`Equation de calcul de la production journalière.docx`**
   - Daily production calculation formulas
   - Business impact metrics

5. **Compressed Archives** (`.7z`, `.zip` files):
   - `Dataframemin.7z` / `Dataframemin1.7z` - Backup data files
   - `LD_PI_05.zip`, `LF_PI_05.zip`, `LE_PI_05.zip`, `LABO_05.zip` - Additional sensor data or lab reports

---

## 🔬 Technical Analysis & Recommendations

### Phase 1: Data Preparation (PRIORITY 1)

#### Immediate Actions:
1. **Load clean data starting from row 15610** (skip "No Data" rows)
2. **Parse timestamps** to datetime objects
3. **Feature engineering**:
   - Calculate rolling statistics (mean, std, min, max) over windows:
     - 10-minute window
     - 1-hour window
     - 4-hour window
   - Rate of change for vibration & temperature
   - Temperature differential (motor side - opposite side)
   - Vibration ratio (motor side / opposite side)

4. **Label creation**:
   - Parse failure log Excel
   - Create binary labels (0=normal, 1=pre-failure)
   - Label X hours before each failure event (where X=24-48)

#### Code Priority:
```python
# 1. Load data (skip first 15609 rows)
df = pd.read_csv('Dataframemin.csv', skiprows=range(1, 15610))

# 2. Parse timestamps
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

# 3. Sort by date
df = df.sort_values('Date').reset_index(drop=True)

# 4. Feature engineering
# - Rolling windows
# - Rate of change
# - Statistical features
```

---

### Phase 2: Exploratory Data Analysis (PRIORITY 2)

#### Key Questions to Answer:
1. **What do normal operations look like?**
   - Baseline sensor ranges
   - Daily/weekly patterns
   - Correlation between sensors

2. **What patterns precede failures?**
   - Vibration increase trends
   - Temperature escalation
   - Current fluctuations
   - Time-to-failure signatures

3. **Are there distinct failure modes?**
   - Vibration-dominated failures
   - Temperature-dominated failures
   - Combined mode failures

#### Visualization Priorities:
- Time series plots of all sensors
- Correlation heatmap
- Distribution plots for each sensor
- Anomaly detection visualization
- Pre-failure behavior analysis (24-48 hours before events)

---

### Phase 3: Model Development (PRIORITY 3)

#### Strategy: Multi-Stage Approach

**Stage 1: Baseline Model (Quick Win)**
```
Algorithm: Random Forest Classifier
Features: Raw sensor values + simple rolling means
Target: Binary classification (failure within 48h: yes/no)
Goal: Establish baseline performance (~70-80% accuracy target)
```

**Stage 2: Advanced Model**
```
Algorithm: XGBoost or LightGBM
Features: Engineered features (100+ features)
  - Rolling statistics (10m, 1h, 4h windows)
  - Rate of change
  - FFT features (frequency domain)
  - Lag features
  - Interaction terms
Target: Multi-class (normal / pre-failure-24h / pre-failure-48h)
Goal: >85% recall for failure detection
```

**Stage 3: Time-Series Deep Learning (If Time Permits)**
```
Algorithm: LSTM or 1D-CNN
Input: Sequence of last 4 hours of sensor readings
Output: Probability of failure in next 24-48 hours
Goal: Capture temporal patterns
```

#### Handling Class Imbalance:
- **SMOTE** (Synthetic Minority Over-sampling)
- **Class weights** in model training
- **Threshold tuning** (optimize for recall over precision)
- **Ensemble methods** (multiple models voting)

#### Model Evaluation Metrics:
```
Primary Metrics:
- Recall (minimize false negatives - critical!)
- F1-Score
- Precision-Recall curve

Secondary Metrics:
- ROC-AUC
- Confusion matrix
- Lead time accuracy (how far in advance was failure predicted)
```

---

### Phase 4: Dashboard & Deployment (PRIORITY 4)

#### Dashboard Requirements:

**Real-Time Monitoring View:**
- Current sensor readings (gauges)
- Risk score (0-100 scale)
- Alert status (green/yellow/red)
- Time to predicted failure

**Historical Analysis View:**
- Sensor trends over last 7 days
- Failure event timeline
- Model prediction history
- Accuracy metrics

**Alert System:**
- Email/SMS when risk >70%
- Maintenance recommendation
- Predicted failure time window

#### Technology Stack Recommendation:
```
Backend: Python (Flask/FastAPI)
Frontend: Streamlit or Dash (quick development)
Database: PostgreSQL for historical data
Real-time: MQTT or REST API for sensor ingestion
Visualization: Plotly
```

---

## 📈 Business Impact Metrics

### ROI Calculation Framework

**Current Costs (Reactive Maintenance):**
- 22 unplanned failures per year
- Average downtime: 4-8 hours per failure
- Downtime cost: ~$10,000-50,000 per hour (estimate for phosphate production)
- Emergency repair cost: 2-3x normal maintenance cost

**Projected Savings (Predictive Maintenance):**
- Reduce unplanned failures by 70% (15 failures prevented)
- Convert to planned maintenance (lower cost)
- Reduce average repair time by 50%
- Extend equipment life by 20%

**Estimated Annual Savings:** $500K - $2M+

---

## ⚠️ Critical Success Factors

### Must-Have Features:
1. ✅ **High recall** (>90%) - Cannot miss real failures
2. ✅ **Sufficient lead time** (24-48 hours) - Time to schedule maintenance
3. ✅ **Low false positive rate** (<20%) - Avoid alert fatigue
4. ✅ **Explainable predictions** - Show which sensors triggered alert
5. ✅ **Simple dashboard** - Maintenance team can understand and act

### Risk Mitigation:
- **Imbalanced data**: Use SMOTE, class weights, careful validation
- **Overfitting**: Cross-validation, regularization, ensemble methods
- **Data quality**: Outlier detection, missing value handling
- **Interpretability**: Feature importance, SHAP values

---

## 🚀 Recommended Development Timeline

### Hour 0-4: Foundation
- ✅ Load and clean data
- ✅ Parse failure log Excel
- ✅ Create labeled dataset
- ✅ Basic EDA and visualizations

### Hour 4-8: Feature Engineering
- ✅ Calculate rolling statistics
- ✅ Engineer domain-specific features
- ✅ Create train/validation/test splits
- ✅ Build baseline Random Forest model

### Hour 8-12: Model Optimization
- ✅ Train XGBoost/LightGBM
- ✅ Hyperparameter tuning
- ✅ Handle class imbalance
- ✅ Evaluate on test set

### Hour 12-16: Dashboard Development
- ✅ Build Streamlit dashboard
- ✅ Real-time risk scoring
- ✅ Alert system
- ✅ Visualization of predictions

### Hour 16-20: Polish & Presentation
- ✅ Calculate business impact metrics
- ✅ Create presentation deck
- ✅ Rehearse demo
- ✅ Final testing

---

## 🎯 Winning Strategy

### Technical Excellence:
1. **Start simple, iterate fast** - Don't over-engineer
2. **Focus on vibration sensors** - Most predictive for mechanical failures
3. **Engineer temporal features** - Rate of change matters more than absolute values
4. **Use ensemble methods** - Combine multiple models

### Business Focus:
1. **Emphasize cost savings** - Quantify ROI clearly
2. **Show real failure predictions** - Use actual 2019 failure events in demo
3. **Demonstrate ease of use** - Simple, actionable dashboard
4. **Explain the "why"** - Feature importance and interpretability

### Presentation Tips:
1. **Tell a story**: "In April 2019, Fan C07 failed unexpectedly. Our model would have predicted this 36 hours in advance..."
2. **Show the money**: "$1.5M annual savings vs $50K implementation cost = 30x ROI"
3. **Live demo**: Real-time risk score updating with simulated sensor data
4. **Acknowledge limitations**: "Currently 85% recall, roadmap to 95%"

---

## 📝 Next Steps (YOUR ACTION ITEMS)

### Immediate (Next 30 minutes):
1. ✅ Install required Python packages:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit openpyxl
   ```

2. ✅ Extract failure log Excel data:
   - Open `Suivi des arrêts_ligne 307_2019.xlsx`
   - Document failure timestamps
   - Categorize failure types

3. ✅ Create project structure:
   ```
   MIP/
   ├── data/           # Raw data files
   ├── notebooks/      # Jupyter notebooks for EDA
   ├── src/            # Python source code
   │   ├── data_processing.py
   │   ├── feature_engineering.py
   │   ├── model.py
   │   └── dashboard.py
   ├── models/         # Trained model files
   └── outputs/        # Results, plots, reports
   ```

### Short-term (Next 2-4 hours):
1. Load data and perform EDA
2. Parse failure events from Excel
3. Create labeled dataset
4. Build baseline model

### Medium-term (Next 4-8 hours):
1. Feature engineering pipeline
2. Advanced model training
3. Model evaluation and tuning

### Final stretch (Last 8-12 hours):
1. Dashboard development
2. Business case documentation
3. Presentation preparation
4. Testing and refinement

---

## 🏆 Competitive Advantages

Your project will stand out if you:
1. **Demonstrate real predictions** on actual 2019 failures
2. **Show interpretability** - explain why model predicts failure
3. **Quantify business impact** with real numbers
4. **Build working dashboard** - not just a notebook
5. **Consider deployment** - discuss how OCP would actually use this

---

## 📚 Key References

### Domain Knowledge:
- Fan vibration standards: ISO 10816
- Typical bearing failure progression: Vibration increases 2-4 weeks before failure
- Motor temperature limits: Usually <80°C for continuous operation

### ML Approaches:
- Time series classification
- Imbalanced learning (SMOTE, class weights)
- Feature importance for interpretability
- Ensemble methods for robustness

### Similar Projects:
- Predictive maintenance in manufacturing
- Anomaly detection in industrial IoT
- Time series forecasting for equipment health

---

## ✅ Success Criteria Checklist

### Minimum Viable Product (MVP):
- [ ] Clean dataset loaded (502K rows)
- [ ] Failure events labeled
- [ ] Baseline model trained (>70% recall)
- [ ] Basic dashboard showing risk score
- [ ] Presentation deck with business case

### Competitive Product:
- [ ] Advanced model trained (>85% recall, <15% false positive)
- [ ] Rich feature engineering (temporal, statistical, domain-specific)
- [ ] Interactive dashboard with real-time simulation
- [ ] Quantified ROI with OCP-specific numbers
- [ ] Model interpretability (feature importance, SHAP)

### Winning Product:
- [ ] Ensemble model (>90% recall)
- [ ] Multi-modal failure detection (vibration, temperature, combined)
- [ ] Professional dashboard with alert system
- [ ] Comprehensive business case with deployment plan
- [ ] Live demo showing actual 2019 failure predictions

---

## 🎤 Pitch Template

### Opening (30 seconds):
"In 2019, OCP's phosphate production line 307 experienced 22 unexpected fan failures, costing millions in lost production and emergency repairs. We built an AI system that would have predicted 19 of those failures with 36-48 hours advance notice."

### Problem (1 minute):
"Fan C07 is critical to phosphate processing. When it fails unexpectedly, the entire line shuts down. Current reactive maintenance means high costs and safety risks."

### Solution (2 minutes):
"Our predictive maintenance system analyzes 6 real-time sensors every minute. Using machine learning on 500,000 historical data points, we detect early warning signs invisible to human operators."

[SHOW DASHBOARD WITH LIVE DEMO]

### Business Impact (1 minute):
"For OCP, this means:
- 70% reduction in unplanned downtime
- $1.5M annual savings
- 20% longer equipment life
- Zero safety incidents from unexpected failures"

### Closing (30 seconds):
"This isn't just a hackathon project. It's a production-ready system that can be deployed across OCP's 50+ production lines, multiplying the impact."

---

**Document Version:** 1.0  
**Last Updated:** Hackathon Day 1  
**Status:** Ready for Development  

**Next Action:** Load the failure log Excel and start coding! 🚀