'use client';

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { OCPLogo, FanIcon } from '@/components/icons/OCPLogo';
import { RiskGauge } from '@/components/visualizations/RiskGauge';
import { SensorControls } from '@/components/controls/SensorControls';
import { ScenarioSelector, QuickScenarioBar } from '@/components/controls/ScenarioSelector';
import { AlertPanel, AlertBanner } from '@/components/alerts/AlertPanel';
import { SensorChart, RiskHistoryChart } from '@/components/visualizations/Charts';
import { FeatureImportance } from '@/components/visualizations/FeatureImportance';
import { DashboardStats, ROICalculator } from '@/components/stats/StatsCard';
import { SensorData, PredictionResult, ScenarioPreset } from '@/types';
import { DEFAULT_SENSORS, SCENARIO_PRESETS, RISK_LEVELS, SENSOR_THRESHOLDS, MODEL_INFO } from '@/lib/constants';
import { getRiskLevel, formatCurrency, generateTimeLabels, addNoise } from '@/lib/utils';

const API_URL = 'http://localhost:8000';

// Types for tracking sensor interactions
interface SensorInteraction {
  timestamp: Date;
  sensor: keyof SensorData;
  oldValue: number;
  newValue: number;
  riskBefore: number;
  riskAfter: number;
  riskDelta: number;
}

interface SessionStats {
  totalInteractions: number;
  highRiskEvents: number;
  criticalAlerts: number;
  maxRiskReached: number;
  sessionStartTime: Date;
  testsPerformed: number;
}

interface DynamicFeatureImportance {
  sensor: keyof SensorData;
  name: string;
  totalRiskImpact: number;
  interactions: number;
  avgRiskDelta: number;
  maxRiskDelta: number;
  category: 'vibration' | 'temperature' | 'current' | 'flow' | 'valve';
}

interface HistoricFailure {
  timestamp: string;
  date: string;
  time: string;
  failure_type: string;
  duration: string;
  description: string;
  severity: string;
  source: 'historical' | 'session';
  sensors?: SensorData;
}

export default function Dashboard() {
  // State
  const [sensors, setSensors] = useState<SensorData>(DEFAULT_SENSORS);
  const [currentScenario, setCurrentScenario] = useState<string | null>('normal');
  const [activeTab, setActiveTab] = useState<'live' | 'analysis' | 'model'>('live');
  const [isSimulating, setIsSimulating] = useState(false);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // NEW: Real-time tracking state
  const [sensorHistory, setSensorHistory] = useState<SensorInteraction[]>([]);
  const [sessionStats, setSessionStats] = useState<SessionStats>({
    totalInteractions: 0,
    highRiskEvents: 0,
    criticalAlerts: 0,
    maxRiskReached: 0,
    sessionStartTime: new Date(),
    testsPerformed: 0,
  });
  const [riskHistory, setRiskHistory] = useState<{time: string; risk: number; sensors: SensorData}[]>([]);
  const prevSensorsRef = React.useRef<SensorData>(DEFAULT_SENSORS);
  const prevRiskRef = React.useRef<number>(0);
  
  // Historic failures state
  const [historicFailures, setHistoricFailures] = useState<HistoricFailure[]>([]);
  const [sessionCriticalErrors, setSessionCriticalErrors] = useState<HistoricFailure[]>([]);
  const lastCriticalRef = React.useRef<number>(0); // Prevent duplicate critical alerts

  // Fetch prediction from API whenever sensors change
  useEffect(() => {
    const fetchPrediction = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(sensors),
        });
        if (response.ok) {
          const data = await response.json();
          setPrediction(data);
        }
      } catch (error) {
        console.error('API Error:', error);
        // Fallback to client-side calculation if API fails
        setPrediction(calculateLocalPrediction(sensors));
      }
      setIsLoading(false);
    };
    fetchPrediction();
  }, [sensors]);

  // Fetch historical failures AND saved session data on mount
  useEffect(() => {
    const fetchInitialData = async () => {
      // Fetch historical failures
      try {
        const response = await fetch(`${API_URL}/api/failures`);
        if (response.ok) {
          const data = await response.json();
          setHistoricFailures(data.failures || []);
        }
      } catch (error) {
        console.error('Failed to fetch historic failures:', error);
      }
      
      // Fetch saved session data (persisted across refresh)
      try {
        const response = await fetch(`${API_URL}/api/session`);
        if (response.ok) {
          const result = await response.json();
          if (result.success && result.data) {
            const data = result.data;
            // Restore session critical errors
            if (data.sessionCriticalErrors?.length > 0) {
              setSessionCriticalErrors(data.sessionCriticalErrors);
            }
            // Restore session stats
            if (data.sessionStats) {
              setSessionStats(prev => ({
                ...prev,
                ...data.sessionStats,
                sessionStartTime: new Date(data.sessionStats.sessionStartTime || Date.now()),
              }));
            }
            // Restore risk history
            if (data.riskHistory?.length > 0) {
              setRiskHistory(data.riskHistory);
            }
            // Restore sensor values
            if (data.currentSensors) {
              setSensors(data.currentSensors);
              console.log('🎛️ Sensor values restored:', data.currentSensors);
            }
            console.log('📂 Session data restored:', {
              criticalErrors: data.sessionCriticalErrors?.length || 0,
              trainingPoints: result.training_points,
              sensorsRestored: !!data.currentSensors
            });
          }
        }
      } catch (error) {
        console.error('Failed to fetch session data:', error);
      }
    };
    fetchInitialData();
  }, []);

  // NEW: Track sensor changes and their risk impact
  useEffect(() => {
    if (!prediction) return;
    
    const currentRisk = prediction.risk_probability * 100;
    const prevSensors = prevSensorsRef.current;
    const prevRisk = prevRiskRef.current;
    
    // Find which sensor changed
    const sensorKeys = Object.keys(sensors) as Array<keyof SensorData>;
    for (const key of sensorKeys) {
      if (sensors[key] !== prevSensors[key]) {
        const interaction: SensorInteraction = {
          timestamp: new Date(),
          sensor: key,
          oldValue: prevSensors[key],
          newValue: sensors[key],
          riskBefore: prevRisk,
          riskAfter: currentRisk,
          riskDelta: currentRisk - prevRisk,
        };
        
        setSensorHistory(prev => [...prev.slice(-99), interaction]); // Keep last 100
        
        // Update session stats
        setSessionStats(prev => ({
          ...prev,
          totalInteractions: prev.totalInteractions + 1,
          highRiskEvents: currentRisk >= 50 ? prev.highRiskEvents + 1 : prev.highRiskEvents,
          criticalAlerts: currentRisk >= 70 ? prev.criticalAlerts + 1 : prev.criticalAlerts,
          maxRiskReached: Math.max(prev.maxRiskReached, currentRisk),
          testsPerformed: prev.testsPerformed + 1,
        }));
      }
    }
    
    // Record critical error when risk crosses into critical zone (>= 70%)
    const now = Date.now();
    const wasCritical = prevRisk >= 70;
    const isCritical = currentRisk >= 70;
    
    // Record when: entering critical zone OR already critical and 10+ seconds passed
    if (isCritical && (!wasCritical || now - lastCriticalRef.current > 10000)) {
      lastCriticalRef.current = now;
      
      // Determine failure type from prediction
      let failureType = prediction.issue_type?.toUpperCase() || 'CRITICAL';
      if (failureType === 'NONE') failureType = 'CRITICAL';
      
      const criticalError = {
        timestamp: new Date().toISOString(),
        date: new Date().toLocaleDateString('en-CA'),
        time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        failure_type: failureType,
        duration: 'Active',
        description: prediction.diagnostic_message || `Critical risk level: ${currentRisk.toFixed(1)}%`,
        severity: 'critical' as const,
        source: 'session' as const,
        sensors: {...sensors},
      };
      
      setSessionCriticalErrors(prev => [...prev, criticalError]);
      console.log('🚨 Critical error recorded:', criticalError);
      
      // Save to backend for persistence and model training
      fetch(`${API_URL}/api/session/critical`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(criticalError),
      }).then(res => res.json()).then(result => {
        console.log('💾 Saved to backend:', result);
      }).catch(err => {
        console.error('Failed to save to backend:', err);
      });
    }
    
    // Add to risk history for charts
    const timeStr = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setRiskHistory(prev => [...prev.slice(-59), { time: timeStr, risk: currentRisk, sensors: {...sensors} }]);
    
    // Update refs
    prevSensorsRef.current = {...sensors};
    prevRiskRef.current = currentRisk;
  }, [prediction, sensors]);

  // Save sensor values to backend (debounced)
  const saveSensorsTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);
  useEffect(() => {
    // Debounce: wait 1 second after last change before saving
    if (saveSensorsTimeoutRef.current) {
      clearTimeout(saveSensorsTimeoutRef.current);
    }
    saveSensorsTimeoutRef.current = setTimeout(() => {
      fetch(`${API_URL}/api/session/sensors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sensors),
      }).then(res => res.json()).then(result => {
        if (result.success) {
          console.log('💾 Sensors saved to backend');
        }
      }).catch(err => {
        console.error('Failed to save sensors:', err);
      });
    }, 1000);
    
    return () => {
      if (saveSensorsTimeoutRef.current) {
        clearTimeout(saveSensorsTimeoutRef.current);
      }
    };
  }, [sensors]);

  // Calculate dynamic feature importance from interaction history
  const dynamicFeatureImportance = useMemo((): DynamicFeatureImportance[] => {
    const sensorNames: Record<keyof SensorData, { name: string; category: 'vibration' | 'temperature' | 'current' | 'flow' | 'valve' }> = {
      motor_current: { name: 'Motor Current', category: 'current' },
      temp_opposite: { name: 'Temperature (Opposite)', category: 'temperature' },
      temp_motor: { name: 'Temperature (Motor)', category: 'temperature' },
      vib_opposite: { name: 'Vibration (Opposite)', category: 'vibration' },
      vib_motor: { name: 'Vibration (Motor)', category: 'vibration' },
      valve_opening: { name: 'Valve Opening', category: 'valve' },
      solid_rate: { name: 'Solid Rate', category: 'flow' },
      pump_flow_rate: { name: 'Pump Flow Rate', category: 'flow' },
    };

    const impactBySensor: Record<string, { totalImpact: number; count: number; maxDelta: number }> = {};
    
    for (const interaction of sensorHistory) {
      const key = interaction.sensor;
      if (!impactBySensor[key]) {
        impactBySensor[key] = { totalImpact: 0, count: 0, maxDelta: 0 };
      }
      impactBySensor[key].totalImpact += Math.abs(interaction.riskDelta);
      impactBySensor[key].count += 1;
      impactBySensor[key].maxDelta = Math.max(impactBySensor[key].maxDelta, Math.abs(interaction.riskDelta));
    }

    const features: DynamicFeatureImportance[] = (Object.keys(sensorNames) as Array<keyof SensorData>).map(sensor => {
      const data = impactBySensor[sensor] || { totalImpact: 0, count: 0, maxDelta: 0 };
      return {
        sensor,
        name: sensorNames[sensor].name,
        totalRiskImpact: data.totalImpact,
        interactions: data.count,
        avgRiskDelta: data.count > 0 ? data.totalImpact / data.count : 0,
        maxRiskDelta: data.maxDelta,
        category: sensorNames[sensor].category,
      };
    });

    // Sort by total risk impact (descending)
    return features.sort((a, b) => b.totalRiskImpact - a.totalRiskImpact);
  }, [sensorHistory]);

  // Fallback local calculation (only used if API is down)
  const calculateLocalPrediction = (sensors: SensorData): PredictionResult => {
    const vibRatio = sensors.vib_motor / (sensors.vib_opposite + 0.001);
    const tempDiff = sensors.temp_motor - sensors.temp_opposite;

    // Calibrated risk factors
    const vibRisk = Math.min(Math.max(0, (sensors.vib_motor - 1.5) / 2.0), 1);
    const tempRisk = Math.min(Math.max(0, (sensors.temp_motor - 60) / 40), 1);
    const currentRisk = Math.min(Math.abs(sensors.motor_current - 24.5) / 4, 1);
    const vibRatioRisk = Math.min(Math.max(0, (vibRatio - 1.5) / 2.5), 1);

    let riskProbability = Math.min(
      vibRisk * 0.30 + tempRisk * 0.25 + vibRatioRisk * 0.15 + currentRisk * 0.10 + 
      (tempDiff > 30 ? 0.15 : Math.max(0, (tempDiff - 5) / 166)),
      1
    );

    // CRITICAL SENSOR BOOST - Ensure critical sensors result in high risk
    let criticalBoost = 0;
    if (sensors.temp_motor > 85) criticalBoost = Math.max(criticalBoost, 0.50);
    else if (sensors.temp_motor > 70) criticalBoost = Math.max(criticalBoost, 0.30);
    
    if (sensors.vib_motor > 3.0) criticalBoost = Math.max(criticalBoost, 0.50);
    else if (sensors.vib_motor > 2.0) criticalBoost = Math.max(criticalBoost, 0.30);
    
    if (sensors.motor_current > 28) criticalBoost = Math.max(criticalBoost, 0.50);
    else if (sensors.motor_current > 25) criticalBoost = Math.max(criticalBoost, 0.30);
    
    if (sensors.valve_opening < 20) criticalBoost = Math.max(criticalBoost, 0.40);
    
    if (sensors.solid_rate > 2.0) criticalBoost = Math.max(criticalBoost, 0.35);
    else if (sensors.solid_rate > 1.0) criticalBoost = Math.max(criticalBoost, 0.20);
    
    if (sensors.pump_flow_rate < 300) criticalBoost = Math.max(criticalBoost, 0.35);
    else if (sensors.pump_flow_rate < 400) criticalBoost = Math.max(criticalBoost, 0.20);
    
    riskProbability = Math.min(riskProbability + criticalBoost, 1.0);

    // Define normal thresholds
    const currentNormal = sensors.motor_current <= 25.0;
    const tempNormal = sensors.temp_motor <= 70.0;
    const vibNormal = sensors.vib_motor <= 2.0;
    const valveNormal = sensors.valve_opening > 20.0;
    const solidNormal = sensors.solid_rate <= 1.0;
    const flowNormal = sensors.pump_flow_rate >= 400;

    // Abnormal flags
    const currentHigh = sensors.motor_current > 25.0;
    const tempHigh = sensors.temp_motor > 70.0;
    const vibHigh = sensors.vib_motor > 2.0;
    const valveLow = sensors.valve_opening < 20.0;
    const currentLow = sensors.motor_current < 15.0;
    const solidHigh = sensors.solid_rate > 1.0;
    const flowLow = sensors.pump_flow_rate < 400;

    // Auto-adjust solid rate based on flow
    let effectiveSolidRate = sensors.solid_rate;
    if (flowLow && sensors.solid_rate < 1.0) {
      const flowFactor = (400 - sensors.pump_flow_rate) / 200;
      effectiveSolidRate = Math.min(sensors.solid_rate + flowFactor * 1.5, 3.0);
    }

    // Initialize diagnostics
    type IssueType = 'none' | 'dust_accumulation' | 'grease_check' | 'both' | 'electrical' | 'overheating' | 'bearing' | 'bearing_axle' | 'power_loss';
    let issueType: IssueType = 'none';
    let diagnosticMessage = '';
    let dustProbability = 0.05;

    // DIAGNOSTIC RULES (Priority Order)
    if (valveLow && currentLow) {
      issueType = 'power_loss';
      diagnosticMessage = '⚡ POWER LOSS: Check power distribution system';
      dustProbability = 0.0;
    } else if (currentHigh && tempNormal && vibNormal && valveNormal) {
      issueType = 'electrical';
      diagnosticMessage = '🔌 ELECTRICAL CIRCUIT PROBLEM: Check motor electrical connections';
      dustProbability = 0.05;
    } else if (tempHigh && currentNormal && vibNormal && valveNormal) {
      issueType = 'overheating';
      diagnosticMessage = '🔥 FAN OVERHEATING: Check cooling system and ventilation';
      dustProbability = 0.10;
    } else if (vibHigh && currentNormal && tempNormal && valveNormal && solidNormal && flowNormal) {
      issueType = 'bearing';
      diagnosticMessage = '⚙️ BEARING ISSUE: Lack of grease or starting imbalance';
      dustProbability = 0.15;
    } else if (tempHigh && vibHigh && currentNormal && valveNormal) {
      issueType = 'bearing_axle';
      diagnosticMessage = '🔧 BEARING/AXLE PROBLEM: Check bearing alignment and axle condition';
      dustProbability = 0.20;
    } else if (flowLow && !solidHigh) {
      issueType = 'dust_accumulation';
      const flowDeficit = (400 - sensors.pump_flow_rate) / 200;
      dustProbability = Math.min(0.30 + flowDeficit * 0.30, 0.60);
      diagnosticMessage = `🌫️ IMBALANCE DETECTED: Dust accumulation due to low pump flow (${sensors.pump_flow_rate.toFixed(0)} m³/h)`;
    } else if (solidHigh || effectiveSolidRate > 1.0) {
      issueType = 'dust_accumulation';
      dustProbability = Math.min(0.60, 0.30 + (effectiveSolidRate - 1.0) * 0.15);
      diagnosticMessage = flowLow 
        ? `🌫️ IMBALANCE DETECTED: Dust in system (Solid Rate: ${effectiveSolidRate.toFixed(1)}%, Flow: ${sensors.pump_flow_rate.toFixed(0)} m³/h)`
        : `⚠️ IMBALANCE DETECTING: Elevated solid rate (${effectiveSolidRate.toFixed(1)}%)`;
    } else if ((currentHigh || tempHigh || vibHigh) && (solidHigh || flowLow)) {
      issueType = 'both';
      dustProbability = Math.min(0.50, 0.25 + vibRisk * 0.25);
      diagnosticMessage = '⚠️ MULTIPLE ISSUES: Mechanical problem combined with dust accumulation';
    } else {
      issueType = 'none';
      diagnosticMessage = '✅ System operating normally';
      dustProbability = Math.max(0.05, vibRisk * 0.1);
    }

    const riskLevel = getRiskLevel(riskProbability);

    const contributingFactors = [
      {
        name: 'Vibration (Motor)',
        value: sensors.vib_motor,
        contribution: vibRisk * 0.30,
        status: sensors.vib_motor >= SENSOR_THRESHOLDS.vib_motor.critical ? 'critical' as const :
                sensors.vib_motor >= SENSOR_THRESHOLDS.vib_motor.warning ? 'warning' as const : 'normal' as const,
        threshold: SENSOR_THRESHOLDS.vib_motor,
      },
      {
        name: 'Temperature (Motor)',
        value: sensors.temp_motor,
        contribution: tempRisk * 0.25,
        status: sensors.temp_motor >= SENSOR_THRESHOLDS.temp_motor.critical ? 'critical' as const :
                sensors.temp_motor >= SENSOR_THRESHOLDS.temp_motor.warning ? 'warning' as const : 'normal' as const,
        threshold: SENSOR_THRESHOLDS.temp_motor,
      },
      {
        name: 'Motor Current',
        value: sensors.motor_current,
        contribution: currentRisk * 0.10,
        status: sensors.motor_current >= SENSOR_THRESHOLDS.motor_current.critical ? 'critical' as const :
                sensors.motor_current >= SENSOR_THRESHOLDS.motor_current.warning ? 'warning' as const : 'normal' as const,
        threshold: SENSOR_THRESHOLDS.motor_current,
      },
      {
        name: 'Solid Rate',
        value: sensors.solid_rate,
        contribution: sensors.solid_rate > 1.0 ? 0.60 : sensors.solid_rate * 0.15,
        status: sensors.solid_rate >= SENSOR_THRESHOLDS.solid_rate.critical ? 'critical' as const :
                sensors.solid_rate >= SENSOR_THRESHOLDS.solid_rate.warning ? 'warning' as const : 'normal' as const,
        threshold: SENSOR_THRESHOLDS.solid_rate,
      },
      {
        name: 'Pump Flow Rate',
        value: sensors.pump_flow_rate,
        contribution: sensors.pump_flow_rate < 350 ? 0.40 : (sensors.pump_flow_rate < 400 ? 0.20 : 0.0),
        status: sensors.pump_flow_rate < SENSOR_THRESHOLDS.pump_flow_rate.critical ? 'critical' as const :
                sensors.pump_flow_rate < SENSOR_THRESHOLDS.pump_flow_rate.warning ? 'warning' as const : 'normal' as const,
        threshold: SENSOR_THRESHOLDS.pump_flow_rate,
      },
    ].sort((a, b) => b.contribution - a.contribution);

    // Recommended actions based on issue type
    let recommendedActions: string[] = [];
    if (issueType === 'power_loss') {
      recommendedActions = ['Check power distribution panel', 'Verify main circuit breakers', 'Inspect power cables'];
    } else if (issueType === 'electrical') {
      recommendedActions = ['Inspect motor electrical connections', 'Check for loose wiring', 'Test motor insulation'];
    } else if (issueType === 'overheating') {
      recommendedActions = ['Check cooling system', 'Inspect ventilation paths', 'Verify coolant levels'];
    } else if (issueType === 'bearing') {
      recommendedActions = ['Lubricate bearings immediately', 'Check bearing alignment', 'Inspect for wear patterns'];
    } else if (issueType === 'bearing_axle') {
      recommendedActions = ['Stop and inspect bearing assembly', 'Check axle alignment', 'Verify shaft balance'];
    } else if (issueType === 'dust_accumulation') {
      recommendedActions = ['Clean fan blades and filters', 'Inspect air intake system', 'Check dust extraction'];
    } else if (issueType === 'both') {
      recommendedActions = ['Full maintenance inspection required', 'Clean all filters', 'Lubricate bearings'];
    } else {
      recommendedActions = ['Continue routine monitoring'];
    }

    return {
      risk_probability: riskProbability,
      risk_level: riskLevel,
      failure_within_24h: riskProbability >= 0.5,
      dust_caused: dustProbability >= 0.5,
      dust_probability: dustProbability,
      confidence: 0.85 + Math.random() * 0.1,
      contributing_factors: contributingFactors,
      recommended_actions: recommendedActions,
      time_to_failure_hours: riskProbability >= 0.5 ? Math.round((1 - riskProbability) * 48) : null,
      potential_savings: riskProbability >= 0.3 ? 200000 : 0,
      issue_type: issueType,
      diagnostic_message: diagnosticMessage,
    };
  };

  // Handle scenario selection
  const handleScenarioSelect = useCallback((preset: ScenarioPreset) => {
    setCurrentScenario(preset.id);
    setSensors(preset.sensors);
  }, []);

  // Generate mock historical data
  useEffect(() => {
    const times = generateTimeLabels(24, 60);
    const data = times.map((time, i) => ({
      time,
      risk: Math.max(5, Math.min(95, 20 + Math.sin(i / 3) * 15 + Math.random() * 20)),
      vib_motor: addNoise(sensors.vib_motor, 0.1),
      temp_motor: addNoise(sensors.temp_motor, 0.05),
      motor_current: addNoise(sensors.motor_current, 0.02),
    }));
    setHistoricalData(data);
  }, []);

  // Default prediction while loading
  const currentPrediction = prediction || calculateLocalPrediction(sensors);

  // Fan spinning state based on risk
  const fanSpeed = currentPrediction.risk_level === 'critical' ? 'fast' :
                   currentPrediction.risk_level === 'high' ? 'fast' :
                   currentPrediction.risk_level === 'medium' ? 'normal' : 'slow';

  const fanStatus = currentPrediction.risk_level === 'critical' || currentPrediction.risk_level === 'high' ? 'critical' :
                    currentPrediction.risk_level === 'medium' ? 'warning' : 'normal';

  return (
    <div className="min-h-screen bg-pattern">
      {/* Header */}
      <header className="sticky top-0 z-50 glass border-b border-white/10">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <OCPLogo size="md" />
            
            <div className="hidden md:flex items-center gap-6">
              <nav className="flex gap-1">
                {[
                  { id: 'live', label: 'Live Simulation', icon: '🔴' },
                  { id: 'analysis', label: 'Analysis', icon: '📊' },
                  { id: 'model', label: 'Model Info', icon: '🧠' },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`
                      px-4 py-2 rounded-lg text-sm font-medium transition-all
                      ${activeTab === tab.id
                        ? 'bg-ocp-green text-white'
                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                      }
                    `}
                  >
                    <span className="mr-1">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>

            <div className="flex items-center gap-4">
              <div className="text-right hidden sm:block">
                <p className="text-xs text-gray-400">Line 307 • Fan C07</p>
                <p className="text-sm text-white font-medium">Real-time Monitoring</p>
              </div>
              <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Tab Bar */}
      <div className="md:hidden sticky top-16 z-40 glass border-b border-white/10 px-4 py-2">
        <div className="flex gap-2">
          {[
            { id: 'live', label: 'Live', icon: '🔴' },
            { id: 'analysis', label: 'Analysis', icon: '📊' },
            { id: 'model', label: 'Model', icon: '🧠' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`
                flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all
                ${activeTab === tab.id
                  ? 'bg-ocp-green text-white'
                  : 'text-gray-400 bg-white/5'
                }
              `}
            >
              <span className="mr-1">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <main className="container mx-auto px-4 py-6">
        {/* Alert Banner */}
        <AlertBanner prediction={currentPrediction} />

        <AnimatePresence mode="wait">
          {/* Live Simulation Tab */}
          {activeTab === 'live' && (
            <motion.div
              key="live"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Quick Scenarios */}
              <QuickScenarioBar currentScenario={currentScenario} onSelect={handleScenarioSelect} />

              {/* Main Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Fan Visualization & Risk */}
                <div className="lg:col-span-1 space-y-6">
                  {/* Fan Status Card */}
                  <motion.div 
                    className="card p-6 text-center"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                  >
                    <h3 className="text-lg font-semibold text-white mb-4">Fan C07 Status</h3>
                    
                    <div className="flex justify-center mb-4">
                      <FanIcon 
                        spinning={true} 
                        speed={fanSpeed}
                        size={120}
                        status={fanStatus}
                      />
                    </div>

                    <RiskGauge value={currentPrediction.risk_probability} size="md" />
                  </motion.div>
                </div>

                {/* Right Column - Alert */}
                <div className="lg:col-span-2 space-y-6">
                  {/* Alert Panel */}
                  <AlertPanel prediction={currentPrediction} />
                </div>
              </div>

              {/* Full Width Sensor Controls */}
              <div className="card p-6 w-full">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>🎛️</span> Adjust Sensor Values
                  <span className="text-xs text-gray-400 font-normal ml-2">
                    (Simulate different scenarios)
                  </span>
                </h3>
                <SensorControls 
                  sensors={sensors} 
                  onChange={setSensors}
                />
              </div>

              {/* Full Width Scenario Selector */}
              <div className="card p-6">
                <ScenarioSelector 
                  currentScenario={currentScenario}
                  onSelect={handleScenarioSelect}
                />
              </div>
            </motion.div>
          )}

          {/* Analysis Tab - REAL-TIME INTERACTIVE */}
          {activeTab === 'analysis' && (
            <motion.div
              key="analysis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Real-Time Session Stats */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <motion.div className="card p-4" whileHover={{ scale: 1.02 }}>
                  <span className="text-2xl">🔬</span>
                  <h3 className="text-gray-400 text-sm mt-2">Tests Performed</h3>
                  <p className="text-2xl font-bold text-white">{sessionStats.testsPerformed}</p>
                </motion.div>
                <motion.div className="card p-4" whileHover={{ scale: 1.02 }}>
                  <span className="text-2xl">📊</span>
                  <h3 className="text-gray-400 text-sm mt-2">Sensor Changes</h3>
                  <p className="text-2xl font-bold text-white">{sessionStats.totalInteractions}</p>
                </motion.div>
                <motion.div className="card p-4" whileHover={{ scale: 1.02 }}>
                  <span className="text-2xl">⚠️</span>
                  <h3 className="text-gray-400 text-sm mt-2">High Risk Events</h3>
                  <p className="text-2xl font-bold text-amber-400">{sessionStats.highRiskEvents}</p>
                </motion.div>
                <motion.div className="card p-4" whileHover={{ scale: 1.02 }}>
                  <span className="text-2xl">🚨</span>
                  <h3 className="text-gray-400 text-sm mt-2">Critical Alerts</h3>
                  <p className="text-2xl font-bold text-red-400">{sessionStats.criticalAlerts}</p>
                </motion.div>
                <motion.div className="card p-4" whileHover={{ scale: 1.02 }}>
                  <span className="text-2xl">📈</span>
                  <h3 className="text-gray-400 text-sm mt-2">Max Risk Reached</h3>
                  <p className="text-2xl font-bold text-white">{sessionStats.maxRiskReached.toFixed(1)}%</p>
                </motion.div>
                <motion.div className="card p-4" whileHover={{ scale: 1.02 }}>
                  <span className="text-2xl">⏱️</span>
                  <h3 className="text-gray-400 text-sm mt-2">Session Duration</h3>
                  <p className="text-2xl font-bold text-white">
                    {Math.floor((Date.now() - sessionStats.sessionStartTime.getTime()) / 60000)}m
                  </p>
                </motion.div>
              </div>

              {/* Real-Time Risk Chart */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>📈</span> Live Risk History
                  <span className="text-xs text-gray-400 font-normal ml-2">
                    (Updates as you adjust sensors)
                  </span>
                </h3>
                {riskHistory.length > 0 ? (
                  <RiskHistoryChart data={riskHistory.map(h => ({ time: h.time, risk: h.risk }))} />
                ) : (
                  <div className="h-64 flex items-center justify-center text-gray-500">
                    <p>Start adjusting sensors to see risk history...</p>
                  </div>
                )}
              </div>

              {/* Dynamic Feature Importance - Based on YOUR interactions */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>🎯</span> Variables Most Likely to Cause Failure
                  <span className="text-xs text-gray-400 font-normal ml-2">
                    (Ranked by YOUR testing - most impactful first)
                  </span>
                </h3>
                
                {sensorHistory.length > 0 ? (
                  <div className="space-y-3">
                    {dynamicFeatureImportance.filter(f => f.interactions > 0).map((feature, idx) => {
                      const maxImpact = Math.max(...dynamicFeatureImportance.map(f => f.totalRiskImpact)) || 1;
                      const barWidth = (feature.totalRiskImpact / maxImpact) * 100;
                      
                      const categoryColors: Record<string, string> = {
                        vibration: 'bg-red-500',
                        temperature: 'bg-orange-500',
                        current: 'bg-yellow-500',
                        flow: 'bg-blue-500',
                        valve: 'bg-purple-500',
                      };
                      
                      return (
                        <motion.div
                          key={feature.sensor}
                          className="relative"
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.05 }}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-2">
                              <span className="text-lg font-bold text-white">#{idx + 1}</span>
                              <span className="text-sm text-gray-300">{feature.name}</span>
                              <span className={`px-2 py-0.5 rounded text-xs ${categoryColors[feature.category]} text-white`}>
                                {feature.category}
                              </span>
                            </div>
                            <div className="text-right text-sm">
                              <span className="text-gray-400">{feature.interactions} changes</span>
                              <span className="text-white ml-2">Δ {feature.avgRiskDelta.toFixed(1)}% avg</span>
                            </div>
                          </div>
                          <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                            <motion.div
                              className={`h-full rounded-full ${categoryColors[feature.category]}`}
                              initial={{ width: 0 }}
                              animate={{ width: `${barWidth}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                        </motion.div>
                      );
                    })}
                    
                    {dynamicFeatureImportance.filter(f => f.interactions === 0).length > 0 && (
                      <div className="mt-4 pt-4 border-t border-white/10">
                        <p className="text-sm text-gray-500">
                          Not tested yet: {dynamicFeatureImportance.filter(f => f.interactions === 0).map(f => f.name).join(', ')}
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="h-48 flex items-center justify-center text-gray-500 border-2 border-dashed border-gray-700 rounded-lg">
                    <div className="text-center">
                      <p className="text-xl mb-2">🔬</p>
                      <p>Start testing sensors to see which ones cause the most risk!</p>
                      <p className="text-sm mt-2">Go to Live Simulation tab and adjust the sliders</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Recent Sensor Interactions Log */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>📋</span> Recent Sensor Changes
                  <span className="text-xs text-gray-400 font-normal ml-2">
                    (Last 10 interactions)
                  </span>
                </h3>
                
                {sensorHistory.length > 0 ? (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {[...sensorHistory].reverse().slice(0, 10).map((interaction, idx) => (
                      <motion.div
                        key={idx}
                        className="flex items-center justify-between p-3 rounded-lg bg-white/5"
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                      >
                        <div className="flex items-center gap-3">
                          <span className={`w-2 h-2 rounded-full ${
                            interaction.riskDelta > 10 ? 'bg-red-500' :
                            interaction.riskDelta > 0 ? 'bg-amber-500' : 'bg-green-500'
                          }`} />
                          <span className="text-gray-300">{interaction.sensor.replace(/_/g, ' ')}</span>
                          <span className="text-gray-500">
                            {interaction.oldValue.toFixed(1)} → {interaction.newValue.toFixed(1)}
                          </span>
                        </div>
                        <div className={`font-medium ${
                          interaction.riskDelta > 0 ? 'text-red-400' : 'text-green-400'
                        }`}>
                          {interaction.riskDelta > 0 ? '+' : ''}{interaction.riskDelta.toFixed(1)}% risk
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-8">No interactions yet. Start testing!</p>
                )}
              </div>

              {/* Historic Failures Section */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>📜</span> Historic Critical Errors
                  <span className="text-xs text-gray-400 font-normal ml-2">
                    ({historicFailures.length} historical + {sessionCriticalErrors.length} from this session)
                  </span>
                </h3>
                
                {/* Tabs for Historical vs Session */}
                <div className="flex gap-2 mb-4">
                  <span className="px-3 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400 border border-blue-500/30">
                    📁 {historicFailures.length} Historical Records (2019)
                  </span>
                  <span className="px-3 py-1 rounded-full text-xs bg-red-500/20 text-red-400 border border-red-500/30">
                    🔴 {sessionCriticalErrors.length} Session Critical Errors
                  </span>
                </div>

                {/* Combined Failures Table */}
                <div className="overflow-x-auto max-h-96 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-ocp-dark">
                      <tr className="text-left text-gray-400 border-b border-white/10">
                        <th className="pb-3 pr-4">Source</th>
                        <th className="pb-3 pr-4">Date</th>
                        <th className="pb-3 pr-4">Time</th>
                        <th className="pb-3 pr-4">Type</th>
                        <th className="pb-3 pr-4">Duration</th>
                        <th className="pb-3">Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {/* Session Critical Errors First (Most Recent) */}
                      {[...sessionCriticalErrors].reverse().map((error, idx) => (
                        <motion.tr
                          key={`session-${idx}`}
                          className="border-b border-white/5 hover:bg-white/5"
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.02 }}
                        >
                          <td className="py-3 pr-4">
                            <span className="px-2 py-1 rounded text-xs bg-red-500/20 text-red-400">
                              🔴 Session
                            </span>
                          </td>
                          <td className="py-3 pr-4 text-white">{error.date}</td>
                          <td className="py-3 pr-4 text-gray-300">{error.time}</td>
                          <td className="py-3 pr-4">
                            <span className="px-2 py-1 rounded text-xs bg-red-500/30 text-red-300">
                              {error.failure_type}
                            </span>
                          </td>
                          <td className="py-3 pr-4 text-amber-400">{error.duration}</td>
                          <td className="py-3 text-gray-300 max-w-md truncate">{error.description}</td>
                        </motion.tr>
                      ))}
                      
                      {/* Historical Failures */}
                      {historicFailures.map((failure, idx) => (
                        <motion.tr
                          key={`hist-${idx}`}
                          className="border-b border-white/5 hover:bg-white/5"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: (sessionCriticalErrors.length + idx) * 0.02 }}
                        >
                          <td className="py-3 pr-4">
                            <span className="px-2 py-1 rounded text-xs bg-blue-500/20 text-blue-400">
                              📁 Historical
                            </span>
                          </td>
                          <td className="py-3 pr-4 text-white">{failure.date}</td>
                          <td className="py-3 pr-4 text-gray-300">{failure.time}</td>
                          <td className="py-3 pr-4">
                            <span className={`px-2 py-1 rounded text-xs ${
                              failure.failure_type === 'VIBRATION' ? 'bg-red-500/30 text-red-300' :
                              failure.failure_type === 'TEMPERATURE' ? 'bg-orange-500/30 text-orange-300' :
                              failure.failure_type === 'BEARING' ? 'bg-purple-500/30 text-purple-300' :
                              failure.failure_type === 'MECHANICAL' ? 'bg-yellow-500/30 text-yellow-300' :
                              'bg-gray-500/30 text-gray-300'
                            }`}>
                              {failure.failure_type}
                            </span>
                          </td>
                          <td className="py-3 pr-4 text-amber-400">{failure.duration}</td>
                          <td className="py-3 text-gray-300 max-w-md truncate">{failure.description}</td>
                        </motion.tr>
                      ))}
                      
                      {historicFailures.length === 0 && sessionCriticalErrors.length === 0 && (
                        <tr>
                          <td colSpan={6} className="py-8 text-center text-gray-500">
                            No failure records yet. Historical data will load from the API.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Summary Stats */}
                <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 rounded-lg bg-blue-500/10">
                    <p className="text-2xl font-bold text-blue-400">{historicFailures.length}</p>
                    <p className="text-xs text-gray-400">Historical Failures</p>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-red-500/10">
                    <p className="text-2xl font-bold text-red-400">{sessionCriticalErrors.length}</p>
                    <p className="text-xs text-gray-400">Session Critical</p>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-purple-500/10">
                    <p className="text-2xl font-bold text-purple-400">
                      {historicFailures.filter(f => f.failure_type === 'VIBRATION').length +
                       sessionCriticalErrors.filter(f => f.failure_type === 'VIBRATION').length}
                    </p>
                    <p className="text-xs text-gray-400">Vibration Issues</p>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-orange-500/10">
                    <p className="text-2xl font-bold text-orange-400">
                      {historicFailures.filter(f => f.failure_type === 'TEMPERATURE' || f.failure_type === 'OVERHEATING').length +
                       sessionCriticalErrors.filter(f => f.failure_type === 'TEMPERATURE' || f.failure_type === 'OVERHEATING').length}
                    </p>
                    <p className="text-xs text-gray-400">Temperature Issues</p>
                  </div>
                </div>
              </div>

              {/* Reset Session Button */}
              <div className="flex justify-center gap-4">
                <button
                  onClick={() => {
                    setSensorHistory([]);
                    setRiskHistory([]);
                    setSessionCriticalErrors([]);
                    setSensors(DEFAULT_SENSORS); // Reset sensors to default
                    setSessionStats({
                      totalInteractions: 0,
                      highRiskEvents: 0,
                      criticalAlerts: 0,
                      maxRiskReached: 0,
                      sessionStartTime: new Date(),
                      testsPerformed: 0,
                    });
                    // Clear backend session (but keep training data)
                    fetch(`${API_URL}/api/session`, { method: 'DELETE' })
                      .then(res => res.json())
                      .then(result => {
                        console.log('🗑️ Session cleared:', result);
                      });
                  }}
                  className="px-6 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-all"
                >
                  🔄 Reset All Data
                </button>
                <button
                  onClick={async () => {
                    const res = await fetch(`${API_URL}/api/training/stats`);
                    const stats = await res.json();
                    alert(`📊 Training Data Stats:\n\n` +
                      `Total Points: ${stats.total_points}\n` +
                      `Critical Points: ${stats.critical_points}\n` +
                      `Ready for Training: ${stats.ready_for_training ? 'Yes ✅' : 'No (need 100+ points)'}\n\n` +
                      `By Issue Type:\n${Object.entries(stats.by_issue_type || {}).map(([k, v]) => `  - ${k}: ${v}`).join('\n')}`
                    );
                  }}
                  className="px-6 py-2 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 transition-all"
                >
                  📊 View Training Stats
                </button>
              </div>
            </motion.div>
          )}

          {/* Model Info Tab */}
          {activeTab === 'model' && (
            <motion.div
              key="model"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Model Architecture */}
              <div className="card p-6">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                  <span>🧠</span> LSTM Deep Learning Model
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  {[
                    { label: 'Architecture', value: 'Bidirectional LSTM', icon: '🏗️' },
                    { label: 'Features', value: `${MODEL_INFO.features} input features`, icon: '📊' },
                    { label: 'Parameters', value: MODEL_INFO.parameters.toLocaleString(), icon: '⚙️' },
                    { label: 'Prediction Window', value: '24 hours', icon: '⏱️' },
                  ].map((item) => (
                    <div key={item.label} className="glass p-4 rounded-lg">
                      <span className="text-2xl">{item.icon}</span>
                      <p className="text-gray-400 text-sm mt-2">{item.label}</p>
                      <p className="text-white font-semibold">{item.value}</p>
                    </div>
                  ))}
                </div>

                {/* Architecture Visualization */}
                <div className="bg-ocp-dark rounded-xl p-6 overflow-x-auto">
                  <div className="flex items-center justify-between min-w-[600px] gap-4">
                    {/* Input Layer */}
                    <div className="text-center">
                      <div className="w-24 h-24 rounded-lg bg-blue-500/20 border border-blue-500/50 flex items-center justify-center mb-2">
                        <div className="text-blue-400">
                          <p className="text-xs">Input</p>
                          <p className="text-lg font-bold">22</p>
                          <p className="text-xs">features</p>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500">Sensor Data</p>
                    </div>

                    <div className="text-ocp-green text-2xl">→</div>

                    {/* LSTM Layers */}
                    <div className="text-center">
                      <div className="w-32 h-24 rounded-lg bg-purple-500/20 border border-purple-500/50 flex items-center justify-center mb-2">
                        <div className="text-purple-400">
                          <p className="text-xs">Bi-LSTM</p>
                          <p className="text-lg font-bold">128</p>
                          <p className="text-xs">units</p>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500">Temporal Patterns</p>
                    </div>

                    <div className="text-ocp-green text-2xl">→</div>

                    <div className="text-center">
                      <div className="w-28 h-24 rounded-lg bg-indigo-500/20 border border-indigo-500/50 flex items-center justify-center mb-2">
                        <div className="text-indigo-400">
                          <p className="text-xs">LSTM</p>
                          <p className="text-lg font-bold">64</p>
                          <p className="text-xs">units</p>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500">Feature Learning</p>
                    </div>

                    <div className="text-ocp-green text-2xl">→</div>

                    {/* Dense Layer */}
                    <div className="text-center">
                      <div className="w-24 h-24 rounded-lg bg-green-500/20 border border-green-500/50 flex items-center justify-center mb-2">
                        <div className="text-green-400">
                          <p className="text-xs">Dense</p>
                          <p className="text-lg font-bold">32</p>
                          <p className="text-xs">units</p>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500">Classification</p>
                    </div>

                    <div className="text-ocp-green text-2xl">→</div>

                    {/* Output */}
                    <div className="text-center">
                      <div className="w-24 h-24 rounded-lg bg-ocp-green/20 border border-ocp-green/50 flex items-center justify-center mb-2">
                        <div className="text-ocp-green">
                          <p className="text-xs">Output</p>
                          <p className="text-lg font-bold">Sigmoid</p>
                          <p className="text-xs">0-1</p>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500">Risk Score</p>
                    </div>
                  </div>
                </div>

                {/* Performance Metrics */}
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: 'Recall', value: MODEL_INFO.recall, color: '#22c55e' },
                    { label: 'Precision', value: MODEL_INFO.precision, color: '#3b82f6' },
                    { label: 'F1-Score', value: MODEL_INFO.f1_score, color: '#8b5cf6' },
                    { label: 'Dust Detection', value: MODEL_INFO.dust_detection_rate, color: '#f59e0b' },
                  ].map((metric) => (
                    <div key={metric.label} className="glass p-4 rounded-lg">
                      <p className="text-gray-400 text-sm">{metric.label}</p>
                      <div className="flex items-end gap-2 mt-1">
                        <span className="text-2xl font-bold" style={{ color: metric.color }}>
                          {(metric.value * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-white/10 rounded-full mt-2 overflow-hidden">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ backgroundColor: metric.color }}
                          initial={{ width: 0 }}
                          animate={{ width: `${metric.value * 100}%` }}
                          transition={{ duration: 1, delay: 0.5 }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* How It Works */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4">How the AI Predicts Failures</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="glass p-4 rounded-lg">
                    <div className="text-3xl mb-3">📊</div>
                    <h4 className="font-medium text-white mb-2">1. Data Collection</h4>
                    <p className="text-sm text-gray-400">
                      Sensors continuously monitor vibration, temperature, and current from Fan C07.
                      Data is collected every minute.
                    </p>
                  </div>
                  
                  <div className="glass p-4 rounded-lg">
                    <div className="text-3xl mb-3">🧠</div>
                    <h4 className="font-medium text-white mb-2">2. Pattern Recognition</h4>
                    <p className="text-sm text-gray-400">
                      The LSTM learns temporal patterns from 22 historical failures.
                      It identifies the signature of impending failures.
                    </p>
                  </div>
                  
                  <div className="glass p-4 rounded-lg">
                    <div className="text-3xl mb-3">🚨</div>
                    <h4 className="font-medium text-white mb-2">3. Early Warning</h4>
                    <p className="text-sm text-gray-400">
                      When patterns match pre-failure conditions, alerts are triggered
                      24 hours before failure occurs.
                    </p>
                  </div>
                </div>

                <div className="mt-6 p-4 rounded-lg bg-amber-500/10 border border-amber-500/30">
                  <h4 className="font-medium text-amber-400 flex items-center gap-2 mb-2">
                    <span>🌫️</span> Dust (Balourd) Detection
                  </h4>
                  <p className="text-sm text-gray-300">
                    The model specifically learned that <strong>vibration increases</strong> are the 
                    strongest predictor of failure. This directly correlates with dust accumulation 
                    on fan blades causing imbalance. When vibration rises without temperature spikes, 
                    it indicates dust buildup — allowing targeted cleaning before failure.
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 mt-12 py-6">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <OCPLogo size="sm" showText={false} />
              <div>
                <p className="text-sm text-gray-400">OCP Group - Predictive Maintenance</p>
                <p className="text-xs text-gray-500">Powered by LSTM Deep Learning</p>
              </div>
            </div>
            
            <div className="text-center md:text-right">
              <p className="text-sm text-gray-400">Line 307 • Fan C07</p>
              <p className="text-xs text-gray-500">
                Preventing $4.4M+ in annual losses
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
