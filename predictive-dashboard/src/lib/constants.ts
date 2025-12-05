import { SensorData, ScenarioPreset, SensorThresholds, ModelInfo } from '@/types';

// Default sensor values (normal operation) - calibrated for LOW risk
export const DEFAULT_SENSORS: SensorData = {
  motor_current: 24.5,
  temp_opposite: 48.0,
  temp_motor: 52.0,   // Below 60°C threshold
  vib_opposite: 0.8,
  vib_motor: 1.2,     // Below 1.5 threshold, ratio ~1.5
  valve_opening: 93.0,
  solid_rate: 0.5,       // Normal: < 1%, Warning: > 1%, Critical: > 2%
  pump_flow_rate: 420,   // Normal: > 400, Warning: < 400, Critical: < 300
};

// Sensor thresholds based on domain knowledge
export const SENSOR_THRESHOLDS: SensorThresholds = {
  motor_current: {
    min: 20,
    max: 28,
    warning: 26,
    critical: 27,
  },
  temp_opposite: {
    min: 30,
    max: 100,
    warning: 75,
    critical: 85,
  },
  temp_motor: {
    min: 30,
    max: 100,
    warning: 80,
    critical: 90,
  },
  vib_opposite: {
    min: 0,
    max: 4,
    warning: 1.2,
    critical: 1.8,
  },
  vib_motor: {
    min: 0,
    max: 4,
    warning: 2.5,
    critical: 3.0,
  },
  valve_opening: {
    min: 0,
    max: 100,
    warning: 85,  // Below this is warning
    critical: 80, // Below this is critical
  },
  solid_rate: {
    min: 0,
    max: 4,
    warning: 1.0,   // Above 1% triggers dust detection
    critical: 2.0,  // Above 2% is critical dust level
  },
  pump_flow_rate: {
    min: 200,
    max: 450,
    warning: 400,   // Below 400 increases dust probability
    critical: 300,  // Below 300 is critical
  },
};

// Preset scenarios for demonstration
export const SCENARIO_PRESETS: ScenarioPreset[] = [
  {
    id: 'normal',
    name: 'Normal Operation',
    description: 'All sensors within optimal range. Fan running smoothly.',
    icon: '✅',
    sensors: {
      motor_current: 24.5,
      temp_opposite: 48.0,
      temp_motor: 52.0,   // Well below warning threshold
      vib_opposite: 0.8,
      vib_motor: 1.2,     // Low vibration, healthy ratio
      valve_opening: 93.0,
      solid_rate: 0.5,    // Normal solid rate
      pump_flow_rate: 420, // Normal flow rate
    },
    expected_risk: 'low',
  },
  {
    id: 'dust_accumulation',
    name: 'Dust Accumulation',
    description: 'Early signs of dust buildup causing vibration increase.',
    icon: '🌫️',
    sensors: {
      motor_current: 25.2,
      temp_opposite: 62.0,
      temp_motor: 68.0,
      vib_opposite: 1.1,
      vib_motor: 2.4,
      valve_opening: 90.0,
      solid_rate: 1.5,     // High solid rate - dust detected!
      pump_flow_rate: 350,  // Low flow - dust accumulating
    },
    expected_risk: 'medium',
  },
  {
    id: 'warning_signs',
    name: 'Warning Signs',
    description: 'Elevated vibration and temperature. Monitor closely.',
    icon: '⚠️',
    sensors: {
      motor_current: 24.5,
      temp_opposite: 60.0,
      temp_motor: 71.0,
      vib_opposite: 1.3,
      vib_motor: 2.2,
      valve_opening: 93.0,
      solid_rate: 0.5,
      pump_flow_rate: 420,
    },
    expected_risk: 'medium',
  },
  {
    id: 'high_risk',
    name: 'High Risk',
    description: 'Significant anomalies detected. Schedule maintenance soon.',
    icon: '🔴',
    sensors: {
      motor_current: 25.8,
      temp_opposite: 75.0,
      temp_motor: 84.0,
      vib_opposite: 1.4,
      vib_motor: 2.7,
      valve_opening: 87.0,
      solid_rate: 2.2,     // Critical dust level
      pump_flow_rate: 310,  // Severely reduced flow
    },
    expected_risk: 'high',
  },
  {
    id: 'critical_failure',
    name: 'Critical - Imminent Failure',
    description: 'Severe imbalance detected. Failure likely within 24h.',
    icon: '🚨',
    sensors: {
      motor_current: 26.2,
      temp_opposite: 82.0,
      temp_motor: 94.0,
      vib_opposite: 1.8,
      vib_motor: 3.2,
      valve_opening: 85.0,
      solid_rate: 2.8,     // Extreme dust
      pump_flow_rate: 260,  // Critical flow restriction
    },
    expected_risk: 'critical',
  },
  {
    id: 'high_temp',
    name: 'High Temperature',
    description: 'Overheating condition. Check cooling system.',
    icon: '🌡️',
    sensors: {
      motor_current: 24.5,
      temp_opposite: 73.0,
      temp_motor: 87.0,
      vib_opposite: 0.8,
      vib_motor: 1.2,
      valve_opening: 93.0,
      solid_rate: 0.5,
      pump_flow_rate: 420,
    },
    expected_risk: 'high',
  },
  {
    id: 'bearing_wear',
    name: 'Bearing Wear',
    description: 'High vibration pattern typical of bearing degradation.',
    icon: '⚙️',
    sensors: {
      motor_current: 25.0,
      temp_opposite: 58.0,
      temp_motor: 65.0,
      vib_opposite: 1.5,
      vib_motor: 3.1,
      valve_opening: 93.0,
      solid_rate: 0.4,     // Normal - not dust related
      pump_flow_rate: 430,  // Normal flow - grease/bearing issue
    },
    expected_risk: 'high',
  },
];

// Model specifications
export const MODEL_INFO: ModelInfo = {
  name: 'LSTM Bidirectional v2',
  type: 'Deep Learning / LSTM',
  architecture: 'Bidirectional LSTM (128) → LSTM (64) → Dense (32) → Sigmoid',
  features: 22,
  parameters: 158721,
  recall: 0.85,
  precision: 0.78,
  f1_score: 0.81,
  dust_detection_rate: 0.92,
  last_trained: '2024-12-04',
};

// Sensor labels and units
export const SENSOR_INFO = {
  motor_current: {
    label: 'Motor Current',
    unit: 'A',
    icon: '⚡',
    description: 'Electrical current drawn by the motor',
  },
  temp_opposite: {
    label: 'Temperature (Opposite)',
    unit: '°C',
    icon: '🌡️',
    description: 'Temperature on the opposite side of motor',
  },
  temp_motor: {
    label: 'Temperature (Motor)',
    unit: '°C',
    icon: '🔥',
    description: 'Motor bearing temperature',
  },
  vib_opposite: {
    label: 'Vibration (Opposite)',
    unit: 'mm/s',
    icon: '📳',
    description: 'Vibration velocity on opposite side',
  },
  vib_motor: {
    label: 'Vibration (Motor)',
    unit: 'mm/s',
    icon: '🔊',
    description: 'Vibration velocity at motor bearing',
  },
  valve_opening: {
    label: 'Valve Opening',
    unit: '%',
    icon: '🔧',
    description: 'Damper valve position',
  },
  solid_rate: {
    label: 'Solid Rate',
    unit: '%',
    icon: '🌫️',
    description: 'Solid content in flow - dust indicator (>1% triggers detection)',
  },
  pump_flow_rate: {
    label: 'Pump Flow Rate',
    unit: 'm³/h',
    icon: '💧',
    description: 'Volumetric flow rate (<400 indicates dust buildup)',
  },
};

// Risk level configurations
export const RISK_LEVELS = {
  low: {
    label: 'Low Risk',
    color: '#22c55e',
    bgColor: 'rgba(34, 197, 94, 0.1)',
    borderColor: 'rgba(34, 197, 94, 0.3)',
    icon: '✅',
    description: 'System operating normally',
    action: 'Continue routine monitoring',
    maxThreshold: 0.3,
  },
  medium: {
    label: 'Moderate Risk',
    color: '#f59e0b',
    bgColor: 'rgba(245, 158, 11, 0.1)',
    borderColor: 'rgba(245, 158, 11, 0.3)',
    icon: '⚡',
    description: 'Early warning signs detected',
    action: 'Increase monitoring frequency',
    maxThreshold: 0.5,
  },
  high: {
    label: 'High Risk',
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.1)',
    borderColor: 'rgba(239, 68, 68, 0.3)',
    icon: '⚠️',
    description: 'Significant anomalies detected',
    action: 'Schedule preventive maintenance',
    maxThreshold: 0.7,
  },
  critical: {
    label: 'Critical',
    color: '#dc2626',
    bgColor: 'rgba(220, 38, 38, 0.15)',
    borderColor: 'rgba(220, 38, 38, 0.5)',
    icon: '🚨',
    description: 'Failure likely within 24 hours',
    action: 'Immediate inspection required',
    maxThreshold: 1.0,
  },
};

// Cost assumptions for ROI calculation
export const COST_ASSUMPTIONS = {
  cost_per_hour_downtime: 50000,
  average_failure_downtime_hours: 4,
  emergency_repair_cost: 25000,
  planned_maintenance_cost: 10000,
  failures_per_year_without_ai: 22,
  implementation_cost: 90000,
};

// Feature importance from LSTM model
export const TOP_FEATURES = [
  { name: 'Vib_Motor', importance: 0.182, category: 'vibration' },
  { name: 'Vib_Motor_roll_std', importance: 0.156, category: 'vibration' },
  { name: 'Vib_Total', importance: 0.134, category: 'vibration' },
  { name: 'Temp_Motor', importance: 0.098, category: 'temperature' },
  { name: 'Vib_Ratio', importance: 0.089, category: 'vibration' },
  { name: 'Temp_Diff', importance: 0.078, category: 'temperature' },
  { name: 'Motor_Current', importance: 0.067, category: 'current' },
  { name: 'Vib_Opposite', importance: 0.054, category: 'vibration' },
  { name: 'Power_Indicator', importance: 0.048, category: 'derived' },
  { name: 'Motor_Current_diff', importance: 0.042, category: 'current' },
];

// API configuration
export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  endpoints: {
    predict: '/api/predict',
    history: '/api/history',
    stats: '/api/stats',
    failures: '/api/failures',
    health: '/api/health',
  },
  refreshInterval: 5000, // 5 seconds for real-time updates
};
