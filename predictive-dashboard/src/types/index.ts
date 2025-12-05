// Sensor data types
export interface SensorData {
  motor_current: number;
  temp_opposite: number;
  temp_motor: number;
  vib_opposite: number;
  vib_motor: number;
  valve_opening: number;
  solid_rate: number;      // 0-4% - Solid content in flow (dust indicator)
  pump_flow_rate: number;  // 200-450 m³/h - Pump flow rate
}

// Prediction result from the model
export interface PredictionResult {
  risk_probability: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  failure_within_24h: boolean;
  dust_caused: boolean;
  dust_probability: number;
  confidence: number;
  contributing_factors: ContributingFactor[];
  recommended_actions: string[];
  time_to_failure_hours: number | null;
  potential_savings: number;
  issue_type: 'none' | 'dust_accumulation' | 'grease_check' | 'both' | 'electrical' | 'overheating' | 'bearing' | 'bearing_axle' | 'power_loss';
  diagnostic_message: string;  // Specific diagnostic message to display
}

export interface ContributingFactor {
  name: string;
  value: number;
  contribution: number;
  status: 'normal' | 'warning' | 'critical';
  threshold: {
    warning: number;
    critical: number;
  };
}

// Historical data point
export interface HistoricalDataPoint {
  timestamp: string;
  motor_current: number;
  temp_opposite: number;
  temp_motor: number;
  vib_opposite: number;
  vib_motor: number;
  valve_opening: number;
  risk_score: number;
  prediction: number;
}

// Failure event
export interface FailureEvent {
  id: string;
  timestamp: string;
  failure_type: string;
  dust_quantity_kg: number;
  was_predicted: boolean;
  prediction_probability: number;
  root_cause: string;
  downtime_hours: number;
  cost_impact: number;
}

// Dashboard stats
export interface DashboardStats {
  total_predictions: number;
  failures_prevented: number;
  total_savings: number;
  model_accuracy: number;
  current_risk_level: string;
  uptime_percentage: number;
  last_failure_date: string | null;
  days_since_failure: number;
}

// Scenario preset
export interface ScenarioPreset {
  id: string;
  name: string;
  description: string;
  icon: string;
  sensors: SensorData;
  expected_risk: 'low' | 'medium' | 'high' | 'critical';
}

// Sensor thresholds
export interface SensorThresholds {
  motor_current: { min: number; max: number; warning: number; critical: number };
  temp_opposite: { min: number; max: number; warning: number; critical: number };
  temp_motor: { min: number; max: number; warning: number; critical: number };
  vib_opposite: { min: number; max: number; warning: number; critical: number };
  vib_motor: { min: number; max: number; warning: number; critical: number };
  valve_opening: { min: number; max: number; warning: number; critical: number };
  solid_rate: { min: number; max: number; warning: number; critical: number };
  pump_flow_rate: { min: number; max: number; warning: number; critical: number };
}

// Model info
export interface ModelInfo {
  name: string;
  type: string;
  architecture: string;
  features: number;
  parameters: number;
  recall: number;
  precision: number;
  f1_score: number;
  dust_detection_rate: number;
  last_trained: string;
}

// Alert
export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'critical';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  sensor?: string;
  value?: number;
  threshold?: number;
}

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: string;
  timestamp: string;
}

// Real-time update
export interface RealTimeUpdate {
  sensors: SensorData;
  prediction: PredictionResult;
  timestamp: string;
}
