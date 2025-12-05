'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { SENSOR_INFO, SENSOR_THRESHOLDS } from '@/lib/constants';
import { SensorData } from '@/types';
import { useI18n } from '@/lib/i18n';

interface SensorControlsProps {
  sensors: SensorData;
  onChange: (sensors: SensorData) => void;
  disabled?: boolean;
}

export function SensorControls({ sensors, onChange, disabled = false }: SensorControlsProps) {
  const { t } = useI18n();
  
  const handleChange = (key: keyof SensorData, value: number) => {
    const newSensors = { ...sensors, [key]: value };
    
    // AUTO-ADJUST: When pump_flow_rate decreases, increase solid_rate (dust accumulation)
    if (key === 'pump_flow_rate' && value < 400) {
      // Calculate solid rate increase based on flow reduction
      const flowDeficit = (400 - value) / 200; // 0 to 1 scale
      const autoSolidRate = Math.min(sensors.solid_rate + flowDeficit * 1.5, 3.5);
      // Only auto-increase if it would be higher than current
      if (autoSolidRate > sensors.solid_rate) {
        newSensors.solid_rate = Math.round(autoSolidRate * 10) / 10; // Round to 1 decimal
      }
    }
    
    onChange(newSensors);
  };

  // Custom display order for 4-column layout:
  const sensorKeys: Array<keyof SensorData> = [
    'motor_current', 'temp_motor', 'vib_motor', 'pump_flow_rate',
    'valve_opening', 'temp_opposite', 'vib_opposite', 'solid_rate',
  ];
  
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 w-full">
      {sensorKeys.map((key, index) => {
        const info = SENSOR_INFO[key];
        const thresholds = SENSOR_THRESHOLDS[key];
        const value = sensors[key];
        
        // Determine status (valve_opening and pump_flow_rate are inverted - lower is worse)
        let status: 'normal' | 'warning' | 'critical' = 'normal';
        if (key === 'valve_opening' || key === 'pump_flow_rate') {
          if (value < thresholds.critical) status = 'critical';
          else if (value < thresholds.warning) status = 'warning';
        } else {
          if (value >= thresholds.critical) status = 'critical';
          else if (value >= thresholds.warning) status = 'warning';
        }

        const statusColors = {
          normal: 'bg-green-500',
          warning: 'bg-amber-500',
          critical: 'bg-red-500',
        };

        const borderColors = {
          normal: 'border-green-500/20 hover:border-green-500/40',
          warning: 'border-amber-500/20 hover:border-amber-500/40',
          critical: 'border-red-500/20 hover:border-red-500/40',
        };

        return (
          <motion.div
            key={key}
            className={`bg-white/5 border ${borderColors[status]} rounded-xl p-3 transition-all duration-300`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                <span className="text-lg">{info.icon}</span>
                <span className="text-xs font-medium text-gray-400">{t.sensors.sensorNames[key] || info.label}</span>
              </div>
              <div className={`w-2 h-2 rounded-full ${statusColors[status]}`} />
            </div>

            {/* Value */}
            <div className="flex items-baseline gap-1 mb-2">
              <span className="text-2xl font-bold text-white">
                {value.toFixed(key.includes('vib') ? 2 : 1)}
              </span>
              <span className="text-gray-500 text-xs">{info.unit}</span>
            </div>

            {/* Slider */}
            <input
              type="range"
              min={thresholds.min}
              max={thresholds.max}
              step={key.includes('vib') ? 0.1 : 1}
              value={value}
              onChange={(e) => handleChange(key, parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full h-1.5 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 bg-white/10"
              style={{
                background: `linear-gradient(to right, ${status === 'normal' ? '#22c55e' : status === 'warning' ? '#f59e0b' : '#ef4444'} 0%, ${status === 'normal' ? '#22c55e' : status === 'warning' ? '#f59e0b' : '#ef4444'} ${((value - thresholds.min) / (thresholds.max - thresholds.min)) * 100}%, rgba(255,255,255,0.1) ${((value - thresholds.min) / (thresholds.max - thresholds.min)) * 100}%, rgba(255,255,255,0.1) 100%)`
              }}
            />
            
            {/* Min/Max labels */}
            <div className="flex justify-between text-[10px] text-gray-500 mt-1">
              <span>{thresholds.min}</span>
              <span>{thresholds.max}</span>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}

// Compact sensor display
export function SensorDisplay({ sensors }: { sensors: SensorData }) {
  const { t } = useI18n();
  
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      {(Object.keys(sensors) as Array<keyof SensorData>).map((key) => {
        const info = SENSOR_INFO[key];
        const thresholds = SENSOR_THRESHOLDS[key];
        const value = sensors[key];
        
        let status: 'normal' | 'warning' | 'critical' = 'normal';
        if (key === 'valve_opening' || key === 'pump_flow_rate') {
          if (value < thresholds.critical) status = 'critical';
          else if (value < thresholds.warning) status = 'warning';
        } else {
          if (value >= thresholds.critical) status = 'critical';
          else if (value >= thresholds.warning) status = 'warning';
        }

        const statusColors = {
          normal: 'border-green-500/50 bg-green-500/10',
          warning: 'border-amber-500/50 bg-amber-500/10',
          critical: 'border-red-500/50 bg-red-500/10',
        };

        const textColors = {
          normal: 'text-green-400',
          warning: 'text-amber-400',
          critical: 'text-red-400',
        };

        // Get first word of translated label
        const translatedLabel = t.sensors.sensorNames[key] || info.label;
        const shortLabel = translatedLabel.split(' ')[0];

        return (
          <div
            key={key}
            className={`p-3 rounded-lg border ${statusColors[status]} transition-all`}
          >
            <div className="flex items-center gap-1 mb-1">
              <span className="text-sm">{info.icon}</span>
              <span className="text-xs text-gray-400 truncate">{shortLabel}</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className={`text-xl font-bold ${textColors[status]}`}>
                {value.toFixed(key.includes('vib') ? 1 : 0)}
              </span>
              <span className="text-xs text-gray-500">{info.unit}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
