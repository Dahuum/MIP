'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { SENSOR_INFO, SENSOR_THRESHOLDS } from '@/lib/constants';
import { SensorData } from '@/types';

interface SensorControlsProps {
  sensors: SensorData;
  onChange: (sensors: SensorData) => void;
  disabled?: boolean;
}

export function SensorControls({ sensors, onChange, disabled = false }: SensorControlsProps) {
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
  // Row 1: Motor Current | Solid Rate | Vib Opposite | Pump Flow Rate
  // Row 2: Temp Opposite | Temp Motor | Vib Motor | Valve Opening
  const sensorKeys: Array<keyof SensorData> = [
    'motor_current', 'temp_motor', 'vib_motor', 'pump_flow_rate',
    'valve_opening', 'temp_opposite', 'vib_opposite', 'solid_rate',
  ];
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 w-full">
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
          normal: 'from-green-500 to-emerald-500',
          warning: 'from-amber-500 to-orange-500',
          critical: 'from-red-500 to-rose-500',
        };

        const statusGlow = {
          normal: 'shadow-green-500/30',
          warning: 'shadow-amber-500/30',
          critical: 'shadow-red-500/30',
        };

        return (
          <motion.div
            key={key}
            className={`card p-4 transition-all duration-300 ${statusGlow[status]} shadow-lg`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-xl">{info.icon}</span>
                <span className="text-sm font-medium text-gray-300">{info.label}</span>
              </div>
              <div className={`
                px-2 py-0.5 rounded-full text-xs font-medium
                bg-gradient-to-r ${statusColors[status]} text-white
              `}>
                {status.toUpperCase()}
              </div>
            </div>

            <div className="flex items-end gap-2 mb-4">
              <span className="text-3xl font-bold text-white">
                {value.toFixed(key.includes('vib') ? 2 : 1)}
              </span>
              <span className="text-gray-400 text-sm mb-1">{info.unit}</span>
            </div>

            <div className="relative">
              <input
                type="range"
                min={thresholds.min}
                max={thresholds.max}
                step={key.includes('vib') ? 0.1 : 1}
                value={value}
                onChange={(e) => handleChange(key, parseFloat(e.target.value))}
                disabled={disabled}
                className="w-full h-2 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
              />
              
              {/* Threshold markers */}
              <div className="relative h-2 mt-2">
                <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 via-amber-500/20 to-red-500/20 rounded" />
                
                {/* Warning threshold */}
                <div 
                  className="absolute top-0 h-full w-0.5 bg-amber-500"
                  style={{ 
                    left: `${((thresholds.warning - thresholds.min) / (thresholds.max - thresholds.min)) * 100}%` 
                  }}
                />
                
                {/* Critical threshold */}
                <div 
                  className="absolute top-0 h-full w-0.5 bg-red-500"
                  style={{ 
                    left: `${((thresholds.critical - thresholds.min) / (thresholds.max - thresholds.min)) * 100}%` 
                  }}
                />
              </div>

              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>{thresholds.min}</span>
                <span className="text-amber-500">{thresholds.warning}</span>
                <span className="text-red-500">{thresholds.critical}</span>
                <span>{thresholds.max}</span>
              </div>
            </div>

            <p className="text-xs text-gray-500 mt-2">{info.description}</p>
          </motion.div>
        );
      })}
    </div>
  );
}

// Compact sensor display
export function SensorDisplay({ sensors }: { sensors: SensorData }) {
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

        return (
          <div
            key={key}
            className={`p-3 rounded-lg border ${statusColors[status]} transition-all`}
          >
            <div className="flex items-center gap-1 mb-1">
              <span className="text-sm">{info.icon}</span>
              <span className="text-xs text-gray-400 truncate">{info.label.split(' ')[0]}</span>
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
