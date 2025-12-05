'use client';

import React from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';
import { motion } from 'framer-motion';

interface SensorChartProps {
  data: {
    time: string;
    value: number;
    prediction?: number;
  }[];
  title: string;
  color: string;
  unit: string;
  warningThreshold?: number;
  criticalThreshold?: number;
  showPrediction?: boolean;
}

export function SensorChart({
  data,
  title,
  color,
  unit,
  warningThreshold,
  criticalThreshold,
  showPrediction = false,
}: SensorChartProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass p-3 rounded-lg">
          <p className="text-gray-400 text-xs mb-1">{label}</p>
          <p className="text-white font-medium">
            {payload[0].value.toFixed(2)} {unit}
          </p>
          {showPrediction && payload[1] && (
            <p className="text-ocp-green text-sm">
              Predicted: {payload[1].value.toFixed(2)} {unit}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      className="card p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="text-sm font-medium text-gray-400 mb-4">{title}</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id={`gradient-${title}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="time" 
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
            />
            <YAxis 
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            
            {warningThreshold && (
              <ReferenceLine 
                y={warningThreshold} 
                stroke="#f59e0b" 
                strokeDasharray="3 3"
                label={{ value: 'Warning', fill: '#f59e0b', fontSize: 10 }}
              />
            )}
            
            {criticalThreshold && (
              <ReferenceLine 
                y={criticalThreshold} 
                stroke="#ef4444" 
                strokeDasharray="3 3"
                label={{ value: 'Critical', fill: '#ef4444', fontSize: 10 }}
              />
            )}
            
            <Area
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              fill={`url(#gradient-${title})`}
            />
            
            {showPrediction && (
              <Line
                type="monotone"
                dataKey="prediction"
                stroke="#00843D"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

interface RiskHistoryChartProps {
  data: {
    time: string;
    risk: number;
    actual?: number;
  }[];
}

export function RiskHistoryChart({ data }: RiskHistoryChartProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const risk = payload[0].value;
      const riskLevel = 
        risk >= 70 ? 'Critical' :
        risk >= 50 ? 'High' :
        risk >= 30 ? 'Medium' : 'Low';
      
      return (
        <div className="glass p-3 rounded-lg">
          <p className="text-gray-400 text-xs mb-1">{label}</p>
          <p className="text-white font-medium">{risk}% Risk</p>
          <p className={`text-sm ${
            risk >= 70 ? 'text-red-400' :
            risk >= 50 ? 'text-orange-400' :
            risk >= 30 ? 'text-amber-400' : 'text-green-400'
          }`}>
            {riskLevel}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      className="card p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="text-sm font-medium text-gray-400 mb-4">Risk History (24h)</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="30%" stopColor="#f59e0b" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#22c55e" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="time" 
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
            />
            <YAxis 
              domain={[0, 100]}
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            
            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
            <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="3 3" />
            <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="3 3" />
            
            <Area
              type="monotone"
              dataKey="risk"
              stroke="#00843D"
              strokeWidth={2}
              fill="url(#riskGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

interface MultiSensorChartProps {
  data: any[];
  sensors: {
    key: string;
    name: string;
    color: string;
  }[];
}

export function MultiSensorChart({ data, sensors }: MultiSensorChartProps) {
  return (
    <motion.div
      className="card p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="text-sm font-medium text-gray-400 mb-4">Multi-Sensor Overview</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="time" 
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
            />
            <YAxis 
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
            />
            <Tooltip 
              contentStyle={{ 
                background: 'rgba(0,0,0,0.8)', 
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
              }}
            />
            <Legend />
            
            {sensors.map((sensor) => (
              <Line
                key={sensor.key}
                type="monotone"
                dataKey={sensor.key}
                name={sensor.name}
                stroke={sensor.color}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
