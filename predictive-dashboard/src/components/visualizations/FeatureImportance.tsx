'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { TOP_FEATURES } from '@/lib/constants';

interface FeatureImportanceProps {
  customFeatures?: typeof TOP_FEATURES;
}

export function FeatureImportance({ customFeatures }: FeatureImportanceProps) {
  const features = customFeatures || TOP_FEATURES;

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'vibration':
        return '#ef4444';
      case 'temperature':
        return '#f59e0b';
      case 'current':
        return '#22c55e';
      default:
        return '#00843D';
    }
  };

  const chartData = features.map(f => ({
    ...f,
    fill: getCategoryColor(f.category),
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="glass p-3 rounded-lg">
          <p className="text-white font-medium">{data.name}</p>
          <p className="text-gray-400 text-sm">
            Importance: {(data.importance * 100).toFixed(1)}%
          </p>
          <p className="text-xs capitalize" style={{ color: getCategoryColor(data.category) }}>
            Category: {data.category}
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
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-400">
          What the AI Learned (Feature Importance)
        </h3>
        <div className="flex gap-3 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            Vibration
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            Temperature
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            Current
          </span>
        </div>
      </div>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={false} />
            <XAxis 
              type="number"
              domain={[0, 0.2]}
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <YAxis 
              type="category"
              dataKey="name"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
              width={100}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar 
              dataKey="importance" 
              radius={[0, 4, 4, 0]}
              fill="#00843D"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
        <p className="text-sm text-amber-400 flex items-start gap-2">
          <span className="text-lg">🌫️</span>
          <span>
            <strong>Key Insight:</strong> Vibration features dominate because dust accumulation 
            on fan blades causes imbalance, leading to increased vibration before failure.
          </span>
        </p>
      </div>
    </motion.div>
  );
}

// Compact version for sidebar
export function FeatureImportanceCompact() {
  return (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-500 uppercase">Top Predictors</h4>
      {TOP_FEATURES.slice(0, 5).map((feature, index) => (
        <motion.div
          key={feature.name}
          className="flex items-center gap-2"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${
                feature.category === 'vibration' ? 'bg-red-500' :
                feature.category === 'temperature' ? 'bg-amber-500' : 'bg-green-500'
              }`}
              initial={{ width: 0 }}
              animate={{ width: `${feature.importance * 500}%` }}
              transition={{ duration: 0.8, delay: index * 0.1 }}
            />
          </div>
          <span className="text-xs text-gray-400 w-24 truncate">{feature.name}</span>
        </motion.div>
      ))}
    </div>
  );
}
