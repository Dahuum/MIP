'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { RISK_LEVELS } from '@/lib/constants';
import { formatPercent } from '@/lib/utils';

interface RiskGaugeProps {
  value: number; // 0 to 1
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  animated?: boolean;
}

export function RiskGauge({ 
  value, 
  size = 'md', 
  showLabel = true,
  animated = true 
}: RiskGaugeProps) {
  const clampedValue = Math.min(Math.max(value, 0), 1);
  const percentage = Math.round(clampedValue * 100);
  
  // Determine risk level
  const riskLevel = 
    clampedValue >= 0.7 ? 'critical' :
    clampedValue >= 0.5 ? 'high' :
    clampedValue >= 0.3 ? 'medium' : 'low';
  
  const risk = RISK_LEVELS[riskLevel];

  const sizes = {
    sm: { width: 160, strokeWidth: 12, fontSize: 'text-2xl', labelSize: 'text-xs' },
    md: { width: 220, strokeWidth: 16, fontSize: 'text-4xl', labelSize: 'text-sm' },
    lg: { width: 300, strokeWidth: 20, fontSize: 'text-5xl', labelSize: 'text-base' },
  };

  const config = sizes[size];
  const radius = (config.width - config.strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - clampedValue * 0.75); // 270 degrees

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: config.width, height: config.width * 0.75 }}>
        <svg 
          width={config.width} 
          height={config.width} 
          className="transform -rotate-[135deg]"
          style={{ marginTop: -config.width * 0.125 }}
        >
          {/* Background arc */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={config.strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${circumference * 0.75} ${circumference}`}
          />
          
          {/* Risk zone indicators */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(34, 197, 94, 0.3)"
            strokeWidth={config.strokeWidth - 4}
            strokeLinecap="round"
            strokeDasharray={`${circumference * 0.225} ${circumference}`}
          />
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(245, 158, 11, 0.3)"
            strokeWidth={config.strokeWidth - 4}
            strokeLinecap="round"
            strokeDasharray={`${circumference * 0.15} ${circumference}`}
            strokeDashoffset={-circumference * 0.225}
          />
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(239, 68, 68, 0.3)"
            strokeWidth={config.strokeWidth - 4}
            strokeLinecap="round"
            strokeDasharray={`${circumference * 0.15} ${circumference}`}
            strokeDashoffset={-circumference * 0.375}
          />
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(220, 38, 38, 0.3)"
            strokeWidth={config.strokeWidth - 4}
            strokeLinecap="round"
            strokeDasharray={`${circumference * 0.225} ${circumference}`}
            strokeDashoffset={-circumference * 0.525}
          />
          
          {/* Value arc */}
          <motion.circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke={risk.color}
            strokeWidth={config.strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${circumference * 0.75} ${circumference}`}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: animated ? 1.5 : 0, ease: 'easeOut' }}
            style={{
              filter: `drop-shadow(0 0 10px ${risk.color})`,
            }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center" style={{ paddingTop: config.width * 0.1 }}>
          <motion.span
            className={`${config.fontSize} font-bold`}
            style={{ color: risk.color }}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5, duration: 0.5 }}
          >
            {percentage}%
          </motion.span>
          
          <motion.div
            className={`flex items-center gap-1 mt-1 ${config.labelSize}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
          >
            <span>{risk.icon}</span>
            <span style={{ color: risk.color }}>{risk.label}</span>
          </motion.div>
        </div>

        {/* Glow effect */}
        <div 
          className="absolute inset-0 rounded-full blur-2xl opacity-30"
          style={{ 
            background: `radial-gradient(circle, ${risk.color} 0%, transparent 70%)`,
          }}
        />
      </div>

      {showLabel && (
        <motion.div 
          className="text-center mt-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <p className="text-gray-400 text-sm">{risk.description}</p>
          <p className="text-gray-300 text-xs mt-1 font-medium">{risk.action}</p>
        </motion.div>
      )}
    </div>
  );
}

// Smaller inline risk indicator
export function RiskIndicator({ value, size = 'sm' }: { value: number; size?: 'sm' | 'md' }) {
  const riskLevel = 
    value >= 0.7 ? 'critical' :
    value >= 0.5 ? 'high' :
    value >= 0.3 ? 'medium' : 'low';
  
  const risk = RISK_LEVELS[riskLevel];
  const percentage = Math.round(value * 100);

  const sizeClasses = {
    sm: 'w-16 h-16 text-lg',
    md: 'w-24 h-24 text-2xl',
  };

  return (
    <motion.div
      className={`${sizeClasses[size]} rounded-full flex items-center justify-center font-bold relative`}
      style={{ 
        background: risk.bgColor,
        border: `2px solid ${risk.borderColor}`,
        color: risk.color,
      }}
      whileHover={{ scale: 1.1 }}
      transition={{ type: 'spring', stiffness: 400 }}
    >
      {percentage}%
      
      {riskLevel === 'critical' && (
        <motion.div
          className="absolute inset-0 rounded-full"
          style={{ border: `2px solid ${risk.color}` }}
          animate={{ scale: [1, 1.2, 1], opacity: [1, 0, 1] }}
          transition={{ duration: 1, repeat: Infinity }}
        />
      )}
    </motion.div>
  );
}
