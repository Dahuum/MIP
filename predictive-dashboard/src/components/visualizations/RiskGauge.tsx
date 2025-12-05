'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { RISK_LEVELS } from '@/lib/constants';
import { formatPercent } from '@/lib/utils';
import { useI18n } from '@/lib/i18n';

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
  const { t } = useI18n();
  const clampedValue = Math.min(Math.max(value, 0), 1);
  const percentage = Math.round(clampedValue * 100);
  
  // Determine risk level
  const riskLevel = 
    clampedValue >= 0.7 ? 'critical' :
    clampedValue >= 0.5 ? 'high' :
    clampedValue >= 0.3 ? 'medium' : 'low';
  
  const risk = RISK_LEVELS[riskLevel];
  
  // Get translated risk label
  const getRiskLabel = () => {
    switch (riskLevel) {
      case 'low': return t.risk.low;
      case 'medium': return t.risk.medium;
      case 'high': return t.risk.high;
      case 'critical': return t.risk.critical;
      default: return risk.label;
    }
  };

  const sizes = {
    sm: { width: 140, strokeWidth: 10, fontSize: 'text-xl', labelSize: 'text-[10px]' },
    md: { width: 180, strokeWidth: 12, fontSize: 'text-3xl', labelSize: 'text-xs' },
    lg: { width: 240, strokeWidth: 16, fontSize: 'text-4xl', labelSize: 'text-sm' },
  };

  const config = sizes[size];
  const radius = (config.width - config.strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  
  // Arc covers 270 degrees (0.75 of circle)
  // strokeDashoffset should go from full arc (no fill) to 0 (full fill)
  const arcLength = circumference * 0.75;
  const filledLength = arcLength * clampedValue;
  const strokeDashoffset = arcLength - filledLength;

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: config.width, height: config.width * 0.7 }}>
        <svg 
          width={config.width} 
          height={config.width} 
          className="transform -rotate-[135deg]"
          style={{ marginTop: -config.width * 0.15 }}
        >
          {/* Background arc - subtle gray */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={config.strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${arcLength} ${circumference}`}
          />
          
          {/* Risk zone indicators (background segments) */}
          {/* Green zone: 0-30% */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(34, 197, 94, 0.2)"
            strokeWidth={config.strokeWidth - 4}
            strokeDasharray={`${arcLength * 0.3} ${circumference}`}
            strokeDashoffset={0}
          />
          {/* Yellow zone: 30-50% */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(245, 158, 11, 0.2)"
            strokeWidth={config.strokeWidth - 4}
            strokeDasharray={`${arcLength * 0.2} ${circumference}`}
            strokeDashoffset={-arcLength * 0.3}
          />
          {/* Orange/Red zone: 50-70% */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(239, 68, 68, 0.2)"
            strokeWidth={config.strokeWidth - 4}
            strokeDasharray={`${arcLength * 0.2} ${circumference}`}
            strokeDashoffset={-arcLength * 0.5}
          />
          {/* Critical zone: 70-100% */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="rgba(220, 38, 38, 0.2)"
            strokeWidth={config.strokeWidth - 4}
            strokeDasharray={`${arcLength * 0.3} ${circumference}`}
            strokeDashoffset={-arcLength * 0.7}
          />
          
          {/* Value arc - shows the actual filled percentage */}
          <motion.circle
            key={`arc-${riskLevel}`}
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke={risk.color}
            strokeWidth={config.strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${arcLength} ${circumference}`}
            initial={{ strokeDashoffset: arcLength }}
            animate={{ 
              strokeDashoffset,
              filter: `drop-shadow(0 0 8px ${risk.color})`,
            }}
            transition={{ duration: animated ? 1.5 : 0, ease: 'easeOut' }}
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
            <span style={{ color: risk.color }}>{getRiskLabel()}</span>
          </motion.div>
        </div>

        {/* Glow effect - positioned at center and uses current risk color */}
        <motion.div 
          key={riskLevel} // Force re-render when risk level changes
          className="absolute rounded-full blur-2xl pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.35 }}
          style={{ 
            background: risk.color,
            width: config.width * 0.7,
            height: config.width * 0.5,
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        />
      </div>

      {showLabel && risk.description && (
        <motion.div 
          className="text-center mt-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <p className="text-gray-400 text-xs">{risk.description}</p>
          {risk.action && <p className="text-gray-300 text-[11px] mt-0.5 font-medium">{risk.action}</p>}
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
