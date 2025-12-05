'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RISK_LEVELS } from '@/lib/constants';
import { PredictionResult } from '@/types';
import { formatCurrency } from '@/lib/utils';
import { useI18n } from '@/lib/i18n';

interface AlertPanelProps {
  prediction: PredictionResult | null;
  onDismiss?: () => void;
}

export function AlertPanel({ prediction, onDismiss }: AlertPanelProps) {
  const { t } = useI18n();
  
  if (!prediction) return null;

  const risk = RISK_LEVELS[prediction.risk_level];
  const isCritical = prediction.risk_level === 'critical' || prediction.risk_level === 'high';

  // Get translated risk label
  const getRiskLabel = () => {
    switch (prediction.risk_level) {
      case 'low': return t.risk.low;
      case 'medium': return t.risk.medium;
      case 'high': return t.risk.high;
      case 'critical': return t.risk.critical;
      default: return risk.label;
    }
  };

  return (
    <AnimatePresence>
      {prediction && (
        <motion.div
          initial={{ opacity: 0, y: -20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -20, scale: 0.95 }}
          className={`
            relative overflow-hidden rounded-xl p-5 h-full
            ${isCritical ? 'alert-pulse' : ''}
          `}
          style={{
            background: risk.bgColor,
            border: `2px solid ${risk.borderColor}`,
          }}
        >
          {/* Background pattern */}
          <div className="absolute inset-0 opacity-10">
            <div 
              className="absolute inset-0"
              style={{
                backgroundImage: `radial-gradient(circle at 50% 50%, ${risk.color} 1px, transparent 1px)`,
                backgroundSize: '20px 20px',
              }}
            />
          </div>

          <div className="relative z-10">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <motion.span 
                  className="text-3xl"
                  animate={isCritical ? { scale: [1, 1.2, 1] } : {}}
                  transition={{ duration: 0.5, repeat: isCritical ? Infinity : 0 }}
                >
                  {risk.icon}
                </motion.span>
                <div>
                  <h3 className="text-lg font-bold" style={{ color: risk.color }}>
                    {getRiskLabel()}
                  </h3>
                  <p className="text-gray-300 text-sm">
                    {prediction.diagnostic_message || risk.description}
                  </p>
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-2xl font-bold" style={{ color: risk.color }}>
                  {Math.round(prediction.risk_probability * 100)}%
                </div>
                <div className="text-xs text-gray-400">{t.risk.riskProbability}</div>
              </div>
            </div>

            {/* Key insights - Compact Grid */}
            <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-2">
              {prediction.dust_caused ? (
                <div className="glass rounded-lg p-2.5">
                  <div className="flex items-center gap-1.5 text-amber-400 text-sm">
                    <span>🌫️</span>
                    <span className="font-medium">{t.alerts.imbalance}</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    {Math.round(prediction.dust_probability * 100)}% {t.alerts.dustProbability}
                  </p>
                </div>
              ) : null}
              
              {prediction.time_to_failure_hours != null && prediction.time_to_failure_hours > 0 ? (
                <div className="glass rounded-lg p-2.5">
                  <div className="flex items-center gap-1.5 text-sm" style={{ color: risk.color }}>
                    <span>⏱️</span>
                    <span className="font-medium">{t.alerts.timeToFailure}</span>
                  </div>
                  <p className="text-xl font-bold text-white mt-0.5">
                    ~{prediction.time_to_failure_hours}h
                  </p>
                </div>
              ) : null}
              
              <div className="glass rounded-lg p-2.5">
                <div className="flex items-center gap-1.5 text-green-400 text-sm">
                  <span>💰</span>
                  <span className="font-medium">{t.alerts.savings}</span>
                </div>
                <p className="text-xl font-bold text-white mt-0.5">
                  {formatCurrency(prediction.potential_savings)}
                </p>
              </div>
            </div>

            {/* Top contributing factors - Compact */}
            <div className="mt-4">
              <h4 className="text-xs font-medium text-gray-400 mb-2">{t.alerts.topContributingFactors}</h4>
              <div className="space-y-1.5">
                {prediction.contributing_factors.slice(0, 3).map((factor, index) => {
                  const factorColors = {
                    normal: 'bg-green-500',
                    warning: 'bg-amber-500',
                    critical: 'bg-red-500',
                  };
                  
                  // Translate factor name using sensorNames mapping
                  const getFactorName = (name: string) => {
                    // Convert display name to key format (e.g., "Motor Current" -> "motor_current")
                    const keyName = name.toLowerCase().replace(/\s+/g, '_').replace(/[()]/g, '');
                    return t.sensors.sensorNames[keyName] || name;
                  };
                  
                  return (
                    <motion.div
                      key={factor.name}
                      className="flex items-center gap-2"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <div className="flex-1">
                        <div className="flex justify-between text-xs mb-0.5">
                          <span className="text-gray-300">{getFactorName(factor.name)}</span>
                          <span className={`font-medium ${
                            factor.status === 'critical' ? 'text-red-400' :
                            factor.status === 'warning' ? 'text-amber-400' : 'text-green-400'
                          }`}>
                            {factor.value.toFixed(2)}
                          </span>
                        </div>
                        <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                          <motion.div
                            className={`h-full rounded-full ${factorColors[factor.status]}`}
                            initial={{ width: 0 }}
                            animate={{ width: `${factor.contribution * 100}%` }}
                            transition={{ duration: 0.8, delay: index * 0.1 }}
                          />
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>

            {/* Recommended actions - Compact */}
            <div className="mt-4 pt-3 border-t border-white/10">
              <h4 className="text-xs font-medium text-gray-400 mb-2">{t.alerts.recommendedActions}</h4>
              <div className="flex flex-wrap gap-1.5">
                {prediction.recommended_actions.map((action, index) => (
                  <motion.div
                    key={index}
                    className="flex items-center gap-1 px-2 py-1 rounded-full bg-white/5 text-xs text-gray-300"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                  >
                    <span className="text-ocp-green text-[10px]">✓</span>
                    {action}
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
// Compact alert banner
export function AlertBanner({ prediction }: { prediction: PredictionResult | null }) {
  const { t } = useI18n();
  
  if (!prediction || prediction.risk_level === 'low') return null;

  const risk = RISK_LEVELS[prediction.risk_level];
  
  // Get translated risk label
  const getRiskLabel = () => {
    switch (prediction.risk_level) {
      case 'low': return t.risk.low;
      case 'medium': return t.risk.medium;
      case 'high': return t.risk.high;
      case 'critical': return t.risk.critical;
      default: return risk.label;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      className="mb-4"
    >
      <div 
        className={`
          flex items-center justify-between px-4 py-3 rounded-lg
          ${prediction.risk_level === 'critical' ? 'alert-pulse' : ''}
        `}
        style={{
          background: risk.bgColor,
          border: `1px solid ${risk.borderColor}`,
        }}
      >
        <div className="flex items-center gap-3">
          <span className="text-xl">{risk.icon}</span>
          <div>
            <span className="font-medium" style={{ color: risk.color }}>
              {getRiskLabel()}:
            </span>{' '}
            <span className="text-gray-300">{risk.action}</span>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {prediction.dust_caused && (
            <span className="text-amber-400 text-sm">🌫️ {prediction.issue_type} {t.alerts.detected}</span>
          )}
          <span className="font-bold" style={{ color: risk.color }}>
            {Math.round(prediction.risk_probability * 100)}%
          </span>
        </div>
      </div>
    </motion.div>
  );
}
