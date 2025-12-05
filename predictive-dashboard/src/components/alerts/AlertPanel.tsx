'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RISK_LEVELS } from '@/lib/constants';
import { PredictionResult } from '@/types';
import { formatCurrency } from '@/lib/utils';

interface AlertPanelProps {
  prediction: PredictionResult | null;
  onDismiss?: () => void;
}

export function AlertPanel({ prediction, onDismiss }: AlertPanelProps) {
  if (!prediction) return null;

  const risk = RISK_LEVELS[prediction.risk_level];
  const isCritical = prediction.risk_level === 'critical' || prediction.risk_level === 'high';

  return (
    <AnimatePresence>
      {prediction && (
        <motion.div
          initial={{ opacity: 0, y: -20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -20, scale: 0.95 }}
          className={`
            relative overflow-hidden rounded-2xl p-6
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
                  className="text-4xl"
                  animate={isCritical ? { scale: [1, 1.2, 1] } : {}}
                  transition={{ duration: 0.5, repeat: isCritical ? Infinity : 0 }}
                >
                  {risk.icon}
                </motion.span>
                <div>
                  <h3 className="text-xl font-bold" style={{ color: risk.color }}>
                    {risk.label}
                  </h3>
                  <p className="text-gray-300">
                    {prediction.diagnostic_message || risk.description}
                  </p>
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-3xl font-bold" style={{ color: risk.color }}>
                  {Math.round(prediction.risk_probability * 100)}%
                </div>
                <div className="text-sm text-gray-400">Risk Probability</div>
              </div>
            </div>

            {/* Key insights */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              {prediction.dust_caused && (
                <div className="glass rounded-lg p-3">
                  <div className="flex items-center gap-2 text-amber-400">
                    <span>🌫️</span>
                    <span className="font-medium">Dust Detected</span>
                  </div>
                  <p className="text-sm text-gray-400 mt-1">
                    {Math.round(prediction.dust_probability * 100)}% probability dust is causing imbalance
                  </p>
                </div>
              )}
              
              {prediction.time_to_failure_hours && (
                <div className="glass rounded-lg p-3">
                  <div className="flex items-center gap-2" style={{ color: risk.color }}>
                    <span>⏱️</span>
                    <span className="font-medium">Time to Failure</span>
                  </div>
                  <p className="text-2xl font-bold text-white mt-1">
                    ~{prediction.time_to_failure_hours}h
                  </p>
                </div>
              )}
              
              <div className="glass rounded-lg p-3">
                <div className="flex items-center gap-2 text-green-400">
                  <span>💰</span>
                  <span className="font-medium">Potential Savings</span>
                </div>
                <p className="text-2xl font-bold text-white mt-1">
                  {formatCurrency(prediction.potential_savings)}
                </p>
              </div>
            </div>

            {/* Top contributing factors */}
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-400 mb-3">Top Contributing Factors</h4>
              <div className="space-y-2">
                {prediction.contributing_factors.slice(0, 3).map((factor, index) => {
                  const factorColors = {
                    normal: 'bg-green-500',
                    warning: 'bg-amber-500',
                    critical: 'bg-red-500',
                  };
                  
                  return (
                    <motion.div
                      key={factor.name}
                      className="flex items-center gap-3"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <div className="flex-1">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-300">{factor.name}</span>
                          <span className={`font-medium ${
                            factor.status === 'critical' ? 'text-red-400' :
                            factor.status === 'warning' ? 'text-amber-400' : 'text-green-400'
                          }`}>
                            {factor.value.toFixed(2)}
                          </span>
                        </div>
                        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
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

            {/* Recommended actions */}
            <div className="mt-6 pt-4 border-t border-white/10">
              <h4 className="text-sm font-medium text-gray-400 mb-3">Recommended Actions</h4>
              <div className="flex flex-wrap gap-2">
                {prediction.recommended_actions.map((action, index) => (
                  <motion.div
                    key={index}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 text-sm text-gray-300"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                  >
                    <span className="text-ocp-green">✓</span>
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
  if (!prediction || prediction.risk_level === 'low') return null;

  const risk = RISK_LEVELS[prediction.risk_level];

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
              {risk.label}:
            </span>{' '}
            <span className="text-gray-300">{risk.action}</span>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {prediction.dust_caused && (
            <span className="text-amber-400 text-sm">🌫️ Dust Detected</span>
          )}
          <span className="font-bold" style={{ color: risk.color }}>
            {Math.round(prediction.risk_probability * 100)}%
          </span>
        </div>
      </div>
    </motion.div>
  );
}
