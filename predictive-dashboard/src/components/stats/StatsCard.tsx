'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { formatCurrency, formatNumber } from '@/lib/utils';

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: {
    value: number;
    label: string;
    positive?: boolean;
  };
  icon: string;
  color?: string;
  delay?: number;
}

export function StatsCard({ title, value, change, icon, color = '#00843D', delay = 0 }: StatsCardProps) {
  return (
    <motion.div
      className="card p-4 relative overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      whileHover={{ scale: 1.02 }}
    >
      {/* Background glow */}
      <div 
        className="absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl opacity-20"
        style={{ backgroundColor: color }}
      />
      
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-3">
          <span className="text-2xl">{icon}</span>
          {change && (
            <div className={`
              flex items-center gap-1 text-xs px-2 py-0.5 rounded-full
              ${change.positive ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}
            `}>
              <span>{change.positive ? '↑' : '↓'}</span>
              <span>{change.value}%</span>
            </div>
          )}
        </div>
        
        <h3 className="text-gray-400 text-sm mb-1">{title}</h3>
        <p className="text-2xl font-bold text-white">{value}</p>
        
        {change && (
          <p className="text-xs text-gray-500 mt-1">{change.label}</p>
        )}
      </div>
    </motion.div>
  );
}

interface DashboardStatsProps {
  stats: {
    failuresPrevented: number;
    totalSavings: number;
    modelAccuracy: number;
    uptime: number;
    daysSinceFailure: number;
  };
}

export function DashboardStats({ stats }: DashboardStatsProps) {
  const cards = [
    {
      title: 'Failures Prevented',
      value: formatNumber(stats.failuresPrevented),
      icon: '🛡️',
      color: '#22c55e',
      change: { value: 15, label: 'vs last month', positive: true },
    },
    {
      title: 'Total Savings',
      value: formatCurrency(stats.totalSavings),
      icon: '💰',
      color: '#C4A000',
      change: { value: 23, label: 'vs projection', positive: true },
    },
    {
      title: 'Model Recall',
      value: `${(stats.modelAccuracy * 100).toFixed(0)}%`,
      icon: '🎯',
      color: '#00843D',
    },
    {
      title: 'System Uptime',
      value: `${stats.uptime.toFixed(1)}%`,
      icon: '⚡',
      color: '#3b82f6',
    },
    {
      title: 'Days Since Failure',
      value: formatNumber(stats.daysSinceFailure),
      icon: '📅',
      color: '#8b5cf6',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
      {cards.map((card, index) => (
        <StatsCard key={card.title} {...card} delay={index * 0.1} />
      ))}
    </div>
  );
}

// ROI Calculator Component
interface ROICalculatorProps {
  failuresCaught: number;
  costPerFailure?: number;
}

export function ROICalculator({ failuresCaught, costPerFailure = 200000 }: ROICalculatorProps) {
  const totalSavings = failuresCaught * costPerFailure;
  const implementationCost = 90000;
  const roi = ((totalSavings - implementationCost) / implementationCost) * 100;
  const paybackMonths = (implementationCost / totalSavings) * 12;

  return (
    <motion.div
      className="card p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <span>💰</span> Return on Investment
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="text-center p-3 rounded-lg bg-green-500/10 border border-green-500/30">
          <p className="text-3xl font-bold text-green-400">{failuresCaught}</p>
          <p className="text-xs text-gray-400">Failures Caught</p>
        </div>
        
        <div className="text-center p-3 rounded-lg bg-ocp-gold/10 border border-ocp-gold/30">
          <p className="text-3xl font-bold text-ocp-gold">{formatCurrency(totalSavings)}</p>
          <p className="text-xs text-gray-400">Total Savings</p>
        </div>
        
        <div className="text-center p-3 rounded-lg bg-blue-500/10 border border-blue-500/30">
          <p className="text-3xl font-bold text-blue-400">{roi.toFixed(0)}%</p>
          <p className="text-xs text-gray-400">ROI</p>
        </div>
        
        <div className="text-center p-3 rounded-lg bg-purple-500/10 border border-purple-500/30">
          <p className="text-3xl font-bold text-purple-400">{paybackMonths.toFixed(1)}mo</p>
          <p className="text-xs text-gray-400">Payback Period</p>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Cost per unplanned failure:</span>
          <span className="text-white font-medium">{formatCurrency(costPerFailure)}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Implementation cost:</span>
          <span className="text-white font-medium">{formatCurrency(implementationCost)}</span>
        </div>
        <div className="flex justify-between text-sm border-t border-white/10 pt-2 mt-2">
          <span className="text-gray-400">Net benefit:</span>
          <span className="text-green-400 font-bold">
            {formatCurrency(totalSavings - implementationCost)}
          </span>
        </div>
      </div>
    </motion.div>
  );
}
