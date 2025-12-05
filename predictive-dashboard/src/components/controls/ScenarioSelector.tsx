'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { SCENARIO_PRESETS } from '@/lib/constants';
import { SensorData, ScenarioPreset } from '@/types';
import { RISK_LEVELS } from '@/lib/constants';

interface ScenarioSelectorProps {
  currentScenario: string | null;
  onSelect: (preset: ScenarioPreset) => void;
}

export function ScenarioSelector({ currentScenario, onSelect }: ScenarioSelectorProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white flex items-center gap-2">
        <span>🎭</span> Demonstration Scenarios
      </h3>
      <p className="text-sm text-gray-400">
        Select a scenario to demonstrate how the AI predicts different failure conditions.
      </p>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {SCENARIO_PRESETS.map((preset, index) => {
          const risk = RISK_LEVELS[preset.expected_risk];
          const isActive = currentScenario === preset.id;
          
          return (
            <motion.button
              key={preset.id}
              onClick={() => onSelect(preset)}
              className={`
                p-4 rounded-xl text-left transition-all duration-300
                ${isActive 
                  ? 'ring-2 ring-ocp-green bg-ocp-green/10' 
                  : 'card hover:border-ocp-green/50'
                }
              `}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-start justify-between mb-2">
                <span className="text-2xl">{preset.icon}</span>
                <div 
                  className="px-2 py-0.5 rounded-full text-xs font-medium"
                  style={{ 
                    backgroundColor: risk.bgColor,
                    color: risk.color,
                    border: `1px solid ${risk.borderColor}`,
                  }}
                >
                  {preset.expected_risk.toUpperCase()}
                </div>
              </div>
              
              <h4 className="font-semibold text-white mb-1">{preset.name}</h4>
              <p className="text-xs text-gray-400 line-clamp-2">{preset.description}</p>
              
              {isActive && (
                <motion.div
                  className="mt-3 pt-3 border-t border-ocp-green/30"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                >
                  <span className="text-xs text-ocp-green font-medium flex items-center gap-1">
                    <span>✓</span> Active Scenario
                  </span>
                </motion.div>
              )}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}

// Quick scenario buttons for mobile
export function QuickScenarioBar({ 
  currentScenario, 
  onSelect 
}: { 
  currentScenario: string | null;
  onSelect: (preset: ScenarioPreset) => void;
}) {
  return (
    <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
      {SCENARIO_PRESETS.map((preset) => {
        const isActive = currentScenario === preset.id;
        
        return (
          <button
            key={preset.id}
            onClick={() => onSelect(preset)}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-full whitespace-nowrap transition-all
              ${isActive 
                ? 'bg-ocp-green text-white' 
                : 'bg-white/5 text-gray-300 hover:bg-white/10'
              }
            `}
          >
            <span>{preset.icon}</span>
            <span className="text-sm font-medium">{preset.name}</span>
          </button>
        );
      })}
    </div>
  );
}
