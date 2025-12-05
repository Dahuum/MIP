'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { SCENARIO_PRESETS } from '@/lib/constants';
import { SensorData, ScenarioPreset } from '@/types';
import { RISK_LEVELS } from '@/lib/constants';
import { useI18n } from '@/lib/i18n';

interface ScenarioSelectorProps {
  currentScenario: string | null;
  onSelect: (preset: ScenarioPreset) => void;
}

export function ScenarioSelector({ currentScenario, onSelect }: ScenarioSelectorProps) {
  const { t } = useI18n();
  
  // Function to get translated scenario name
  const getScenarioName = (id: string, fallbackName: string) => {
    return t.scenarios.scenarioNames[id] || fallbackName;
  };
  
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <span>🎭</span> {t.scenarios.demonstrationScenarios}
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            {t.scenarios.selectToSimulate}
          </p>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-xs">
          <span className="flex items-center gap-1 px-2 py-1 rounded bg-green-500/10 text-green-400">🟢 {t.risk.low}</span>
          <span className="flex items-center gap-1 px-2 py-1 rounded bg-amber-500/10 text-amber-400">🟡 {t.risk.medium}</span>
          <span className="flex items-center gap-1 px-2 py-1 rounded bg-red-500/10 text-red-400">🔴 {t.risk.high}/{t.risk.critical}</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {SCENARIO_PRESETS.map((preset, index) => {
          const risk = RISK_LEVELS[preset.expected_risk];
          const isActive = currentScenario === preset.id;
          
          return (
            <motion.button
              key={preset.id}
              onClick={() => onSelect(preset)}
              className={`
                relative p-4 rounded-xl text-left transition-all duration-300 h-[120px] flex flex-col
                ${isActive 
                  ? 'ring-2 ring-ocp-green bg-ocp-green/10' 
                  : 'bg-white/5 border border-white/10 hover:border-ocp-green/50 hover:bg-white/[0.07]'
                }
              `}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              {/* Risk indicator dot */}
              <div 
                className="absolute top-3 right-3 w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: risk.color }}
              />
              
              {/* Icon */}
              <span className="text-xl mb-2">{preset.icon}</span>
              
              {/* Title */}
              <h4 className="font-medium text-white text-sm leading-tight flex-1">{getScenarioName(preset.id, preset.name)}</h4>
              
              {/* Active indicator */}
              {isActive && (
                <div className="mt-2 flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-ocp-green animate-pulse" />
                  <span className="text-[10px] text-ocp-green font-medium uppercase">{t.scenarios.active}</span>
                </div>
              )}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}

// Quick scenario buttons - horizontal scroll bar
export function QuickScenarioBar({ 
  currentScenario, 
  onSelect 
}: { 
  currentScenario: string | null;
  onSelect: (preset: ScenarioPreset) => void;
}) {
  const { t } = useI18n();
  
  // Function to get translated scenario name
  const getScenarioName = (id: string, fallbackName: string) => {
    return t.scenarios.scenarioNames[id] || fallbackName;
  };
  
  return (
    <div className="relative">
      <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide px-1">
        {SCENARIO_PRESETS.map((preset) => {
          const risk = RISK_LEVELS[preset.expected_risk];
          const isActive = currentScenario === preset.id;
          
          return (
            <button
              key={preset.id}
              onClick={() => onSelect(preset)}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-full whitespace-nowrap transition-all text-sm
                ${isActive 
                  ? 'bg-ocp-green text-white shadow-lg shadow-ocp-green/20' 
                  : 'bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10 hover:border-white/20'
                }
              `}
            >
              <span className="text-base">{preset.icon}</span>
              <span className="font-medium">{getScenarioName(preset.id, preset.name)}</span>
              {!isActive && (
                <span 
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: risk.color }}
                />
              )}
            </button>
          );
        })}
      </div>
      {/* Fade edges */}
      <div className="absolute right-0 top-0 bottom-2 w-8 bg-gradient-to-l from-ocp-dark to-transparent pointer-events-none" />
    </div>
  );
}
