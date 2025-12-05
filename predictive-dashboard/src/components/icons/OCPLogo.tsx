'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Image from 'next/image';

interface OCPLogoProps {
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
}

export function OCPLogo({ size = 'md', showText = true }: OCPLogoProps) {
  const sizes = {
    sm: { logo: 36, text: 'text-lg' },
    md: { logo: 52, text: 'text-xl' },
    lg: { logo: 72, text: 'text-2xl' },
  };

  return (
    <div className="flex items-center gap-3">
      <motion.div
        className="relative"
        whileHover={{ scale: 1.05 }}
        transition={{ type: 'spring', stiffness: 400 }}
      >
        {/* OCP Official Logo */}
        <Image
          src="/ocp-logo.png"
          alt="OCP Group Logo"
          width={sizes[size].logo}
          height={sizes[size].logo}
          className="object-contain"
          priority
        />
      </motion.div>

      {showText && (
        <div className="flex flex-col">
          <span className={`font-bold ${sizes[size].text} gradient-text`}>
            OCP GROUP
          </span>
          <span className="text-xs text-gray-400">Predictive Maintenance</span>
        </div>
      )}
    </div>
  );
}

export function FanIcon({ 
  spinning = false, 
  speed = 'normal',
  size = 64,
  status = 'normal'
}: { 
  spinning?: boolean; 
  speed?: 'slow' | 'normal' | 'fast';
  size?: number;
  status?: 'normal' | 'warning' | 'critical';
}) {
  const speedClass = {
    slow: 'fan-slow',
    normal: 'fan-spinning',
    fast: 'fan-warning',
  };

  const statusColors = {
    normal: '#22c55e',
    warning: '#f59e0b',
    critical: '#ef4444',
  };

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg
        viewBox="0 0 100 100"
        className={spinning ? speedClass[speed] : ''}
        style={{ width: size, height: size }}
      >
        {/* Fan blades */}
        <g fill={statusColors[status]} opacity="0.9">
          {/* Blade 1 */}
          <path d="M50 50 Q60 25 50 10 Q40 25 50 50" />
          {/* Blade 2 */}
          <path d="M50 50 Q75 60 90 50 Q75 40 50 50" />
          {/* Blade 3 */}
          <path d="M50 50 Q60 75 50 90 Q40 75 50 50" />
          {/* Blade 4 */}
          <path d="M50 50 Q25 60 10 50 Q25 40 50 50" />
        </g>
        
        {/* Center hub */}
        <circle cx="50" cy="50" r="10" fill={statusColors[status]} />
        <circle cx="50" cy="50" r="6" fill="#1a1a2e" />
      </svg>

      {/* Status glow */}
      <div 
        className="absolute inset-0 rounded-full blur-xl opacity-40"
        style={{ backgroundColor: statusColors[status] }}
      />
    </div>
  );
}

export function PhosphateIcon({ size = 48 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 100 100"
      style={{ width: size, height: size }}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Phosphate rock representation */}
      <defs>
        <linearGradient id="phosphateGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#00843D" />
          <stop offset="100%" stopColor="#C4A000" />
        </linearGradient>
      </defs>
      
      <path
        d="M20 80 Q10 70 15 55 Q20 40 35 35 Q50 25 65 35 Q80 40 85 55 Q90 70 80 80 Z"
        fill="url(#phosphateGrad)"
        opacity="0.8"
      />
      
      {/* Crystalline structure */}
      <path
        d="M35 55 L50 40 L65 55 L50 70 Z"
        fill="white"
        opacity="0.6"
      />
      <circle cx="50" cy="55" r="8" fill="white" opacity="0.8" />
    </svg>
  );
}