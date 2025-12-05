import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Format number with commas
export function formatNumber(num: number): string {
  return new Intl.NumberFormat("en-US").format(num);
}

// Format currency
export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
}

// Format percentage
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

// Get risk level from probability
export function getRiskLevel(probability: number): 'low' | 'medium' | 'high' | 'critical' {
  if (probability >= 0.7) return 'critical';
  if (probability >= 0.5) return 'high';
  if (probability >= 0.3) return 'medium';
  return 'low';
}

// Get risk color
export function getRiskColor(level: string): string {
  switch (level) {
    case 'critical':
      return '#dc2626';
    case 'high':
      return '#ef4444';
    case 'medium':
      return '#f59e0b';
    case 'low':
    default:
      return '#22c55e';
  }
}

// Get risk gradient
export function getRiskGradient(level: string): string {
  switch (level) {
    case 'critical':
      return 'linear-gradient(135deg, #dc2626, #991b1b)';
    case 'high':
      return 'linear-gradient(135deg, #ef4444, #dc2626)';
    case 'medium':
      return 'linear-gradient(135deg, #f59e0b, #d97706)';
    case 'low':
    default:
      return 'linear-gradient(135deg, #22c55e, #16a34a)';
  }
}

// Get status icon
export function getStatusIcon(level: string): string {
  switch (level) {
    case 'critical':
      return '🚨';
    case 'high':
      return '⚠️';
    case 'medium':
      return '⚡';
    case 'low':
    default:
      return '✅';
  }
}

// Calculate sensor status
export function getSensorStatus(
  value: number,
  thresholds: { warning: number; critical: number; isHighBad?: boolean }
): 'normal' | 'warning' | 'critical' {
  const { warning, critical, isHighBad = true } = thresholds;
  
  if (isHighBad) {
    if (value >= critical) return 'critical';
    if (value >= warning) return 'warning';
  } else {
    if (value <= critical) return 'critical';
    if (value <= warning) return 'warning';
  }
  return 'normal';
}

// Generate time series labels
export function generateTimeLabels(count: number, intervalMinutes: number = 5): string[] {
  const labels: string[] = [];
  const now = new Date();
  
  for (let i = count - 1; i >= 0; i--) {
    const time = new Date(now.getTime() - i * intervalMinutes * 60000);
    labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
  }
  
  return labels;
}

// Simulate sensor noise
export function addNoise(value: number, noisePercent: number = 0.02): number {
  const noise = (Math.random() - 0.5) * 2 * noisePercent * value;
  return value + noise;
}

// Clamp value between min and max
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

// Calculate exponential moving average
export function ema(data: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const emaArray: number[] = [data[0]];
  
  for (let i = 1; i < data.length; i++) {
    emaArray.push(data[i] * k + emaArray[i - 1] * (1 - k));
  }
  
  return emaArray;
}

// Format relative time
export function formatRelativeTime(date: Date | string): string {
  const now = new Date();
  const past = new Date(date);
  const diffMs = now.getTime() - past.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return past.toLocaleDateString();
}

// Debounce function
export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}
