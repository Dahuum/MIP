import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ocp: {
          green: "#00843D",
          "green-dark": "#006B31",
          "green-light": "#4CAF50",
          gold: "#C4A000",
          "gold-light": "#FFD700",
          dark: "#1a1a2e",
          darker: "#16162a",
          navy: "#0f0f23",
        },
        risk: {
          low: "#22c55e",
          medium: "#f59e0b",
          high: "#ef4444",
          critical: "#dc2626",
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "ocp-gradient": "linear-gradient(135deg, #00843D 0%, #006B31 50%, #004D26 100%)",
        "ocp-dark": "linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%)",
        "card-gradient": "linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "spin-slow": "spin 20s linear infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        glow: {
          "0%": { boxShadow: "0 0 20px rgba(0, 132, 61, 0.3)" },
          "100%": { boxShadow: "0 0 40px rgba(0, 132, 61, 0.6)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
