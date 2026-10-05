/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        base: {
          950: '#070A0F',
          900: '#0A0E14',
          800: '#10151D',
          700: '#1A212C',
          600: '#252E3B',
        },
        signal: {
          critical: '#FF4757',
          high: '#FFA502',
          medium: '#3B82F6',
          low: '#6B7280',
        },
        pulse: '#00FF9C',
        pulseDim: '#00B377',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['Inter', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan': 'scan 4s linear infinite',
      },
      keyframes: {
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
      },
    },
  },
  plugins: [],
};