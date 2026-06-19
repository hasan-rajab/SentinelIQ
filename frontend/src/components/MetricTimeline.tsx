'use client';

/**
 * SentinelIQ — Metric Timeline
 * The signature element: a live waveform of fused anomaly scores,
 * styled like an oscilloscope / radar sweep. This is the system's heartbeat.
 */

import { useEffect, useRef } from 'react';
import type { StreamPayload } from '@/lib/websocket';
import { severityColor } from '@/lib/api';

export default function MetricTimeline({ feed }: { feed: StreamPayload[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = '#1A212C';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (H / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    const points = [...feed].reverse().slice(-80);
    if (points.length < 2) return;

    const stepX = W / Math.max(points.length - 1, 1);

    // Draw the waveform line
    ctx.beginPath();
    ctx.strokeStyle = '#00FF9C';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';

    points.forEach((p, i) => {
      const score = p.alert?.fused_score ?? estimateScore(p);
      const y = H - score * H;
      const x = i * stepX;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Glow under the line
    const gradient = ctx.createLinearGradient(0, 0, 0, H);
    gradient.addColorStop(0, 'rgba(0, 255, 156, 0.15)');
    gradient.addColorStop(1, 'rgba(0, 255, 156, 0)');
    ctx.lineTo(W, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Mark anomaly spikes
    points.forEach((p, i) => {
      if (p.type === 'alert' && p.alert) {
        const x = i * stepX;
        const y = H - p.alert.fused_score * H;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = severityColor(p.alert.severity);
        ctx.fill();
      }
    });
  }, [feed]);

  return (
    <div className="relative w-full h-40 bg-base-800 rounded-xl border border-base-700 overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full" />
      <div className="absolute top-3 left-4 text-[10px] font-mono text-zinc-500 uppercase tracking-wider">
        Fused Anomaly Score
      </div>
      <div className="absolute top-3 right-4 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-pulse animate-pulse-slow" />
        <span className="text-[10px] font-mono text-pulse uppercase tracking-wider">Live</span>
      </div>
    </div>
  );
}

function estimateScore(p: StreamPayload): number {
  // For non-alert records, render a low ambient baseline so the line stays continuous
  return 0.08 + Math.random() * 0.06;
}
