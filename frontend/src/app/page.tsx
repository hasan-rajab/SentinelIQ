'use client';

import { useState, useMemo } from 'react';
import { useLiveStream, type StreamPayload } from '@/lib/websocket';
import AnomalyFeed from '@/components/AnomalyFeed';
import MetricTimeline from '@/components/MetricTimeline';
import MitreCard from '@/components/MitreCard';
import ShapChart from '@/components/ShapChart';
import { AlertTriangle, Radio, TrendingUp } from 'lucide-react';

export default function DashboardPage() {
  const { feed, alerts, connected } = useLiveStream(200);
  const [selected, setSelected] = useState<StreamPayload | null>(null);

  const stats = useMemo(() => {
    const total = feed.length;
    const anomalies = alerts.length;
    const critical = alerts.filter((a) => a.severity === 'critical').length;
    return { total, anomalies, critical, rate: total ? ((anomalies / total) * 100).toFixed(1) : '0' };
  }, [feed, alerts]);

  const activeAlert = selected?.alert;

  return (
    <div className="flex h-screen">
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="border-b border-base-700 px-6 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Live Anomaly Feed</h1>
            <p className="text-xs text-zinc-500 font-mono mt-0.5">
              Multimodal detection — IT Ops &amp; Cybersecurity
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Radio className={`w-4 h-4 ${connected ? 'text-pulse' : 'text-zinc-600'}`} />
            <span className="text-xs font-mono text-zinc-400">
              {connected ? 'connected' : 'reconnecting…'}
            </span>
          </div>
        </header>

        {/* Stats row */}
        <div className="grid grid-cols-4 gap-px bg-base-700 shrink-0">
          <StatCard label="Records / window" value={stats.total} />
          <StatCard label="Anomalies detected" value={stats.anomalies} accent="#FFA502" />
          <StatCard label="Critical" value={stats.critical} accent="#FF4757" />
          <StatCard label="Anomaly rate" value={`${stats.rate}%`} />
        </div>

        {/* Waveform */}
        <div className="p-6 shrink-0">
          <MetricTimeline feed={feed} />
        </div>

        {/* Feed + Detail split */}
        <div className="flex-1 flex min-h-0">
          <div className="w-full lg:w-[420px] border-r border-base-700 overflow-y-auto">
            <AnomalyFeed feed={feed} onSelect={setSelected} />
          </div>

          <div className="hidden lg:flex flex-1 overflow-y-auto p-6">
            {activeAlert ? (
              <div className="w-full max-w-xl space-y-4">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-signal-high" />
                  <h2 className="text-base font-semibold text-zinc-100 capitalize">
                    {activeAlert.anomaly_type.replace(/_/g, ' ')}
                  </h2>
                </div>
                <p className="text-sm text-zinc-400 font-mono">{activeAlert.source}</p>
                <MitreCard alert={activeAlert} />
                <ShapChart alert={activeAlert} />
                {activeAlert.narrative && (
                  <div className="rounded-xl border border-base-700 bg-base-800 p-5">
                    <div className="text-xs font-mono uppercase tracking-wider text-zinc-400 mb-2">
                      Narrative
                    </div>
                    <p className="text-sm text-zinc-300 leading-relaxed">{activeAlert.narrative}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="w-full flex flex-col items-center justify-center text-zinc-600 gap-2">
                <TrendingUp className="w-10 h-10" />
                <p className="text-sm">Select an alert to view detail</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, accent }: { label: string; value: string | number; accent?: string }) {
  return (
    <div className="bg-base-900 px-5 py-4">
      <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 mb-1">{label}</div>
      <div
        className="text-2xl font-mono font-semibold tabular-nums"
        style={{ color: accent || '#E5E9F0' }}
      >
        {value}
      </div>
    </div>
  );
}
