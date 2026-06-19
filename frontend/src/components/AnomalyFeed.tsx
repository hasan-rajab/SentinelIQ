'use client';

/**
 * SentinelIQ — Anomaly Feed
 * Live-streaming list of incoming records, with anomalous ones
 * highlighted by severity. Clicking an alert opens its detail.
 */

import { AlertTriangle, FileText, Activity, Network } from 'lucide-react';
import type { StreamPayload } from '@/lib/websocket';
import { severityColor } from '@/lib/api';
import { formatDistanceToNow } from 'date-fns';

const MODALITY_ICON: Record<string, React.ReactNode> = {
  log: <FileText className="w-3.5 h-3.5" />,
  metric: <Activity className="w-3.5 h-3.5" />,
  network: <Network className="w-3.5 h-3.5" />,
};

export default function AnomalyFeed({
  feed,
  onSelect,
}: {
  feed: StreamPayload[];
  onSelect?: (payload: StreamPayload) => void;
}) {
  if (feed.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-600 gap-2 py-16">
        <Activity className="w-8 h-8 animate-pulse-slow" />
        <p className="text-sm font-mono">Waiting for stream connection…</p>
      </div>
    );
  }

  return (
    <div className="divide-y divide-base-700">
      {feed.map((item, idx) => {
        const isAlert = item.type === 'alert' && item.alert;
        const color = isAlert ? severityColor(item.alert!.severity) : '#252E3B';

        return (
          <button
            key={idx}
            onClick={() => onSelect?.(item)}
            className={`w-full text-left px-4 py-3 flex items-start gap-3 transition-colors hover:bg-base-800 ${
              isAlert ? '' : 'opacity-50'
            }`}
          >
            <div
              className="mt-0.5 w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: color, boxShadow: isAlert ? `0 0 8px ${color}` : 'none' }}
            />

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 text-xs text-zinc-500 font-mono mb-0.5">
                <span className="flex items-center gap-1">
                  {MODALITY_ICON[item.modality]}
                  {item.modality}
                </span>
                <span>·</span>
                <span>{item.record.source || item.record.host || item.record.src_ip || 'unknown'}</span>
              </div>

              <p className={`text-sm truncate ${isAlert ? 'text-zinc-100' : 'text-zinc-500'}`}>
                {isAlert
                  ? item.alert!.anomaly_type.replace(/_/g, ' ')
                  : item.record.message?.slice(0, 60) || 'normal activity'}
              </p>

              {isAlert && (
                <div className="flex items-center gap-2 mt-1">
                  <AlertTriangle className="w-3 h-3" style={{ color }} />
                  <span className="text-[11px] font-mono uppercase tracking-wide" style={{ color }}>
                    {item.alert!.severity}
                  </span>
                  <span className="text-[11px] font-mono text-zinc-600">
                    score {item.alert!.fused_score.toFixed(3)}
                  </span>
                </div>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
