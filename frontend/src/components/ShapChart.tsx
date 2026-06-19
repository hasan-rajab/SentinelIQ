/**
 * SentinelIQ — SHAP Chart
 * Horizontal bar chart of feature attributions for an anomaly,
 * positive (pushing toward anomaly) in signal-red, negative in blue.
 */

import type { AnomalyAlert } from '@/lib/api';

export default function ShapChart({ alert }: { alert: AnomalyAlert }) {
  const entries = Object.entries(alert.shap_attribution || {})
    .filter(([, v]) => v !== 0)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 8);

  if (entries.length === 0) {
    return (
      <div className="rounded-xl border border-base-700 bg-base-800 p-5">
        <div className="text-xs font-mono uppercase tracking-wider text-zinc-400 mb-3">
          Feature Attribution
        </div>
        <p className="text-sm text-zinc-500">
          Top contributing features: {alert.top_features.join(', ') || 'n/a'}
        </p>
      </div>
    );
  }

  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)));

  return (
    <div className="rounded-xl border border-base-700 bg-base-800 p-5">
      <div className="text-xs font-mono uppercase tracking-wider text-zinc-400 mb-4">
        Feature Attribution (SHAP)
      </div>
      <div className="space-y-2.5">
        {entries.map(([feature, value]) => {
          const pct = (Math.abs(value) / maxAbs) * 100;
          const positive = value > 0;
          return (
            <div key={feature} className="flex items-center gap-3">
              <span className="text-xs font-mono text-zinc-400 w-32 truncate text-right">
                {feature}
              </span>
              <div className="flex-1 h-5 bg-base-700 rounded-sm overflow-hidden relative">
                <div
                  className="h-full rounded-sm transition-all"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: positive ? '#FF4757' : '#3B82F6',
                  }}
                />
              </div>
              <span className="text-xs font-mono text-zinc-500 w-16 tabular-nums">
                {value > 0 ? '+' : ''}
                {value.toFixed(4)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-4 mt-4 pt-3 border-t border-base-700">
        <div className="flex items-center gap-1.5 text-[10px] font-mono text-zinc-500">
          <span className="w-2 h-2 rounded-sm bg-[#FF4757]" /> increases anomaly score
        </div>
        <div className="flex items-center gap-1.5 text-[10px] font-mono text-zinc-500">
          <span className="w-2 h-2 rounded-sm bg-[#3B82F6]" /> decreases anomaly score
        </div>
      </div>
    </div>
  );
}
