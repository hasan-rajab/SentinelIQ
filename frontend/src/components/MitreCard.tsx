/**
 * SentinelIQ — MITRE Card
 * Displays the MITRE ATT&CK tactic + technique mapping for an alert,
 * styled like a classification stamp.
 */

import { Target, Shield } from 'lucide-react';
import { severityColor } from '@/lib/api';
import type { AnomalyAlert } from '@/lib/api';

export default function MitreCard({ alert }: { alert: AnomalyAlert }) {
  const color = severityColor(alert.severity);

  return (
    <div
      className="rounded-xl border bg-base-800 p-5"
      style={{ borderColor: `${color}40` }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Target className="w-4 h-4" style={{ color }} />
          <span className="text-xs font-mono uppercase tracking-wider text-zinc-400">
            MITRE ATT&amp;CK
          </span>
        </div>
        <span
          className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full border"
          style={{ color, borderColor: color }}
        >
          {alert.severity}
        </span>
      </div>

      <div className="space-y-3">
        <div>
          <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider mb-1">Tactic</div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-100">{alert.mitre_tactic}</span>
            <span className="text-xs font-mono text-zinc-500">{alert.mitre_tactic_id}</span>
          </div>
        </div>

        <div>
          <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider mb-1">Technique</div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-100">{alert.mitre_technique}</span>
            <span className="text-xs font-mono text-zinc-500">{alert.mitre_technique_id}</span>
          </div>
        </div>

        <div className="pt-3 border-t border-base-700">
          <p className="text-sm text-zinc-400 leading-relaxed">{alert.description}</p>
        </div>

        <div className="flex items-start gap-2 pt-2">
          <Shield className="w-3.5 h-3.5 text-pulse mt-0.5 shrink-0" />
          <p className="text-sm text-zinc-300 leading-relaxed">{alert.recommended_action}</p>
        </div>
      </div>
    </div>
  );
}
