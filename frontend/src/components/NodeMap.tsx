'use client';

/**
 * SentinelIQ — Node Map
 * Visualizes federated nodes as connected satellites around a central
 * server, showing per-node training status without exposing raw data.
 */

import { Server, HardDrive } from 'lucide-react';
import type { FederatedNodeStatus } from '@/lib/api';

export default function NodeMap({ nodes }: { nodes: FederatedNodeStatus[] }) {
  const angleStep = (2 * Math.PI) / Math.max(nodes.length, 1);
  const radius = 140;
  const center = 180;

  return (
    <div className="relative w-full" style={{ height: 360 }}>
      <svg viewBox="0 0 360 360" className="w-full h-full">
        {/* Connection lines */}
        {nodes.map((node, i) => {
          const angle = i * angleStep - Math.PI / 2;
          const x = center + radius * Math.cos(angle);
          const y = center + radius * Math.sin(angle);
          return (
            <line
              key={`line-${node.node_id}`}
              x1={center}
              y1={center}
              x2={x}
              y2={y}
              stroke="#252E3B"
              strokeWidth={1.5}
              strokeDasharray="4 4"
            />
          );
        })}

        {/* Center server */}
        <circle cx={center} cy={center} r={28} fill="#10151D" stroke="#00FF9C" strokeWidth={2} />
        <foreignObject x={center - 12} y={center - 12} width={24} height={24}>
          <Server className="w-6 h-6 text-pulse" />
        </foreignObject>

        {/* Nodes */}
        {nodes.map((node, i) => {
          const angle = i * angleStep - Math.PI / 2;
          const x = center + radius * Math.cos(angle);
          const y = center + radius * Math.sin(angle);
          const statusColor =
            node.status === 'training' ? '#00FF9C' : node.status === 'offline' ? '#FF4757' : '#6B7280';

          return (
            <g key={node.node_id}>
              <circle cx={x} cy={y} r={22} fill="#10151D" stroke={statusColor} strokeWidth={2} />
              <foreignObject x={x - 10} y={y - 10} width={20} height={20}>
                <HardDrive className="w-5 h-5" style={{ color: statusColor }} />
              </foreignObject>
              <text
                x={x}
                y={y + 38}
                textAnchor="middle"
                className="fill-zinc-400 text-[10px] font-mono"
              >
                Node {node.node_id}
              </text>
              <text
                x={x}
                y={y + 50}
                textAnchor="middle"
                className="fill-zinc-600 text-[9px] font-mono"
              >
                loss {node.last_loss.toFixed(4)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
