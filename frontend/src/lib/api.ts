/**
 * SentinelIQ — API Client
 * Thin fetch wrapper around the FastAPI backend.
 * Uses Next.js rewrites so calls go through /api/* in the browser.
 */

const BASE = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export interface AnomalyAlert {
  id: string;
  timestamp: string;
  source: string;
  modality: 'log' | 'metric' | 'network';
  anomaly_type: string;
  fused_score: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  is_acknowledged: boolean;
  mitre_tactic: string;
  mitre_tactic_id: string;
  mitre_technique: string;
  mitre_technique_id: string;
  description: string;
  recommended_action: string;
  top_features: string[];
  feature_values: Record<string, any>;
  shap_attribution: Record<string, number>;
  narrative?: string;
  raw_record: Record<string, any>;
}

export interface FederatedNodeStatus {
  node_id: number;
  status: string;
  last_round: number;
  last_loss: number;
  n_samples: number;
  n_anomalies: number;
}

export interface FederatedStatus {
  is_training: boolean;
  current_round: number;
  total_rounds: number;
  nodes: FederatedNodeStatus[];
  round_history: Array<{ round: number; n_clients: number; avg_loss: number; total_examples: number }>;
}

export interface StreamStats {
  total_records_processed: number;
  total_anomalies_detected: number;
  anomaly_rate: number;
  records_per_second: number;
  active_since: string;
}

export const api = {
  health: () => request<{ status: string; models_loaded: Record<string, boolean> }>('/health'),

  listAlerts: (params?: { limit?: number; severity?: string }) => {
    const qs = new URLSearchParams();
    if (params?.limit) qs.set('limit', String(params.limit));
    if (params?.severity) qs.set('severity', params.severity);
    return request<{ total: number; alerts: AnomalyAlert[] }>(`/alerts?${qs}`);
  },

  getAlert: (id: string) => request<AnomalyAlert>(`/alerts/${id}`),

  acknowledgeAlert: (id: string) =>
    request<{ alert_id: string; is_acknowledged: boolean; message: string }>('/alerts/acknowledge', {
      method: 'POST',
      body: JSON.stringify({ alert_id: id }),
    }),

  statsBySeverity: () => request<Record<string, number>>('/alerts/stats/by-severity'),
  statsByTactic: () => request<Record<string, number>>('/alerts/stats/by-tactic'),

  explainAlert: (id: string) => request<AnomalyAlert>(`/explain/${id}`),

  federatedStatus: () => request<FederatedStatus>('/federated/status'),

  streamStats: () => request<StreamStats>('/stream/stats'),
};

export function severityColor(severity: string): string {
  switch (severity) {
    case 'critical': return '#FF4757';
    case 'high': return '#FFA502';
    case 'medium': return '#3B82F6';
    case 'low': return '#6B7280';
    default: return '#6B7280';
  }
}
