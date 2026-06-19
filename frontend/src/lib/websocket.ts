'use client';

/**
 * SentinelIQ — WebSocket Hook
 * Connects to /stream/live and exposes the rolling feed of records/alerts.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import type { AnomalyAlert } from './api';

export interface StreamPayload {
  type: 'alert' | 'record';
  modality: 'log' | 'metric' | 'network';
  record: Record<string, any>;
  alert: AnomalyAlert | null;
}

const WS_URL =
  typeof window !== 'undefined'
    ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/stream/live`
    : '';

export function useLiveStream(maxBuffer = 200) {
  const [feed, setFeed] = useState<StreamPayload[]>([]);
  const [alerts, setAlerts] = useState<AnomalyAlert[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (!WS_URL) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      // auto-reconnect after a short delay
      setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
      try {
        const payload: StreamPayload = JSON.parse(event.data);
        setFeed((prev) => [payload, ...prev].slice(0, maxBuffer));
        if (payload.type === 'alert' && payload.alert) {
          setAlerts((prev) => [payload.alert as AnomalyAlert, ...prev].slice(0, maxBuffer));
        }
      } catch (e) {
        console.error('Failed to parse stream payload', e);
      }
    };
  }, [maxBuffer]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  return { feed, alerts, connected };
}
