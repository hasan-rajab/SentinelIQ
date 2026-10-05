'use client';

/**
 * SentinelIQ — WebSocket Hook
 * Connects to /stream/live and exposes the rolling feed of records/alerts.
 */

import { useEffect, useRef, useState } from 'react';
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

  useEffect(() => {
    let stopped = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    function connect() {
      if (stopped || !WS_URL) return;
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => { if (!stopped) setConnected(true); };
      ws.onclose = () => {
        if (!stopped) {
          setConnected(false);
          retry = setTimeout(connect, 2000);
        }
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (event) => {
        if (stopped) return;
        try {
          const payload: StreamPayload = JSON.parse(event.data);
          setFeed((prev) => [payload, ...prev].slice(0, maxBuffer));
          if (payload.type === 'alert' && payload.alert) {
            setAlerts((prev) => [payload.alert as AnomalyAlert, ...prev].slice(0, maxBuffer));
          }
        } catch {
          console.error('Invalid stream payload');
        }
      };
    }
    connect();
    return () => {
      stopped = true;
      if (retry) clearTimeout(retry);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [maxBuffer]);

  return { feed, alerts, connected };
}
