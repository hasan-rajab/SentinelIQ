"""
SentinelIQ — Network Feature Engineering
Derives rate/ratio features from raw NetFlow fields that directly target
the attack signatures in the simulator:
  - port_scan: near-zero bytes, tiny duration → near-zero rate
  - c2_beacon: small, regular outbound, suspiciously consistent duration
  - data_exfiltration: massive one-way outbound transfer
  - dns_tunneling: high packet count relative to bytes (small payloads, many queries)
  - lateral_movement: moderate bidirectional bytes on admin ports

Raw tabular features (bytes_out, bytes_in, packets, duration_ms, ports)
don't capture these *shapes* well in isolation — ratios and rates do.
Used identically at training time and live inference time so the model
never sees a feature distribution it wasn't trained on.
"""

import pandas as pd
import numpy as np


ENGINEERED_FEATURES = [
    "bytes_out",
    "bytes_in",
    "packets",
    "duration_ms",
    "src_port",
    "dst_port",
    "bytes_out_in_ratio",
    "bytes_per_sec",
    "bytes_per_packet",
    "is_common_port",
]

_COMMON_PORTS = {80, 443, 53, 22, 25, 587, 8080, 3306, 5432, 6379}


def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered columns to a network flow dataframe (or single-row
    dataframe for live inference). Safe against zero-division and
    missing columns — always returns all ENGINEERED_FEATURES columns.
    """
    df = df.copy()

    bytes_out = df.get("bytes_out", pd.Series(0, index=df.index)).fillna(0)
    bytes_in = df.get("bytes_in", pd.Series(0, index=df.index)).fillna(0)
    packets = df.get("packets", pd.Series(0, index=df.index)).fillna(0)
    duration_ms = df.get("duration_ms", pd.Series(0, index=df.index)).fillna(0)
    dst_port = df.get("dst_port", pd.Series(0, index=df.index)).fillna(0)

    # Outbound/inbound ratio — exfiltration and beaconing are heavily
    # outbound-skewed; normal traffic is usually more balanced or inbound-heavy
    df["bytes_out_in_ratio"] = bytes_out / (bytes_in + 1)

    # Transfer rate — exfiltration is a sustained high-rate transfer,
    # port scans are near-instant near-zero-byte probes
    df["bytes_per_sec"] = (bytes_out + bytes_in) / ((duration_ms / 1000) + 0.01)

    # Bytes per packet — DNS tunneling shows many packets carrying little
    # payload each; normal large transfers show high bytes/packet
    df["bytes_per_packet"] = (bytes_out + bytes_in) / (packets + 1)

    # Common port flag — most attacks in this simulator use uncommon or
    # high ports (admin ports, listener ports like 4444/1337) while normal
    # traffic clusters on well-known service ports
    df["is_common_port"] = dst_port.isin(_COMMON_PORTS).astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame, feature_cols: list = None) -> np.ndarray:
    """Convenience helper: engineer features then extract as a numpy array."""
    feature_cols = feature_cols or ENGINEERED_FEATURES
    enriched = add_network_features(df)
    missing = [c for c in feature_cols if c not in enriched.columns]
    if missing:
        raise ValueError(f"Missing feature columns after engineering: {missing}")
    return enriched[feature_cols].fillna(0).values