"""Deterministic anomaly-type inference from observed telemetry only.

These rules deliberately ignore simulator-only ground-truth fields such as
``is_anomaly`` and ``anomaly_type``. The ML score decides whether a record is
anomalous; these rules only attach an interpretable incident category after
that decision has been made.
"""

from __future__ import annotations

from ipaddress import ip_address
from typing import Any, Mapping


def _number(record: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _is_private_ip(value: Any) -> bool:
    try:
        return ip_address(str(value)).is_private
    except ValueError:
        return False


def infer_anomaly_type(record: Mapping[str, Any], modality: str) -> str:
    """Infer an operational anomaly category without consulting labels."""
    if modality == "metric":
        return _infer_metric(record)
    if modality == "network":
        return _infer_network(record)
    if modality == "log":
        return _infer_log(record)
    return "unknown"


def _infer_metric(record: Mapping[str, Any]) -> str:
    process_count = _number(record, "process_count")
    open_connections = _number(record, "open_connections")
    disk_write = _number(record, "disk_write_mbps")
    disk_read = _number(record, "disk_read_mbps")
    net_out = _number(record, "net_out_mbps")
    mem = _number(record, "mem_percent")
    cpu = _number(record, "cpu_percent")

    if process_count >= 800:
        return "process_bomb"
    if open_connections >= 5000:
        return "connection_storm"
    if disk_write >= 300 or disk_read >= 100:
        return "disk_flood"
    if net_out >= 200:
        return "network_exfiltration"
    if mem >= 93:
        return "memory_leak"
    if cpu >= 90:
        return "cpu_spike"
    return "unknown"


def _infer_network(record: Mapping[str, Any]) -> str:
    src_ip = record.get("src_ip")
    dst_ip = record.get("dst_ip")
    dst_port = int(_number(record, "dst_port"))
    protocol = str(record.get("protocol", "")).upper()
    bytes_out = _number(record, "bytes_out")
    bytes_in = _number(record, "bytes_in")
    packets = _number(record, "packets")
    duration_ms = _number(record, "duration_ms")

    if bytes_out >= 10_000_000:
        return "data_exfiltration"
    if dst_port == 53 and protocol == "UDP" and packets >= 100 and bytes_out >= 1000:
        return "dns_tunneling"
    if _is_private_ip(src_ip) and _is_private_ip(dst_ip) and dst_port in {22, 445, 3389, 5985}:
        return "lateral_movement"
    if dst_port in {4444, 8888, 1337, 31337} or (
        240 <= duration_ms <= 270 and bytes_out <= 1000 and packets <= 20
    ):
        return "c2_beacon"
    if bytes_out <= 100 and bytes_in <= 1 and packets <= 2 and 1 <= dst_port <= 1024:
        return "port_scan"
    return "unknown"


def _infer_log(record: Mapping[str, Any]) -> str:
    message = str(record.get("message", "")).lower()

    if "failed password" in message:
        return "brute_force"
    if "invalid user" in message:
        return "invalid_user"
    if "../" in message or "etc/passwd" in message:
        return "path_traversal"
    if "union select" in message:
        return "sqli"
    if "cmd=" in message or "shell.php" in message:
        return "web_shell"
    if "command not allowed" in message or "failed su" in message:
        return "privilege_escalation"
    if "authentication failure" in message:
        return "auth_failure"
    if "/admin/" in message or "wp-admin" in message or "/api/v1/exec" in message:
        return "unauthorized_access"
    return "unknown"
