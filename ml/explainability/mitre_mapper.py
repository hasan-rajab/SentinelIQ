"""
SentinelIQ — MITRE ATT&CK Mapper
Maps detected anomaly patterns to MITRE ATT&CK tactics and techniques.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class MitreMapping:
    tactic: str
    tactic_id: str
    technique: str
    technique_id: str
    description: str
    severity: str          # low / medium / high / critical
    recommended_action: str


# ── Mapping table ─────────────────────────────────────────────────────────────
MITRE_MAP = {
    # Log anomaly types
    "brute_force": MitreMapping(
        tactic="Credential Access",
        tactic_id="TA0006",
        technique="Brute Force",
        technique_id="T1110",
        description="Multiple failed authentication attempts detected, indicating a brute force attack.",
        severity="high",
        recommended_action="Block source IP, enforce account lockout policy, enable MFA.",
    ),
    "privilege_escalation": MitreMapping(
        tactic="Privilege Escalation",
        tactic_id="TA0004",
        technique="Sudo and Sudo Caching",
        technique_id="T1548.003",
        description="Unauthorized privilege escalation attempt detected via sudo or su.",
        severity="critical",
        recommended_action="Audit sudoers file, review PAM configuration, alert security team.",
    ),
    "invalid_user": MitreMapping(
        tactic="Reconnaissance",
        tactic_id="TA0043",
        technique="Active Scanning",
        technique_id="T1595",
        description="Login attempts using non-existent usernames, indicating user enumeration.",
        severity="medium",
        recommended_action="Enable fail2ban, review SSH configuration, monitor for patterns.",
    ),
    "auth_failure": MitreMapping(
        tactic="Credential Access",
        tactic_id="TA0006",
        technique="Valid Accounts",
        technique_id="T1078",
        description="Authentication failure with valid username, possible credential stuffing.",
        severity="high",
        recommended_action="Reset credentials, enforce MFA, review account access logs.",
    ),
    "unauthorized_access": MitreMapping(
        tactic="Initial Access",
        tactic_id="TA0001",
        technique="Exploit Public-Facing Application",
        technique_id="T1190",
        description="Unauthorized access attempt to restricted web resource.",
        severity="medium",
        recommended_action="Review WAF rules, audit access controls, patch vulnerable endpoints.",
    ),
    "path_traversal": MitreMapping(
        tactic="Discovery",
        tactic_id="TA0007",
        technique="File and Directory Discovery",
        technique_id="T1083",
        description="Path traversal attack detected attempting to access sensitive files.",
        severity="high",
        recommended_action="Patch application, sanitize inputs, review file access controls.",
    ),
    "sqli": MitreMapping(
        tactic="Collection",
        tactic_id="TA0009",
        technique="Data from Local System",
        technique_id="T1005",
        description="SQL injection attempt detected in HTTP request.",
        severity="critical",
        recommended_action="Enable WAF SQLi rules, use parameterized queries, audit DB access.",
    ),
    "web_shell": MitreMapping(
        tactic="Persistence",
        tactic_id="TA0003",
        technique="Server Software Component: Web Shell",
        technique_id="T1505.003",
        description="Web shell execution detected via HTTP request with command parameter.",
        severity="critical",
        recommended_action="Remove web shell immediately, audit web root, forensic analysis required.",
    ),

    # Metric anomaly types
    "cpu_spike": MitreMapping(
        tactic="Impact",
        tactic_id="TA0040",
        technique="Resource Hijacking",
        technique_id="T1496",
        description="Abnormal CPU utilization spike, possible cryptomining or DoS.",
        severity="high",
        recommended_action="Identify and terminate offending process, check for malware.",
    ),
    "memory_leak": MitreMapping(
        tactic="Impact",
        tactic_id="TA0040",
        technique="Endpoint Denial of Service",
        technique_id="T1499",
        description="Abnormal memory consumption detected, possible memory leak or malware.",
        severity="medium",
        recommended_action="Restart affected service, analyze heap dumps, patch application.",
    ),
    "network_exfiltration": MitreMapping(
        tactic="Exfiltration",
        tactic_id="TA0010",
        technique="Exfiltration Over C2 Channel",
        technique_id="T1041",
        description="Abnormally high outbound network traffic detected, possible data exfiltration.",
        severity="critical",
        recommended_action="Block outbound connection, preserve evidence, initiate IR process.",
    ),
    "disk_flood": MitreMapping(
        tactic="Impact",
        tactic_id="TA0040",
        technique="Data Destruction",
        technique_id="T1485",
        description="Abnormal disk I/O detected, possible ransomware encryption or log wiping.",
        severity="critical",
        recommended_action="Isolate host, preserve disk image, check for ransomware indicators.",
    ),
    "connection_storm": MitreMapping(
        tactic="Impact",
        tactic_id="TA0040",
        technique="Network Denial of Service",
        technique_id="T1498",
        description="Abnormal number of open connections, possible DoS attack or botnet activity.",
        severity="high",
        recommended_action="Enable rate limiting, activate DDoS protection, block source ranges.",
    ),
    "process_bomb": MitreMapping(
        tactic="Impact",
        tactic_id="TA0040",
        technique="Resource Hijacking",
        technique_id="T1496",
        description="Abnormal process count detected, possible fork bomb or malware spawning.",
        severity="critical",
        recommended_action="Kill runaway processes, check for fork bombs, isolate host.",
    ),

    # Network anomaly types
    "port_scan": MitreMapping(
        tactic="Discovery",
        tactic_id="TA0007",
        technique="Network Service Discovery",
        technique_id="T1046",
        description="Port scanning activity detected from external source.",
        severity="medium",
        recommended_action="Block scanning IP, review firewall rules, monitor for follow-up activity.",
    ),
    "c2_beacon": MitreMapping(
        tactic="Command and Control",
        tactic_id="TA0011",
        technique="Application Layer Protocol",
        technique_id="T1071",
        description="Regular beaconing pattern detected to suspicious external IP, possible C2.",
        severity="critical",
        recommended_action="Block C2 IP, isolate infected host, full forensic analysis required.",
    ),
    "lateral_movement": MitreMapping(
        tactic="Lateral Movement",
        tactic_id="TA0008",
        technique="Remote Services",
        technique_id="T1021",
        description="Internal host connecting to multiple hosts on admin ports, possible lateral movement.",
        severity="critical",
        recommended_action="Isolate source host, reset credentials, audit all affected systems.",
    ),
    "data_exfiltration": MitreMapping(
        tactic="Exfiltration",
        tactic_id="TA0010",
        technique="Exfiltration Over Alternative Protocol",
        technique_id="T1048",
        description="Large volume data transfer to external IP detected.",
        severity="critical",
        recommended_action="Block transfer, preserve network logs, notify data protection officer.",
    ),
    "dns_tunneling": MitreMapping(
        tactic="Command and Control",
        tactic_id="TA0011",
        technique="Protocol Tunneling",
        technique_id="T1572",
        description="High-volume DNS queries with large payloads, possible DNS tunneling.",
        severity="high",
        recommended_action="Block suspicious DNS resolver, deploy DNS monitoring, inspect traffic.",
    ),
}

# Fallback for unknown anomaly types
UNKNOWN_MAPPING = MitreMapping(
    tactic="Unknown",
    tactic_id="N/A",
    technique="Unknown Technique",
    technique_id="N/A",
    description="Anomaly detected but no MITRE ATT&CK mapping available.",
    severity="medium",
    recommended_action="Investigate manually and correlate with other indicators.",
)


class MitreMapper:
    def __init__(self):
        self.mapping = MITRE_MAP

    def map(self, anomaly_type: str) -> MitreMapping:
        return self.mapping.get(anomaly_type, UNKNOWN_MAPPING)

    def map_to_dict(self, anomaly_type: str) -> dict:
        return asdict(self.map(anomaly_type))

    def map_dataframe(self, df: pd.DataFrame, anomaly_type_col: str = "anomaly_type") -> pd.DataFrame:
        import pandas as pd
        mapped = df[anomaly_type_col].apply(lambda x: self.map_to_dict(x) if x else asdict(UNKNOWN_MAPPING))
        mapped_df = pd.json_normalize(mapped)
        return pd.concat([df.reset_index(drop=True), mapped_df], axis=1)

    def severity_score(self, severity: str) -> int:
        return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(severity, 0)

    def get_all_tactics(self) -> list:
        return sorted(set(m.tactic for m in self.mapping.values()))

    def get_by_tactic(self, tactic: str) -> dict:
        return {k: v for k, v in self.mapping.items() if v.tactic == tactic}