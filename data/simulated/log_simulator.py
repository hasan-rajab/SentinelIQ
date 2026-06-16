"""
SentinelIQ — Log Stream Simulator
Generates realistic syslog, auth.log, and apache access log entries
with synthetic anomalies injected at configurable rates.
"""

import random
import time
import json
import datetime
from typing import Generator

# ── Normal templates ──────────────────────────────────────────────────────────
SYSLOG_NORMAL = [
    "systemd[1]: Started Session {sid} of user {user}.",
    "sshd[{pid}]: Accepted publickey for {user} from {ip} port {port} ssh2",
    "sshd[{pid}]: pam_unix(sshd:session): session opened for user {user}",
    "kernel: [UFW ALLOW] IN=eth0 OUT= SRC={ip} DST=10.0.0.1 PROTO=TCP SPT={port} DPT=443",
    "cron[{pid}]: ({user}) CMD (/usr/bin/backup.sh)",
    "systemd[1]: Reloading.",
    "ntpd[{pid}]: synchronized to {ip}, stratum 2",
    "sudo: {user} : TTY=pts/0 ; PWD=/home/{user} ; USER=root ; COMMAND=/bin/systemctl status nginx",
]

AUTH_NORMAL = [
    "sshd[{pid}]: Accepted password for {user} from {ip} port {port} ssh2",
    "sshd[{pid}]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost={ip} user={user}",
    "sudo: {user} : TTY=pts/1 ; PWD=/var/log ; USER=root ; COMMAND=/bin/cat syslog",
    "useradd[{pid}]: new user: name={user}, UID=1001, GID=1001",
    "passwd[{pid}]: pam_unix(passwd:chauthtok): password changed for {user}",
]

APACHE_NORMAL = [
    '{ip} - {user} [{timestamp}] "GET /index.html HTTP/1.1" 200 1024',
    '{ip} - - [{timestamp}] "GET /api/v1/status HTTP/1.1" 200 512',
    '{ip} - {user} [{timestamp}] "POST /api/v1/login HTTP/1.1" 200 256',
    '{ip} - - [{timestamp}] "GET /static/main.css HTTP/1.1" 304 0',
    '{ip} - - [{timestamp}] "GET /favicon.ico HTTP/1.1" 404 0',
]

# ── Anomaly templates ─────────────────────────────────────────────────────────
SYSLOG_ANOMALY = [
    "sshd[{pid}]: Failed password for root from {ip} port {port} ssh2",
    "sshd[{pid}]: Failed password for root from {ip} port {port} ssh2",
    "sshd[{pid}]: Failed password for root from {ip} port {port} ssh2",
    "sshd[{pid}]: Failed password for root from {ip} port {port} ssh2",
    "sshd[{pid}]: Failed password for root from {ip} port {port} ssh2",  # brute force burst
    "kernel: [UFW BLOCK] IN=eth0 OUT= SRC={ip} DST=10.0.0.1 PROTO=TCP SPT={port} DPT=22",
    "sudo: {user} : command not allowed ; TTY=pts/0 ; PWD=/ ; USER=root ; COMMAND=/bin/bash",
    "sshd[{pid}]: Received disconnect from {ip} port {port}: 11: Bye Bye [preauth]",
    "kernel: possible SYN flooding on port 80. Sending cookies.",
]

AUTH_ANOMALY = [
    "sshd[{pid}]: Invalid user admin from {ip} port {port}",
    "sshd[{pid}]: Invalid user oracle from {ip} port {port}",
    "sshd[{pid}]: Invalid user testuser from {ip} port {port}",
    "sshd[{pid}]: authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost={ip}",
    "sudo: pam_unix(sudo:auth): authentication failure; logname={user} uid=1000 euid=0 tty=/dev/pts/0",
    "su[{pid}]: FAILED SU (to root) {user} on pts/0",
]

APACHE_ANOMALY = [
    '{ip} - - [{timestamp}] "GET /admin/config.php HTTP/1.1" 403 512',
    '{ip} - - [{timestamp}] "GET /../../../etc/passwd HTTP/1.1" 400 0',
    '{ip} - - [{timestamp}] "GET /wp-admin/install.php HTTP/1.1" 404 0',
    '{ip} - - [{timestamp}] "POST /api/v1/exec HTTP/1.1" 500 0',
    '{ip} - - [{timestamp}] "GET /shell.php?cmd=whoami HTTP/1.1" 200 32',
    '{ip} - - [{timestamp}] "UNION SELECT * FROM users-- HTTP/1.1" 400 0',
]

# ── Helpers ───────────────────────────────────────────────────────────────────
NORMAL_IPS = [f"192.168.1.{i}" for i in range(10, 50)]
ANOMALY_IPS = [f"45.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}" for _ in range(20)]
USERS = ["alice", "bob", "carol", "dave", "svc_backup", "svc_monitor"]


def _fill(template: str, anomaly: bool = False) -> str:
    now = datetime.datetime.utcnow()
    return template.format(
        sid=random.randint(100, 999),
        pid=random.randint(1000, 9999),
        user=random.choice(USERS),
        ip=random.choice(ANOMALY_IPS if anomaly else NORMAL_IPS),
        port=random.randint(1024, 65535),
        timestamp=now.strftime("%d/%b/%Y:%H:%M:%S +0000"),
    )


def _record(log_type: str, message: str, anomaly: bool, anomaly_type: str = None) -> dict:
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "log_type": log_type,
        "message": message,
        "is_anomaly": anomaly,
        "anomaly_type": anomaly_type,
        "source": f"host-{random.randint(1, 5)}.sentineliq.internal",
    }


# ── Generators ────────────────────────────────────────────────────────────────
def generate_syslog(anomaly_rate: float = 0.05) -> dict:
    is_anomaly = random.random() < anomaly_rate
    if is_anomaly:
        template = random.choice(SYSLOG_ANOMALY)
        msg = _fill(template, anomaly=True)
        return _record("syslog", msg, True, "brute_force" if "Failed" in msg else "privilege_escalation")
    return _record("syslog", _fill(random.choice(SYSLOG_NORMAL)), False)


def generate_auth(anomaly_rate: float = 0.05) -> dict:
    is_anomaly = random.random() < anomaly_rate
    if is_anomaly:
        msg = _fill(random.choice(AUTH_ANOMALY), anomaly=True)
        return _record("auth", msg, True, "invalid_user" if "Invalid" in msg else "auth_failure")
    return _record("auth", _fill(random.choice(AUTH_NORMAL)), False)


def generate_apache(anomaly_rate: float = 0.05) -> dict:
    is_anomaly = random.random() < anomaly_rate
    if is_anomaly:
        msg = _fill(random.choice(APACHE_ANOMALY), anomaly=True)
        atype = "path_traversal" if "passwd" in msg else "sqli" if "UNION" in msg else "web_shell" if "cmd=" in msg else "unauthorized_access"
        return _record("apache", msg, True, atype)
    return _record("apache", _fill(random.choice(APACHE_NORMAL)), False)


def stream_logs(
    rate_per_second: float = 5.0,
    anomaly_rate: float = 0.05,
    duration_seconds: int = None,
    output_file: str = None,
) -> Generator[dict, None, None]:
    """
    Yields log records at the given rate.
    If output_file is set, also writes JSONL to disk.
    If duration_seconds is None, runs indefinitely.
    """
    generators = [generate_syslog, generate_auth, generate_apache]
    interval = 1.0 / rate_per_second
    count = 0
    start = time.time()

    f = open(output_file, "w") if output_file else None

    try:
        while True:
            record = random.choice(generators)(anomaly_rate=anomaly_rate)
            if f:
                f.write(json.dumps(record) + "\n")
                f.flush()
            yield record
            count += 1
            if duration_seconds and (time.time() - start) >= duration_seconds:
                break
            time.sleep(interval)
    finally:
        if f:
            f.close()
        print(f"[log_simulator] Generated {count} records.")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SentinelIQ Log Simulator")
    parser.add_argument("--rate", type=float, default=5.0, help="Records per second")
    parser.add_argument("--anomaly-rate", type=float, default=0.08, help="Fraction of records that are anomalies")
    parser.add_argument("--duration", type=int, default=30, help="Seconds to run (0 = infinite)")
    parser.add_argument("--output", type=str, default="data/simulated/logs.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    print(f"[SentinelIQ] Streaming logs → {args.output}")
    print(f"  rate={args.rate}/s  anomaly_rate={args.anomaly_rate}  duration={args.duration}s\n")

    for record in stream_logs(
        rate_per_second=args.rate,
        anomaly_rate=args.anomaly_rate,
        duration_seconds=args.duration if args.duration > 0 else None,
        output_file=args.output,
    ):
        label = "🔴 ANOMALY" if record["is_anomaly"] else "🟢 normal "
        print(f"[{label}] [{record['log_type']:6}] {record['message'][:80]}")