"""
SentinelIQ — Network Flow Simulator
Generates realistic NetFlow-style records with synthetic
anomalies (port scans, C2 beaconing, lateral movement, exfiltration).
"""

import random
import time
import json
import datetime
from typing import Generator

# ── Known internal infrastructure ────────────────────────────────────────────
INTERNAL_HOSTS = [f"10.0.0.{i}" for i in range(1, 20)]
INTERNAL_SUBNETS = ["10.0.0.0/24", "10.0.1.0/24", "172.16.0.0/24"]
EXTERNAL_LEGIT = [
    "8.8.8.8", "1.1.1.1", "142.250.80.46",   # Google DNS, Cloudflare, Google
    "151.101.1.140", "13.32.99.87",            # Fastly CDN, AWS CloudFront
    "185.199.108.153",                          # GitHub Pages
]
EXTERNAL_MALICIOUS = [
    f"45.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
    for _ in range(30)
]

COMMON_PORTS = [80, 443, 53, 22, 25, 587, 8080, 3306, 5432, 6379]
PROTOCOLS = ["TCP", "UDP", "ICMP"]

SERVICES = {
    80: "HTTP", 443: "HTTPS", 53: "DNS", 22: "SSH",
    25: "SMTP", 587: "SUBMISSION", 8080: "HTTP-ALT",
    3306: "MYSQL", 5432: "POSTGRES", 6379: "REDIS",
}


def _flow(src_ip, dst_ip, src_port, dst_port, proto, bytes_out, bytes_in,
          packets, duration_ms, is_anomaly, anomaly_type=None) -> dict:
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto,
        "bytes_out": bytes_out,
        "bytes_in": bytes_in,
        "packets": packets,
        "duration_ms": duration_ms,
        "service": SERVICES.get(dst_port, "UNKNOWN"),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
    }


# ── Normal flow generators ────────────────────────────────────────────────────
def normal_web_traffic() -> dict:
    src = random.choice(INTERNAL_HOSTS)
    dst = random.choice(EXTERNAL_LEGIT)
    port = random.choice([80, 443])
    return _flow(
        src, dst, random.randint(49152, 65535), port, "TCP",
        random.randint(500, 5000), random.randint(2000, 50000),
        random.randint(5, 50), random.randint(100, 2000),
        False
    )


def normal_dns() -> dict:
    src = random.choice(INTERNAL_HOSTS)
    return _flow(
        src, "8.8.8.8", random.randint(49152, 65535), 53, "UDP",
        random.randint(50, 200), random.randint(50, 500),
        random.randint(1, 4), random.randint(5, 100),
        False
    )


def normal_internal() -> dict:
    src, dst = random.sample(INTERNAL_HOSTS, 2)
    port = random.choice(COMMON_PORTS)
    return _flow(
        src, dst, random.randint(49152, 65535), port, "TCP",
        random.randint(100, 10000), random.randint(100, 10000),
        random.randint(2, 30), random.randint(10, 500),
        False
    )


# ── Anomaly flow generators ───────────────────────────────────────────────────
def port_scan() -> dict:
    """Single source scanning many ports on one target."""
    src = random.choice(EXTERNAL_MALICIOUS)
    dst = random.choice(INTERNAL_HOSTS)
    return _flow(
        src, dst, random.randint(49152, 65535),
        random.randint(1, 1024), "TCP",
        60, 0, 1, random.randint(1, 10),
        True, "port_scan"
    )


def c2_beacon() -> dict:
    """Regular low-volume outbound to suspicious external IP."""
    src = random.choice(INTERNAL_HOSTS)
    dst = random.choice(EXTERNAL_MALICIOUS)
    return _flow(
        src, dst, random.randint(49152, 65535),
        random.choice([4444, 8888, 1337, 31337, 443]), "TCP",
        random.randint(200, 800), random.randint(100, 400),
        random.randint(3, 10), random.randint(250, 260),   # suspiciously regular timing
        True, "c2_beacon"
    )


def lateral_movement() -> dict:
    """Internal host connecting to many other internal hosts on admin ports."""
    src = random.choice(INTERNAL_HOSTS)
    dst = random.choice([h for h in INTERNAL_HOSTS if h != src])
    return _flow(
        src, dst, random.randint(49152, 65535),
        random.choice([22, 445, 3389, 5985]), "TCP",
        random.randint(1000, 5000), random.randint(500, 2000),
        random.randint(10, 50), random.randint(50, 300),
        True, "lateral_movement"
    )


def data_exfiltration() -> dict:
    """Massive outbound transfer to external IP."""
    src = random.choice(INTERNAL_HOSTS)
    dst = random.choice(EXTERNAL_MALICIOUS)
    return _flow(
        src, dst, random.randint(49152, 65535),
        random.choice([443, 80, 21, 22]), "TCP",
        random.randint(50_000_000, 500_000_000),  # 50MB–500MB
        random.randint(1000, 5000),
        random.randint(50000, 500000),
        random.randint(30000, 300000),
        True, "data_exfiltration"
    )


def dns_tunneling() -> dict:
    """High-volume DNS queries with large payloads — typical DNS tunnel."""
    src = random.choice(INTERNAL_HOSTS)
    return _flow(
        src, random.choice(EXTERNAL_MALICIOUS),
        random.randint(49152, 65535), 53, "UDP",
        random.randint(5000, 20000), random.randint(3000, 10000),
        random.randint(200, 1000), random.randint(5000, 30000),
        True, "dns_tunneling"
    )


ANOMALY_GENERATORS = [port_scan, c2_beacon, lateral_movement, data_exfiltration, dns_tunneling]
NORMAL_GENERATORS = [normal_web_traffic, normal_dns, normal_internal,
                     normal_web_traffic, normal_web_traffic]   # web traffic is most common


def stream_network(
    rate_per_second: float = 10.0,
    anomaly_rate: float = 0.05,
    duration_seconds: int = None,
    output_file: str = None,
) -> Generator[dict, None, None]:
    interval = 1.0 / rate_per_second
    count = 0
    start = time.time()
    f = open(output_file, "w") if output_file else None

    try:
        while True:
            if random.random() < anomaly_rate:
                record = random.choice(ANOMALY_GENERATORS)()
            else:
                record = random.choice(NORMAL_GENERATORS)()

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
        print(f"[network_simulator] Generated {count} records.")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SentinelIQ Network Flow Simulator")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--anomaly-rate", type=float, default=0.08)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output", type=str, default="data/simulated/network.jsonl")
    args = parser.parse_args()

    print(f"[SentinelIQ] Streaming network flows → {args.output}\n")

    for record in stream_network(
        rate_per_second=args.rate,
        anomaly_rate=args.anomaly_rate,
        duration_seconds=args.duration if args.duration > 0 else None,
        output_file=args.output,
    ):
        label = "🔴 ANOMALY" if record["is_anomaly"] else "🟢 normal "
        print(
            f"[{label}] {record['src_ip']:15} → {record['dst_ip']:15} "
            f":{record['dst_port']:5} {record['protocol']:4} "
            f"{record['bytes_out']:>12,}B out"
            + (f"  [{record['anomaly_type']}]" if record["is_anomaly"] else "")
        )