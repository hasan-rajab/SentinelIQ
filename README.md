# SentinelIQ

**AI-native multimodal anomaly intelligence for IT Ops & Cybersecurity**

SentinelIQ ingests logs, system metrics, and network flows in real time, fuses independent ML models into a single anomaly score, explains every alert with SHAP feature attribution and MITRE ATT&CK mapping, and trains collaboratively across multiple nodes without ever sharing raw data.

---

## Why This Exists

Most anomaly detection tools are black boxes—they flag something and leave you guessing why. Most are also single-modality, missing the bigger picture when an attack spans logs, metrics, and network traffic at once.

SentinelIQ was built to solve both problems, with privacy-preserving federated training as a third differentiator for organizations that cannot centralize sensitive data.

---

## Architecture

```text
┌─────────────┐     ┌──────────────────────────┐     ┌─────────────────┐
│  Simulated  │────▶│       ML Layer           │────▶│  Fusion + SHAP  │
│  Data       │     │  XGBoost + AE (network) │     │  + MITRE ATT&CK │
│  Streams    │     │  AE only   (metrics)    │     │  Explainability │
│             │     │  BERT      (logs)       │     │                 │
└─────────────┘     └──────────────────────────┘     └────────┬────────┘
                                                               │
      ┌────────────────────────────────────────────────────────┘
      ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  FastAPI    │────▶│  WebSocket   │────▶│  Next.js         │
│  Backend    │     │  Live Stream │     │  SOC Dashboard   │
└─────────────┘     └──────────────┘     └──────────────────┘

      Federated Layer (Flower): 3 nodes train collaboratively,
      only model weights are exchanged — never raw data.
```

---

## Tech Stack

| Layer              | Technology                                                                       |
| ------------------ | -------------------------------------------------------------------------------- |
| Data Simulation    | Python (custom log, metric, and network generators)                              |
| ML Models          | XGBoost, PyTorch Autoencoder, Hugging Face Transformers (BERT), Isolation Forest |
| Explainability     | SHAP, custom MITRE ATT&CK mapper                                                 |
| Federated Learning | Flower (flwr)                                                                    |
| Backend            | FastAPI, WebSockets                                                              |
| Frontend           | Next.js 14, TypeScript, Tailwind CSS                                             |
| Training           | Kaggle (free T4 GPU tier)                                                        |
| Orchestration      | Docker Compose, Kafka                                                            |

---

## Project Structure

```text
sentineliq/
├── data/
│   └── simulated/           # Synthetic log, metric, and network generators
├── ml/
│   ├── models/              # XGBoost, Autoencoder, BERT, Isolation Forest
│   ├── training/            # CLI training and calibration scripts
│   ├── fusion/              # Ensemble scoring logic
│   ├── explainability/      # SHAP + MITRE ATT&CK mapping
│   ├── features/            # Network feature engineering
│   └── saved_models/        # Trained weights (excluded from git)
├── federated/               # Flower server, client, simulations
├── backend/                 # FastAPI application
├── frontend/                # Next.js SOC dashboard
├── notebooks/               # Kaggle notebooks (01–07)
├── configs/                 # YAML configurations
└── docker-compose.yml
```

---

# Quick Start

## 1. Train the Models (Kaggle)

Open notebooks **01–07** in order using Kaggle's free GPU environment.

Each notebook:

* Clones the repository
* Generates fresh simulated data
* Trains the relevant model
* Saves model artifacts

See `notebooks/README.md` for detailed Kaggle instructions.

---

## 2. Download Trained Weights

Large model files (particularly BERT) exceed GitHub's file size limits and are not included in the repository.

Place downloaded models into:

```text
ml/saved_models/
├── xgboost_network_*
├── autoencoder_metrics_*
├── autoencoder_network_*
├── isolation_forest_metrics_*
├── isolation_forest_network_*
├── bert_log/
├── bert_log_meta.json
└── ensemble_config.json
```

---

## 3. Run with Docker

```bash
cp .env.example .env
docker-compose up --build
```

### Services

| Service            | URL                        |
| ------------------ | -------------------------- |
| Backend API        | http://localhost:8000      |
| Swagger Docs       | http://localhost:8000/docs |
| Frontend Dashboard | http://localhost:3000      |
| Kafka              | localhost:9092             |

---

## 4. Run Without Docker

### Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# Features

### Multimodal Detection

Analyze multiple telemetry sources simultaneously:

* Logs → BERT
* Metrics → Autoencoder
* Network flows → XGBoost + Autoencoder Ensemble

### Supervised + Unsupervised Fusion

* XGBoost identifies known attack patterns using labeled data.
* Autoencoders provide protection against previously unseen attack behavior.

### Explainable by Design

Every alert includes:

* SHAP feature attribution
* MITRE ATT&CK tactic mapping
* MITRE ATT&CK technique mapping

### Federated Learning

Train across multiple environments without centralizing sensitive data.

* Flower
* FedAvg aggregation
* Model-weight exchange only

### Live SOC Dashboard

Real-time visualization including:

* Live WebSocket feed
* Anomaly score waveform
* Alert triage panel
* Federated node topology

### Zero-Cost Development Stack

* Kaggle GPU training
* Self-hosted Kafka
* Open-source tooling

---

# Model Performance

Evaluated on fresh synthetic data not seen during training.

Dataset sizes:

* Network records: 1,142
* Metric records: 605
* Log records: 585

| Modality | Model                        | Recall  | Precision | F1 Score |
| -------- | ---------------------------- | ------- | --------- | -------- |
| Metrics  | Autoencoder                  | 100.00% | 97.62%    | 0.988    |
| Network  | XGBoost                      | 85.85%  | 100.00%   | 0.924    |
| Network  | XGBoost + Autoencoder Fusion | 99.06%  | 100.00%   | 0.995    |
| Logs     | BERT                         | 100.00% | 100.00%   | 1.000    |

---

## Known Weak Spots

### Lateral Movement

Most difficult attack category due to similarity with legitimate administrative traffic.

### Memory Leak

Gradual memory growth can resemble legitimate high-load workloads, resulting in approximately one missed detection per ~600 records.

---

## Important Note

Performance metrics are based entirely on synthetic simulator-generated data with relatively clean class separation.

Real-world performance will be lower, especially for:

* Low-and-slow attacks
* Traffic throttling techniques
* Adversaries intentionally mimicking normal behavior

These results should be treated as **upper-bound research benchmarks**, not production guarantees.

---

# Detection Pipeline

```text
Network Flow
    │
    ▼
Feature Engineering (15 Features)
    │
    ▼
XGBoost Score
    │
    ├───────────────╮
    ▼               │
Weighted Fusion     │
(0.7 / 0.3)         │
    │               │
    ▼               │
Threshold 0.484 ◀───╯
    │
    ▼
Alert

Autoencoder Reconstruction Error
    │
    └─────────────────────────────▶ Fusion


Metric Record
    │
    ▼
Autoencoder
    │
    ▼
Threshold 0.743
    │
    ▼
Alert


Log Record
    │
    ▼
BERT Classifier
    │
    ▼
Threshold 0.500
    │
    ▼
Alert
```

---

# Future Roadmap

* Real-world benchmark datasets (CICIDS2017, UNSW-NB15)
* Online learning and drift adaptation
* Kubernetes deployment
* Multi-tenant SOC dashboard
* Graph-based attack correlation
* LLM-assisted incident summaries
* Streaming Kafka ingestion pipeline

---

# License

MIT License

Feel free to use, modify, and distribute this project under the terms of the MIT License.
