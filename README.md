# SentinelIQ

**AI-native multimodal anomaly intelligence for IT Ops & Cybersecurity**

SentinelIQ ingests logs, system metrics, and network flows in real time, fuses three independent ML models into a single anomaly score, explains every alert with SHAP feature attribution and MITRE ATT&CK mapping, and trains collaboratively across multiple nodes without ever sharing raw data.

---

## Why this exists

Most anomaly detection tools are black boxes — they flag something and leave you guessing why. Most are also single-modality, missing the bigger picture when an attack spans logs, metrics, and network traffic at once. SentinelIQ was built to solve both problems, with privacy-preserving federated training as a third differentiator for organizations that can't centralize sensitive data.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Simulated  │────▶│   ML Layer   │────▶│  Fusion + SHAP   │
│  Data       │     │  IF / AE /   │     │  + MITRE ATT&CK  │
│  Streams    │     │  BERT        │     │  Explainability  │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                     │
      ┌──────────────────────────────────────────────┘
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  FastAPI    │────▶│  WebSocket   │────▶│  Next.js         │
│  Backend    │     │  Live Stream │     │  SOC Dashboard   │
└─────────────┘     └──────────────┘     └──────────────────┘

      Federated Layer (Flower): 3 nodes train collaboratively,
      only model weights are exchanged — never raw data.
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data simulation | Python (custom log/metric/network generators) |
| ML models | scikit-learn (Isolation Forest), PyTorch (Autoencoder), HuggingFace Transformers (fine-tuned BERT) |
| Explainability | SHAP, custom MITRE ATT&CK mapper |
| Federated learning | Flower (flwr) |
| Backend | FastAPI, WebSockets |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Training | Kaggle (free T4 GPU tier) |
| Orchestration | Docker Compose, Kafka |

---

## Project Structure

```
sentineliq/
├── data/simulated/        Synthetic log/metric/network generators
├── ml/
│   ├── models/             Isolation Forest, Autoencoder, BERT classes
│   ├── training/            CLI training scripts
│   ├── fusion/                Ensemble scoring
│   ├── explainability/    SHAP + MITRE ATT&CK mapping
│   └── saved_models/      Trained weights (large files excluded from git)
├── federated/              Flower server, client, simulation runner
├── backend/                 FastAPI app, routes, services, schemas
├── frontend/                 Next.js SOC dashboard
├── notebooks/               Kaggle training notebooks (01–07)
├── configs/                  YAML configs for models + federated setup
└── docker-compose.yml
```

---

## Quickstart

### 1. Train the models (Kaggle)
Open `notebooks/01` through `notebooks/07` in order on Kaggle (free GPU tier). Each notebook clones this repo, generates fresh simulated data, trains, and saves model weights. See [`notebooks/README.md`](notebooks/) for Kaggle-specific setup.

### 2. Download trained weights
Models larger than GitHub's 100MB limit (BERT weights) are excluded from the repo. Download them from your Kaggle output and place them in `ml/saved_models/`:

```
ml/saved_models/
├── isolation_forest_metrics_*
├── isolation_forest_network_*
├── autoencoder_metrics_*
├── bert_log/                  ← download separately, not in git
├── bert_log_meta.json
└── ensemble_config.json
```

### 3. Run locally with Docker
```bash
cp .env.example .env
docker-compose up --build
```

- Backend: `http://localhost:8000` (docs at `/docs`)
- Frontend: `http://localhost:3000`
- Kafka: `localhost:9092`

### 4. Run without Docker (development)
```bash
# Backend
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

---

## Features

- **Multimodal detection** — logs (BERT), metrics (Isolation Forest + Autoencoder), network flows (Isolation Forest)
- **Explainable by design** — every alert ships with SHAP feature attribution and a MITRE ATT&CK tactic/technique mapping
- **Federated learning** — train across multiple data-sensitive environments without centralizing raw data (Flower + FedAvg)
- **Live SOC dashboard** — real-time WebSocket feed, anomaly score waveform, alert triage, federated node topology
- **Zero-cost stack** — Kaggle GPU training, self-hosted Kafka, free-tier everything

---

## Model Performance

| Model | ROC-AUC | F1 |
|---|---|---|
| Isolation Forest (metrics) | ~0.85 | — |
| Isolation Forest (network) | ~0.85 | — |
| Autoencoder (metrics) | ~0.85 | — |
| BERT (logs) | 1.00 | 1.00 |
| Ensemble (fused, tuned threshold) | 0.85 | 0.33 |

*Fusion threshold tuned for high recall (0.81) given security use case — false positives are cheaper than missed attacks.*

---

## License

MIT
