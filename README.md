SentinelIQ
AI-native multimodal anomaly intelligence for IT Ops & Cybersecurity
SentinelIQ ingests logs, system metrics, and network flows in real time, fuses independent ML models into a single anomaly score, explains every alert with SHAP feature attribution and MITRE ATT&CK mapping, and trains collaboratively across multiple nodes without ever sharing raw data.

Why this exists
Most anomaly detection tools are black boxes — they flag something and leave you guessing why. Most are also single-modality, missing the bigger picture when an attack spans logs, metrics, and network traffic at once. SentinelIQ was built to solve both problems, with privacy-preserving federated training as a third differentiator for organizations that can't centralize sensitive data.

Architecture
┌─────────────┐     ┌──────────────────────────┐     ┌─────────────────┐
│  Simulated  │────▶│       ML Layer            │────▶│  Fusion + SHAP  │
│  Data       │     │  XGBoost + AE (network)  │     │  + MITRE ATT&CK │
│  Streams    │     │  AE only   (metrics)     │     │  Explainability │
│             │     │  BERT      (logs)        │     │                 │
└─────────────┘     └──────────────────────────┘     └────────┬────────┘
                                                               │
      ┌────────────────────────────────────────────────────────┘
      ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  FastAPI    │────▶│  WebSocket   │────▶│  Next.js          │
│  Backend    │     │  Live Stream │     │  SOC Dashboard   │
└─────────────┘     └──────────────┘     └──────────────────┘

      Federated Layer (Flower): 3 nodes train collaboratively,
      only model weights are exchanged — never raw data.

Tech Stack
LayerTechnologyData simulationPython (custom log/metric/network generators)ML modelsXGBoost (network classifier), PyTorch (Autoencoder), HuggingFace Transformers (fine-tuned BERT), scikit-learn (Isolation Forest)ExplainabilitySHAP, custom MITRE ATT&CK mapperFederated learningFlower (flwr)BackendFastAPI, WebSocketsFrontendNext.js 14, TypeScript, Tailwind CSSTrainingKaggle (free T4 GPU tier)OrchestrationDocker Compose, Kafka

Project Structure
sentineliq/
├── data/simulated/         Synthetic log/metric/network generators
├── ml/
│   ├── models/             XGBoost, Autoencoder, BERT, Isolation Forest
│   ├── training/           CLI training + calibration scripts
│   ├── fusion/             Ensemble scoring (XGBoost+AE, AE-only paths)
│   ├── explainability/     SHAP + MITRE ATT&CK mapping
│   ├── features/           Network feature engineering (15 features)
│   └── saved_models/       Trained weights (large files excluded from git)
├── federated/              Flower server, client, simulation runner
├── backend/                FastAPI app, routes, services, schemas
├── frontend/               Next.js SOC dashboard
├── notebooks/              Kaggle training notebooks (01–07)
├── configs/                YAML configs for models + federated setup
└── docker-compose.yml

Quickstart
1. Train the models (Kaggle)
Open notebooks/01 through notebooks/07 in order on Kaggle (free GPU tier). Each notebook clones this repo, generates fresh simulated data, trains, and saves model weights. See notebooks/README.md for Kaggle-specific setup.
2. Download trained weights
Models larger than GitHub's 100MB limit (BERT weights) are excluded from the repo. Download them from your Kaggle output and place them in ml/saved_models/:
ml/saved_models/
├── xgboost_network_*
├── autoencoder_metrics_*
├── autoencoder_network_*
├── isolation_forest_metrics_*
├── isolation_forest_network_*
├── bert_log/                  ← download separately, not in git
├── bert_log_meta.json
└── ensemble_config.json
3. Run locally with Docker
bashcp .env.example .env
docker-compose up --build

Backend: http://localhost:8000 (docs at /docs)
Frontend: http://localhost:3000
Kafka: localhost:9092

4. Run without Docker (development)
bash# Backend
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev

Features

Multimodal detection — logs (BERT), metrics (Autoencoder), network flows (supervised XGBoost + Autoencoder ensemble)
Supervised + unsupervised fusion — XGBoost classifies known attack patterns using labeled data; Autoencoder acts as a safety net for novel/unknown attack shapes via reconstruction error
Explainable by design — every alert ships with SHAP feature attribution and a MITRE ATT&CK tactic/technique mapping
Federated learning — train across multiple data-sensitive environments without centralizing raw data (Flower + FedAvg)
Live SOC dashboard — real-time WebSocket feed, anomaly score waveform, alert triage, federated node topology
Zero-cost stack — Kaggle GPU training, self-hosted Kafka, free-tier everything


Model Performance
Evaluated on fresh synthetic data not seen during training (1,142 network records, 605 metric records, 585 log records).
ModalityModelRecallPrecisionF1MetricsAutoencoder100%97.62%0.988NetworkXGBoost alone85.85%100%0.924NetworkXGBoost + AE fused99.06%100%0.995LogsBERT100%100%1.000
Known weak spots:

lateral_movement — hardest attack type; low feature separation from normal admin traffic
memory_leak — gradual memory growth overlaps with legitimate high-load behavior; 1 irreducible miss per ~600 records

Synthetic data caveat: results are on simulator-generated data with clean class separation. Real-world performance will be lower, particularly for attack types that throttle or mimic normal traffic. These numbers are upper bounds, not production guarantees.

Detection Pipeline
Network flow ──▶ Feature engineering (15 features) ──▶ XGBoost score ──▶ ╮
                                                                           weighted fusion (0.7/0.3) ──▶ threshold 0.484 ──▶ Alert
                                                    Autoencoder recon ──▶ ╯

Metric record ──▶ Autoencoder ──▶ threshold 0.743 ──▶ Alert

Log record ──▶ BERT classifier ──▶ threshold 0.500 ──▶ Alert

License
MIT
