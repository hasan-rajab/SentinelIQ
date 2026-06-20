# SentinelIQ Startup Guide

## 1. Backend (local development)

1. Activate the Python virtual environment:

```bash
cd /Users/hassanali/Desktop/SentinelIQ
source .venv/bin/activate
```

2. Install dependencies if not already installed:

```bash
pip install -r backend/requirements.txt
```

3. Start the FastAPI backend:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

4. Verify the backend is running:

```bash
curl -fs http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","models_loaded":{...},"version":"1.0.0"}
```

---

## 2. Frontend (local development)

1. Open a new terminal and go to the frontend folder:

```bash
cd /Users/hassanali/Desktop/SentinelIQ/frontend
```

2. Install Node dependencies if not already installed:

```bash
npm install
```

3. Start the Next.js development server:

```bash
npm run dev -- --hostname 0.0.0.0 --port 3000
```

4. Open the dashboard in your browser:

```text
http://localhost:3000
```

---

## 3. Optional Docker Compose startup

If you want to run the full stack with Kafka and the app containers, use:

```bash
cd /Users/hassanali/Desktop/SentinelIQ
cp .env.example .env
docker-compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- Kafka: `http://localhost:9092`

---

## 4. Notes

- The backend expects trained model files in `ml/saved_models/`.
- If the model files are missing or incompatible, the backend may start but return incomplete model loading state.
- For quick local development, start backend and frontend separately rather than using Docker if Dockerfiles are not found.
