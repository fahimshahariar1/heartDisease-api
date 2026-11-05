# Heart Disease Prediction API

A FastAPI service that predicts the presence of heart disease from 13 clinical features, packaged with Docker and deployed on Render.

**Live API:** `https://<your-service>.onrender.com` · **Swagger UI:** `https://<your-service>.onrender.com/docs`

> Educational project. Not a medical tool.

## Project structure

```
heart-disease-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app: /health, /info, /predict
│   └── schemas.py         # Pydantic request/response models
├── model/
│   ├── heart_model.joblib # trained model (created by train.py)
│   └── metadata.json      # feature order, metrics, versions
├── data/                  # put heart.csv here (not committed)
├── train.py               # trains and saves the model
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Model

- **Dataset:** [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), 13 features, binary `target` (1 = disease present, 0 = absent)
- **Preprocessing:** duplicate rows removed before splitting, so the same patient can't appear in both train and test sets
- **Classifier:** Random Forest (200 trees, max depth 6), 80/20 stratified split
- **Test metrics:** see `GET /info` (written to `model/metadata.json` at training time)

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check and whether the model is loaded |
| GET | `/info` | Model type, feature list, test metrics, scikit-learn version |
| POST | `/predict` | Returns `heart_disease` (true/false) and its probability |
| GET | `/docs` | Interactive Swagger UI |

**Example request** (`POST /predict`):

```json
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
  "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
```

**Example response:**

```json
{ "heart_disease": false, "probability": 0.2115 }
```

Invalid input (missing fields, wrong types, out-of-range values) is rejected by Pydantic with a `422` error before it reaches the model.

## Run locally

### 1. Train the model

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py --data data/heart.csv
```

### 2. Run with Docker Compose

```bash
docker compose build      # or: docker-compose build
docker compose up         # or: docker-compose up
```

Open http://localhost:8000/docs.

### 3. Test the endpoints

**curl (macOS/Linux/Git Bash):**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

**PowerShell:**

```powershell
$body = @{ age=52; sex=1; cp=0; trestbps=125; chol=212; fbs=0; restecg=1; thalach=168; exang=0; oldpeak=1.0; slope=2; ca=2; thal=3 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $body
```

## Deploy on Render

1. Push this repository to GitHub, including `model/heart_model.joblib` and `model/metadata.json`.
2. On Render: **New → Web Service** → connect the GitHub repo.
3. **Language:** Docker. **Branch:** `main`. **Root directory:** leave blank (the Dockerfile is at the repo root).
4. **Instance type:** Free. Under *Advanced*, set **Health Check Path** to `/health`.
5. Create the service and wait for the build and deploy to finish.
6. Test the live URL with the curl/PowerShell commands above, replacing `http://localhost:8000` with the Render URL.

Render free instances sleep after inactivity, so the first request after a pause can take up to about a minute.

## Design decisions

- **Pinned ML versions:** `scikit-learn`, `pandas`, `numpy` and `joblib` are pinned exactly. A model saved with one scikit-learn version may fail or misbehave when loaded with another, so training and the container must match.
- **Model loaded once at startup** (FastAPI `lifespan`), not on every request.
- **Feature order from `metadata.json`:** the API builds its input in exactly the column order used for training.
- **`PORT` environment variable:** Render assigns the port at runtime; the container uses `${PORT:-8000}`, so it works both on Render and locally.
- **`--host 0.0.0.0`:** makes uvicorn reachable from outside the container.
- **Dependency layer cached:** `requirements.txt` is copied and installed before the app code, so code changes don't reinstall packages.
- **Non-root container user** for basic security.

## Tech stack

Python 3.12 · FastAPI · Pydantic · scikit-learn · Docker · Docker Compose · Render
