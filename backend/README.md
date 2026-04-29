## Backend (FastAPI)

This backend loads the saved DoshaNet model from the repo root and exposes API endpoints consumed by the React frontend.

### Endpoints

- `GET /api/artifacts`
  - Returns `model_card` + URLs for saved plots (confusion matrix + explanation images).
- `GET /api/schema`
  - Returns the feature schema for the symptom input form.
- `POST /api/predict`
  - Body: `{ "features": { "<FeatureName>": "<selected option>" } }`
  - Response includes predicted dosha, confidence, probabilities, and remedy guidance.

### Run

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Notes

- `torch-geometric` install can require extra wheels depending on your OS/CUDA setup.
- The backend serves repo-root assets (images/json) under `/assets/...`.

