## Frontend (React)

This folder contains a Vite + React (JavaScript) app for the DoshaNet demo.

### Setup

1. Install dependencies:
   - `npm install`
2. (Optional) Set the backend URL:
   - copy `.env.example` to `.env`
3. Run:
   - `npm run dev`

### Expected backend

The frontend expects a backend at `VITE_API_BASE_URL` (default: `http://localhost:8000`) that exposes:

- `GET /api/artifacts`
- `GET /api/schema`
- `POST /api/predict`

