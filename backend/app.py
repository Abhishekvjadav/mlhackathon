import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CARD_PATH = PROJECT_ROOT / "model_card.json"
PRAKRITI_PATH = PROJECT_ROOT / "prakriti_clean.json"
AYUR_PATH = PROJECT_ROOT / "ayurgenixai_clean.json"
ENCODERS_PATH = PROJECT_ROOT / "encoders.pkl"
GRAPH_DATA_PATH = PROJECT_ROOT / "graph_data.pt"
BEST_MODEL_PATH = PROJECT_ROOT / "best_model.pt"


app = FastAPI(title="DoshaNet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve files (images + json) from the repo root under /assets
app.mount("/assets", StaticFiles(directory=str(PROJECT_ROOT)), name="assets")


class HeteroDoshaNet(torch.nn.Module):
    """
    Must match the architecture used when producing `best_model.pt`.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_dim: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.metadata = (
            ["patient", "symptom", "dosha"],
            [
                ("patient", "has_trait", "symptom"),
                ("patient", "belongs_to", "dosha"),
                ("patient", "similar_to", "patient"),
            ],
        )
        self.han1 = HANConv(
            in_channels_dict,
            hidden_dim,
            metadata=self.metadata,
            heads=heads,
            dropout=dropout,
        )
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.han2 = HANConv(
            hidden_dim,
            num_classes,
            metadata=self.metadata,
            heads=1,
            dropout=dropout,
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.han1(x_dict, edge_index_dict)
        out = {k: F.elu(self.bn(v)) for k, v in out.items()}
        out = {k: self.drop(v) for k, v in out.items()}
        out = self.han2(out, edge_index_dict)
        return F.log_softmax(out["patient"], dim=1)


class PredictRequest(BaseModel):
    features: Dict[str, str]

class ProbabilityItem(BaseModel):
    dosha: str
    prob: float


class PredictResponse(BaseModel):
    predictedDosha: str
    confidence: float
    probabilities: List[ProbabilityItem]
    remedy: Optional[Dict[str, Any]] = None


# -----------------------------
# Asset loading (startup)
# -----------------------------

X_FULL: Optional[np.ndarray] = None  # encoded ints
Y_FULL: Optional[np.ndarray] = None  # encoded dosha idx ints
FEATURE_COLS: Optional[List[str]] = None
DOSHA_NAMES: Optional[np.ndarray] = None
NUM_SYMPTOMS: Optional[int] = None
NUM_DOSHA: Optional[int] = None
ENCODERS: Optional[Dict[str, LabelEncoder]] = None
MEDIANS: Optional[np.ndarray] = None

K_NEIGHBORS: int = 15

AYUR_DATA: Optional[List[Dict[str, Any]]] = None
SCHEMA_OPTIONS: Optional[Dict[str, Any]] = None

# Precomputed edge indices for the existing patients
PATIENT_SYMPTOM_ROWS: Optional[np.ndarray] = None
PATIENT_SYMPTOM_COLS: Optional[np.ndarray] = None
PATIENT_DOSHA_ROWS: Optional[np.ndarray] = None
PATIENT_DOSHA_COLS: Optional[np.ndarray] = None

MODEL: Optional[torch.nn.Module] = None


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_assets() -> None:
    global X_FULL, Y_FULL, FEATURE_COLS, DOSHA_NAMES, NUM_SYMPTOMS, NUM_DOSHA
    global ENCODERS, MEDIANS, K_NEIGHBORS
    global PATIENT_SYMPTOM_ROWS, PATIENT_SYMPTOM_COLS, PATIENT_DOSHA_ROWS, PATIENT_DOSHA_COLS
    global MODEL, AYUR_DATA, SCHEMA_OPTIONS

    _require_file(MODEL_CARD_PATH)
    _require_file(PRAKRITI_PATH)
    _require_file(AYUR_PATH)
    _require_file(ENCODERS_PATH)
    _require_file(GRAPH_DATA_PATH)
    _require_file(BEST_MODEL_PATH)

    with open(PRAKRITI_PATH, "r") as f:
        prakriti = json.load(f)

    if not prakriti:
        raise RuntimeError("prakriti_clean.json is empty")

    # Feature cols are all keys except label column.
    keys = list(prakriti[0].keys())
    if "Dosha" not in keys:
        raise RuntimeError("Expected label column `Dosha` not found in prakriti_clean.json")

    FEATURE_COLS = [k for k in keys if k != "Dosha"]
    NUM_SYMPTOMS = len(FEATURE_COLS)

    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    ENCODERS = encoders

    dosha_encoder = ENCODERS.get("Dosha")
    if dosha_encoder is None:
        raise RuntimeError("encoders.pkl does not contain encoder for 'Dosha'")
    DOSHA_NAMES = np.array(dosha_encoder.classes_, dtype=object)
    NUM_DOSHA = len(DOSHA_NAMES)

    # Encode entire dataset (predict.py encodes via LabelEncoder on strings).
    X_encoded = []
    Y_encoded = []
    for row in prakriti:
        x_row = []
        for col in FEATURE_COLS:
            v = str(row[col])
            x_row.append(int(ENCODERS[col].transform([v])[0]))
        X_encoded.append(x_row)
        Y_encoded.append(int(dosha_encoder.transform([str(row["Dosha"])])[0]))

    X_FULL = np.asarray(X_encoded, dtype=np.float32)
    Y_FULL = np.asarray(Y_encoded, dtype=np.int64)

    # Load remedy lookup table once (used by /api/predict).
    with open(AYUR_PATH, "r") as f:
        AYUR_DATA = json.load(f)

    # Use the same median-based thresholding logic as the repo's `predict.py`.
    MEDIANS = np.median(X_FULL, axis=0)

    # k-NN for similar_to edge type.
    graph_data = torch.load(GRAPH_DATA_PATH, map_location="cpu", weights_only=False)
    K_NEIGHBORS = int(graph_data.get("k_neighbors", 15))

    # Precompute patient -> symptom edges (threshold on medians).
    rows: List[int] = []
    cols: List[int] = []
    num_patients = X_FULL.shape[0]
    for p in range(num_patients):
        for s in range(NUM_SYMPTOMS):
            if X_FULL[p, s] >= MEDIANS[s]:
                rows.append(p)
                cols.append(s)
    PATIENT_SYMPTOM_ROWS = np.asarray(rows, dtype=np.int64)
    PATIENT_SYMPTOM_COLS = np.asarray(cols, dtype=np.int64)

    # Precompute patient -> dosha edges from ground-truth labels.
    PATIENT_DOSHA_ROWS = np.arange(num_patients, dtype=np.int64)
    PATIENT_DOSHA_COLS = Y_FULL.astype(np.int64)

    # Read model hyperparameters from the saved model card to ensure
    # the architecture matches the checkpoint in `best_model.pt`.
    with open(MODEL_CARD_PATH, "r") as f:
        model_card = json.load(f)
    training_hp = model_card.get("training", {}).get("hyperparams") or {}
    arch = model_card.get("architecture", {}) or {}

    hidden_dim = training_hp.get("hidden") or arch.get("hidden_dim") or 128
    heads = training_hp.get("heads") or arch.get("attention_heads") or 2
    dropout = training_hp.get("dropout") or arch.get("dropout") or 0.3

    # Load the trained model.
    in_ch = {"patient": X_FULL.shape[1], "symptom": NUM_SYMPTOMS, "dosha": NUM_DOSHA}
    MODEL = HeteroDoshaNet(
        in_channels_dict=in_ch,
        hidden_dim=int(hidden_dim),
        num_classes=NUM_DOSHA,
        heads=int(heads),
        dropout=float(dropout),
    )
    MODEL.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True))
    MODEL.eval()

    # Precompute schema options for the form (done once at startup).
    unique = {c: set() for c in FEATURE_COLS}
    doshas = set()
    for row in prakriti:
        for c in FEATURE_COLS:
            unique[c].add(str(row[c]))
        doshas.add(str(row["Dosha"]))
    SCHEMA_OPTIONS = {
        "features": [
            {"name": c, "options": sorted(unique[c], key=lambda s: s.lower())}
            for c in FEATURE_COLS
        ],
        "doshaClasses": sorted(list(doshas), key=lambda s: s.lower()),
    }


@app.on_event("startup")
def _startup() -> None:
    load_assets()


def _encode_new_patient(features: Dict[str, str]) -> np.ndarray:
    if FEATURE_COLS is None or ENCODERS is None:
        raise RuntimeError("Assets not loaded")

    missing = [c for c in FEATURE_COLS if c not in features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing feature(s): {missing}")

    x_row = []
    for col in FEATURE_COLS:
        v = str(features[col])
        try:
            x_row.append(int(ENCODERS[col].transform([v])[0]))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid option for feature '{col}': '{v}'",
            ) from e
    return np.asarray(x_row, dtype=np.float32)


def _get_remedy(dosha_pred: str) -> Optional[Dict[str, Any]]:
    # This logic mirrors `predict.py` and matches the JSON lookup table shape.
    if AYUR_DATA is None:
        return None
    ayur = AYUR_DATA

    parts = [p.strip() for p in str(dosha_pred).replace("+", ",").split(",") if p.strip()]
    parts_lower = [p.lower() for p in parts]

    # Exact multi-dosha match first.
    for row in ayur:
        doshas_in_row = str(row.get("Doshas", "")).lower()
        if all(p in doshas_in_row for p in parts_lower):
            return row

    # Fallback: match first dosha substring.
    if not parts_lower:
        return None
    first = parts_lower[0]
    for row in ayur:
        if first in str(row.get("Doshas", "")).lower():
            return row

    return None


def _build_graph_for_prediction(x_new_encoded: np.ndarray) -> HeteroData:
    """
    Build a hetero graph containing the training patients plus one new patient.
    """

    if (
        X_FULL is None
        or Y_FULL is None
        or FEATURE_COLS is None
        or DOSHA_NAMES is None
        or MEDIANS is None
        or PATIENT_SYMPTOM_ROWS is None
        or PATIENT_SYMPTOM_COLS is None
        or PATIENT_DOSHA_ROWS is None
        or PATIENT_DOSHA_COLS is None
        or NUM_SYMPTOMS is None
        or NUM_DOSHA is None
        or K_NEIGHBORS is None
    ):
        raise RuntimeError("Assets not loaded")

    num_patients = X_FULL.shape[0]
    new_idx = num_patients

    X_full_new = np.vstack([X_FULL, x_new_encoded.reshape(1, -1)]).astype(np.float32)
    # y isn't used by the model for inference; it's only here to keep the graph complete.
    y_full_new = np.concatenate([Y_FULL, np.asarray([0], dtype=np.int64)], axis=0)

    # Patient -> symptom edges: reuse precomputed edges, plus new patient's threshold.
    new_symptom_cols = np.where(x_new_encoded >= MEDIANS)[0].astype(np.int64)
    new_symptom_rows = np.full_like(new_symptom_cols, fill_value=new_idx, dtype=np.int64)
    patient_sym_rows = np.concatenate([PATIENT_SYMPTOM_ROWS, new_symptom_rows], axis=0)
    patient_sym_cols = np.concatenate([PATIENT_SYMPTOM_COLS, new_symptom_cols], axis=0)

    # Patient -> dosha edges:
    # - Existing patients keep their ground-truth belongs_to edge.
    # - New patient connects to ALL dosha nodes (so the model can pick via attention).
    all_dosha_cols = np.arange(NUM_DOSHA, dtype=np.int64)
    new_dosha_rows = np.full_like(all_dosha_cols, fill_value=new_idx, dtype=np.int64)
    patient_dosha_rows = np.concatenate([PATIENT_DOSHA_ROWS, new_dosha_rows], axis=0)
    patient_dosha_cols = np.concatenate([PATIENT_DOSHA_COLS, all_dosha_cols], axis=0)

    # Patient <-> patient similar_to edges:
    # Rebuild including the new patient for correctness.
    adj = kneighbors_graph(
        X_full_new,
        n_neighbors=K_NEIGHBORS,
        metric="cosine",
        include_self=False,
    )
    rows, cols = adj.nonzero()
    patient_sim_rows = rows.astype(np.int64)
    patient_sim_cols = cols.astype(np.int64)

    data = HeteroData()
    data["patient"].x = torch.tensor(X_full_new, dtype=torch.float)
    data["patient"].y = torch.tensor(y_full_new, dtype=torch.long)
    data["symptom"].x = torch.eye(NUM_SYMPTOMS, dtype=torch.float)
    data["dosha"].x = torch.eye(NUM_DOSHA, dtype=torch.float)

    # Build edge_index using torch.from_numpy to avoid slow tensor-from-list warning.
    data["patient", "has_trait", "symptom"].edge_index = torch.from_numpy(
        np.vstack([patient_sym_rows, patient_sym_cols]).astype(np.int64)
    )
    data["patient", "belongs_to", "dosha"].edge_index = torch.from_numpy(
        np.vstack([patient_dosha_rows, patient_dosha_cols]).astype(np.int64)
    )
    data["patient", "similar_to", "patient"].edge_index = torch.from_numpy(
        np.vstack([patient_sim_rows, patient_sim_cols]).astype(np.int64)
    )
    return data


def _compute_prediction(x_new_encoded: np.ndarray) -> Dict[str, Any]:
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    data = _build_graph_for_prediction(x_new_encoded)
    with torch.no_grad():
        log_probs = MODEL(data.x_dict, data.edge_index_dict)

    # New patient is the last node.
    last = log_probs[-1]
    probs = torch.exp(last)
    pred_idx = int(torch.argmax(probs).item())

    predicted_dosha = str(DOSHA_NAMES[pred_idx])

    probabilities = [
        {"dosha": str(DOSHA_NAMES[i]), "prob": float(probs[i].item())}
        for i in range(len(DOSHA_NAMES))
    ]
    confidence = float(probs[pred_idx].item() * 100.0)

    remedy = _get_remedy(predicted_dosha)
    return {
        "predictedDosha": predicted_dosha,
        "confidence": confidence,
        "probabilities": probabilities,
        "remedy": remedy,
    }


def _load_model_card() -> Dict[str, Any]:
    with open(MODEL_CARD_PATH, "r") as f:
        return json.load(f)


def _find_explanation_images() -> List[Dict[str, Any]]:
    # Uses repo-root images produced during training.
    imgs = []
    for p in PROJECT_ROOT.glob("explanation_patient_*.png"):
        stem = p.stem  # explanation_patient_959
        try:
            idx = int(stem.split("_")[-1])
        except Exception:
            continue
        imgs.append({"patientIndex": idx, "url": f"/assets/{p.name}"})
    imgs.sort(key=lambda x: x["patientIndex"])
    return imgs


def _load_schema_options() -> Dict[str, Any]:
    if SCHEMA_OPTIONS is None:
        raise HTTPException(status_code=500, detail="Schema options not initialized")
    return SCHEMA_OPTIONS


@app.get("/api/model-card")
def model_card() -> Dict[str, Any]:
    return _load_model_card()


@app.get("/api/artifacts")
def artifacts() -> Dict[str, Any]:
    return {
        "model_card": _load_model_card(),
        "artifacts": {
            "confusion_matrix_url": "/assets/confusion_matrix.png"
            if (PROJECT_ROOT / "confusion_matrix.png").exists()
            else None,
            "explanations": _find_explanation_images(),
        },
    }


@app.get("/api/schema")
def schema() -> Dict[str, Any]:
    return _load_schema_options()


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> Dict[str, Any]:
    x_new_encoded = _encode_new_patient(req.features)
    return _compute_prediction(x_new_encoded)

