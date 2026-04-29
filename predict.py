"""
DoshaNet — Prediction Engine (HANConv Heterogeneous Graph)
"""
import json, torch, numpy as np, pandas as pd, pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════
# 1. LOAD TRAINED ASSETS
# ═══════════════════════════════════════════════════════════
with open('prakriti_clean.json', 'r') as f:
    prakriti = pd.DataFrame(json.load(f))
with open('ayurgenixai_clean.json', 'r') as f:
    ayur = pd.DataFrame(json.load(f))

feature_cols = [c for c in prakriti.columns if c != 'Dosha']

# Rebuild encoders from prakriti
encoders = {}
df = prakriti.copy()
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X_full = df[feature_cols].values.astype(np.float32)
y_full = df['Dosha'].values
dosha_names = encoders['Dosha'].classes_
NUM_CLASSES = len(dosha_names)

# Load saved graph data
graph_data = torch.load('graph_data.pt', map_location='cpu', weights_only=False)
k_neighbors = graph_data.get('k_neighbors', 15)

# ═══════════════════════════════════════════════════════════
# 2. BUILD INITIAL HETERO GRAPH
# ═══════════════════════════════════════════════════════════
def build_hetero_graph(X, y):
    data = HeteroData()
    num_patients = len(X)
    num_symptoms = len(feature_cols)

    data['patient'].x = torch.tensor(X, dtype=torch.float)
    data['patient'].y = torch.tensor(y, dtype=torch.long)
    data['symptom'].x = torch.eye(num_symptoms, dtype=torch.float)
    data['dosha'].x   = torch.eye(NUM_CLASSES, dtype=torch.float)

    # Patient → symptom edges
    medians = np.median(X, axis=0)
    p_idx, s_idx = [], []
    for p in range(num_patients):
        for s in range(num_symptoms):
            if X[p, s] >= medians[s]:
                p_idx.append(p)
                s_idx.append(s)
    data['patient', 'has_trait', 'symptom'].edge_index = torch.tensor([p_idx, s_idx], dtype=torch.long)

    # Patient → dosha edges
    data['patient', 'belongs_to', 'dosha'].edge_index = torch.tensor(
        [list(range(num_patients)), list(y)], dtype=torch.long
    )

    # Patient ↔ patient (k-NN)
    adj = kneighbors_graph(X, n_neighbors=k_neighbors, metric='cosine', include_self=False)
    rows, cols = adj.nonzero()
    data['patient', 'similar_to', 'patient'].edge_index = torch.tensor([rows, cols], dtype=torch.long)

    return data

base_data = build_hetero_graph(X_full, y_full)

# ═══════════════════════════════════════════════════════════
# 3. MODEL DEFINITION
# ═══════════════════════════════════════════════════════════
class HeteroDoshaNet(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_dim, num_classes, heads=4, dropout=0.3):
        super().__init__()
        self.metadata = (
            ['patient', 'symptom', 'dosha'],
            [('patient', 'has_trait', 'symptom'),
             ('patient', 'belongs_to', 'dosha'),
             ('patient', 'similar_to', 'patient')]
        )
        self.han1 = HANConv(in_channels_dict, hidden_dim, metadata=self.metadata,
                            heads=heads, dropout=dropout)
        self.bn   = torch.nn.BatchNorm1d(hidden_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.han2 = HANConv(hidden_dim, num_classes, metadata=self.metadata,
                            heads=1, dropout=dropout)

    def forward(self, x_dict, edge_index_dict):
        out = self.han1(x_dict, edge_index_dict)
        out = {k: F.elu(self.bn(v)) for k, v in out.items()}
        out = {k: self.drop(v) for k, v in out.items()}
        out = self.han2(out, edge_index_dict)
        return F.log_softmax(out['patient'], dim=1)

# Load trained model
in_ch = {'patient': X_full.shape[1], 'symptom': len(feature_cols), 'dosha': NUM_CLASSES}
model = HeteroDoshaNet(in_ch, hidden_dim=64, num_classes=NUM_CLASSES, heads=4, dropout=0.3)
model.load_state_dict(torch.load('best_model.pt', map_location='cpu', weights_only=True))
model.eval()
print("✅ HANConv model loaded!")

# ═══════════════════════════════════════════════════════════
# 4. REMEDY LOOKUP
# ═══════════════════════════════════════════════════════════
def get_remedy(dosha_pred):
    parts = [p.strip() for p in dosha_pred.replace('+', ',').split(',')]

    # Exact multi-dosha match first
    for _, row in ayur.iterrows():
        doshas_in_row = str(row['Doshas']).lower()
        if all(p in doshas_in_row for p in parts):
            return row

    # Fallback: match first dosha
    for _, row in ayur.iterrows():
        if parts[0] in str(row['Doshas']).lower():
            return row

    return None

# ═══════════════════════════════════════════════════════════
# 5. PREDICT FUNCTION
# ═══════════════════════════════════════════════════════════
def predict_patient(patient_idx):
    with torch.no_grad():
        out  = model(base_data.x_dict, base_data.edge_index_dict)
        pred = out[patient_idx].argmax().item()
        prob = torch.exp(out[patient_idx])
        dosha = dosha_names[pred]

    print(f"\n{'='*50}")
    print(f"🧬 PATIENT {patient_idx} ANALYSIS")
    print(f"{'='*50}")
    print(f"📊 Predicted Dosha : {dosha.upper()}")
    print(f"📈 Confidence      : {prob[pred]*100:.1f}%")
    print(f"\n📋 Dosha Probabilities:")
    for i, name in enumerate(dosha_names):
        bar = '█' * int(prob[i]*20)
        print(f"   {name:<12} {prob[i]*100:5.1f}% {bar}")

    remedy = get_remedy(dosha)
    if remedy is not None:
        print(f"\n{'='*50}")
        print(f"🌿 AYURVEDIC REMEDY")
        print(f"{'='*50}")
        print(f"🌱 Herbs     : {remedy['Ayurvedic Herbs']}")
        print(f"💊 Formulation: {remedy['Formulation']}")
        print(f"🍽️  Diet      : {remedy['Diet and Lifestyle Recommendations']}")
        print(f"🧘 Yoga      : {remedy['Yoga & Physical Therapy']}")
        print(f"🛡️  Prevention: {remedy['Prevention']}")
    else:
        print("⚠️  No remedy found")

    return dosha

# ═══════════════════════════════════════════════════════════
# 6. TEST
# ═══════════════════════════════════════════════════════════
print("\n🚀 Testing predictions on 5 patients...")
for i in [0, 1, 2, 10, 50]:
    predict_patient(i)

print("\n✅ Prediction engine ready!")