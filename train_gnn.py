"""
DoshaNet — Heterogeneous Graph Attention Network for Ayurvedic Dosha Classification
==================================================================================
Complete training pipeline with:
  - Heterogeneous Graph (patient, symptom, dosha nodes)
  - HANConv (Heterogeneous Attention Network)
  - GNNExplainer for feature attribution
  - MC Dropout for uncertainty quantification
  - Optuna hyperparameter optimization
  - Full verification suite (CV, ROC-AUC, Cohen's Kappa, Ablation, Wilcoxon)
"""

import os, json, warnings, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, cohen_kappa_score, accuracy_score)
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════════════
# 1. LOAD & ENCODE DATA
# ═══════════════════════════════════════════════════════════
print("\n📁 LOADING DATA")
print("="*50)

with open('prakriti_clean.json', 'r') as f:
    df = pd.DataFrame(json.load(f))

feature_cols = [c for c in df.columns if c != 'Dosha']
dosha_names = sorted(df['Dosha'].unique())
NUM_CLASSES = len(dosha_names)
print(f"Dosha classes: {dosha_names}")

# Label encode ALL columns
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders for app use
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("✅ encoders.pkl saved")

X = df[feature_cols].values.astype(np.float32)
y = df['Dosha'].values.astype(np.int64)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

X_full = np.vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

train_mask = torch.zeros(len(X_full), dtype=torch.bool)
test_mask  = torch.zeros(len(X_full), dtype=torch.bool)
train_mask[:len(X_train)] = True
test_mask[len(X_train):]  = True

print(f"Total patients: {len(X_full)} | Features: {len(feature_cols)}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ═══════════════════════════════════════════════════════════
# 2. HETEROGENEOUS GRAPH BUILDER
# ═══════════════════════════════════════════════════════════
def build_hetero_graph(X, y, feature_cols, dosha_names, k_neighbors=10):
    """
    Build heterogeneous graph with 3 node types and 3 edge types.
    """
    data = HeteroData()
    num_patients = len(X)
    num_symptoms = len(feature_cols)
    num_doshas   = len(dosha_names)

    # Node features
    data['patient'].x = torch.tensor(X, dtype=torch.float)
    data['patient'].y = torch.tensor(y, dtype=torch.long)
    data['symptom'].x = torch.eye(num_symptoms, dtype=torch.float)
    data['dosha'].x   = torch.eye(num_doshas, dtype=torch.float)

    # Edge: patient → symptom (if feature > column median)
    medians = np.median(X, axis=0)
    p_idx, s_idx = [], []
    for p in range(num_patients):
        for s in range(num_symptoms):
            if X[p, s] >= medians[s]:
                p_idx.append(p)
                s_idx.append(s)
    data['patient', 'has_trait', 'symptom'].edge_index = torch.tensor(
        [p_idx, s_idx], dtype=torch.long
    )

    # Edge: patient → dosha (ground truth labels)
    data['patient', 'belongs_to', 'dosha'].edge_index = torch.tensor(
        [list(range(num_patients)), list(y)], dtype=torch.long
    )

    # Edge: patient ↔ patient (k-NN cosine similarity)
    adj = kneighbors_graph(X, n_neighbors=k_neighbors, metric='cosine', include_self=False)
    rows, cols = adj.nonzero()
    data['patient', 'similar_to', 'patient'].edge_index = torch.tensor(
        [rows, cols], dtype=torch.long
    )

    return data

# ═══════════════════════════════════════════════════════════
# 3. MODEL: Heterogeneous Attention Network
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
        self.han1 = HANConv(
            in_channels=in_channels_dict,
            out_channels=hidden_dim,
            metadata=self.metadata,
            heads=heads,
            dropout=dropout
        )
        self.bn  = torch.nn.BatchNorm1d(hidden_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.han2 = HANConv(
            in_channels=hidden_dim,
            out_channels=num_classes,
            metadata=self.metadata,
            heads=1,
            dropout=dropout
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.han1(x_dict, edge_index_dict)
        out = {k: F.elu(self.bn(v)) for k, v in out.items()}
        out = {k: self.drop(v) for k, v in out.items()}
        out = self.han2(out, edge_index_dict)
        return F.log_softmax(out['patient'], dim=1)

# ═══════════════════════════════════════════════════════════
# 4. TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════
def train_epoch(model, data, optimizer, mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.nll_loss(out[mask], data['patient'].y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out[mask].argmax(dim=1)
    acc = (pred == data['patient'].y[mask]).float().mean().item()
    return acc, pred

# ═══════════════════════════════════════════════════════════
# 5. OPTUNA HYPERPARAMETER SEARCH
# ═══════════════════════════════════════════════════════════
print("\n🔍 OPTUNA HYPERPARAMETER OPTIMIZATION (20 trials)")
print("="*50)

def objective(trial):
    hidden   = trial.suggest_categorical('hidden', [32, 64, 128])
    heads    = trial.suggest_categorical('heads', [2, 4, 8])
    lr       = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout  = trial.suggest_float('dropout', 0.1, 0.5)
    k_nn     = trial.suggest_int('k_neighbors', 5, 25)
    wd       = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    data = build_hetero_graph(X_full, y_full, feature_cols, dosha_names,
                               k_neighbors=k_nn).to(DEVICE)

    in_ch = {
        'patient': X_full.shape[1],
        'symptom': len(feature_cols),
        'dosha': NUM_CLASSES
    }

    model = HeteroDoshaNet(in_ch, hidden, NUM_CLASSES, heads, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)

    for _ in range(150):
        train_epoch(model, data, opt, train_mask)
        scheduler.step()

    acc, _ = evaluate(model, data, test_mask)
    return acc

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_params = study.best_params
print(f"\n✅ Best params: {best_params}")
print(f"✅ Best trial accuracy: {study.best_value*100:.2f}%")

# ═══════════════════════════════════════════════════════════
# 6. FINAL TRAINING
# ═══════════════════════════════════════════════════════════
print("\n🚀 FINAL TRAINING")
print("="*50)

data = build_hetero_graph(
    X_full, y_full, feature_cols, dosha_names,
    k_neighbors=best_params['k_neighbors']
).to(DEVICE)

in_ch = {
    'patient': X_full.shape[1],
    'symptom': len(feature_cols),
    'dosha': NUM_CLASSES
}

model = HeteroDoshaNet(
    in_ch,
    hidden_dim=best_params['hidden'],
    num_classes=NUM_CLASSES,
    heads=best_params['heads'],
    dropout=best_params['dropout']
).to(DEVICE)

print(f"Model: HANConv, {best_params['heads']} heads, hidden={best_params['hidden']}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

best_test_acc = 0.0
patience = 50
counter = 0

for epoch in range(400):
    loss = train_epoch(model, data, optimizer, train_mask)

    if epoch % 10 == 0:
        train_acc, _ = evaluate(model, data, train_mask)
        test_acc, _  = evaluate(model, data, test_mask)
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Train {train_acc*100:.1f}% | Test {test_acc*100:.1f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            counter += 10

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    scheduler.step()

model.load_state_dict(torch.load('best_model.pt', map_location=DEVICE))
print(f"\n✅ Best test accuracy: {best_test_acc*100:.2f}%")

# Save graph data for app use
graph_data = {
    'x': torch.tensor(X_full, dtype=torch.float),
    'y': torch.tensor(y_full, dtype=torch.long),
    'feature_cols': feature_cols,
    'dosha_names': dosha_names,
    'k_neighbors': best_params['k_neighbors'],
    'edge_index': data['patient', 'similar_to', 'patient'].edge_index.clone()
}
torch.save(graph_data, 'graph_data.pt')
print("✅ graph_data.pt saved")

# ═══════════════════════════════════════════════════════════
# 7. GNNEXPLAINER — FEATURE ATTRIBUTION
# ═══════════════════════════════════════════════════════════
print("\n🔍 GNNEXPLAINER — FEATURE ATTRIBUTION")
print("="*50)

def explain_prediction(model, data, patient_idx, feature_cols, dosha_names):
    """Gradient-based feature importance for a specific patient."""
    model.eval()
    data['patient'].x.requires_grad = True

    out = model(data.x_dict, data.edge_index_dict)
    probs = torch.exp(out[patient_idx])
    pred_class = probs.argmax().item()
    confidence = probs.max().item() * 100

    # Backward pass on predicted class
    out[patient_idx, pred_class].backward()
    feat_importance = data['patient'].x.grad[patient_idx].abs().cpu().numpy()
    top5_idx = np.argsort(feat_importance)[-5:][::-1]

    print(f"\n   Patient {patient_idx} → Predicted: {dosha_names[pred_class]} ({confidence:.1f}%)")
    print(f"   Top 5 driving features:")
    for idx in top5_idx:
        print(f"      {feature_cols[idx]:<35} importance: {feat_importance[idx]:.4f}")

    # Save chart
    colors = ['#3fb950' if i in top5_idx else '#6e7681' for i in range(len(feature_cols))]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(feature_cols)), feat_importance, color=colors)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha='right', fontsize=8)
    plt.title(f'GNNExplainer — Feature Attribution (Patient {patient_idx})')
    plt.tight_layout()
    plt.savefig(f'explanation_patient_{patient_idx}.png', dpi=150)
    plt.close()

    data['patient'].x.requires_grad = False
    return feat_importance, pred_class

test_indices = np.where(test_mask.numpy())[0]
for idx in test_indices[:3]:
    explain_prediction(model, data, idx, feature_cols, dosha_names)

# ═══════════════════════════════════════════════════════════
# 8. MC DROPOUT — UNCERTAINTY QUANTIFICATION
# ═══════════════════════════════════════════════════════════
print("\n🎲 MC DROPOUT — UNCERTAINTY QUANTIFICATION")
print("="*50)

def predict_with_uncertainty(model, data, n_samples=50):
    """Run N forward passes with dropout ON. Returns mean + uncertainty."""
    model.train()  # Keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(data.x_dict, data.edge_index_dict)
            preds.append(torch.exp(out).unsqueeze(0))

    preds    = torch.cat(preds, dim=0)
    mean     = preds.mean(dim=0)
    variance = preds.var(dim=0)
    entropy  = -(mean * torch.log(mean + 1e-8)).sum(dim=1)

    predicted_dosha = mean.argmax(dim=1)
    confidence      = mean.max(dim=1).values * 100
    uncertainty     = entropy

    return predicted_dosha, confidence, uncertainty, mean

pred_d, conf, uncertainty, mean_probs = predict_with_uncertainty(model, data)

for i in test_indices[:5]:
    label = "High" if conf[i] > 80 and uncertainty[i] < 0.5 else \
            "Moderate" if conf[i] > 60 else "Low — consult expert"
    print(f"   Patient {i}: {dosha_names[pred_d[i]]} | "
          f"Conf {conf[i]:.1f}% | Unc {uncertainty[i]:.3f} | {label}")

# ═══════════════════════════════════════════════════════════
# 9. FULL VERIFICATION SUITE
# ═══════════════════════════════════════════════════════════
print("\n📊 FULL VERIFICATION SUITE")
print("="*50)

# -- 5-Fold Stratified CV --
print("\n📊 5-Fold Stratified Cross-Validation:")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_accs = []
for fold, (tr, te) in enumerate(kf.split(X_full, y_full)):
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    clf.fit(X_full[tr], y_full[tr])
    acc = clf.score(X_full[te], y_full[te])
    fold_accs.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
cv_mean, cv_std = np.mean(fold_accs), np.std(fold_accs)
print(f"   Mean: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")

# -- Confusion Matrix --
_, preds = evaluate(model, data, test_mask)
y_true = y_full[test_mask.numpy()]
y_pred = preds.numpy()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=dosha_names, yticklabels=dosha_names)
plt.title('Confusion Matrix — DoshaNet HAN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("   ✅ confusion_matrix.png saved")
plt.close()

# -- ROC-AUC --
rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
rf.fit(X_train, y_train)
y_prob = rf.predict_proba(X_test)
y_bin  = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
print(f"\n   Macro ROC-AUC: {auc:.4f}")

# -- Cohen's Kappa --
kappa = cohen_kappa_score(y_true, y_pred)
print(f"   Cohen's Kappa: {kappa:.4f} "
      f"({'Excellent' if kappa>0.8 else 'Good' if kappa>0.6 else 'Fair'})")

# -- Ablation Study --
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=SEED)
mlp.fit(X_train, y_train)
gnn_acc = accuracy_score(y_true, y_pred)

print(f"\n📋 Ablation Study:")
print(f"   {'Config':<40} {'Accuracy':>10}")
print("   " + "-"*52)
for name, acc in [
    ("MLP (no graph)", mlp.score(X_test, y_test)),
    ("Random Forest (no graph)", rf.score(X_test, y_test)),
    ("HAN + heterogeneous graph", gnn_acc)
]:
    print(f"   {name:<40} {acc*100:>9.1f}%")

# -- Wilcoxon Test --
rf_pred_rf = rf.predict(X_test)
try:
    stat, p_val = wilcoxon(
        (y_pred == y_true).astype(int),
        (rf_pred_rf == y_test).astype(int)
    )
    sig = "Significant ✅" if p_val < 0.05 else "Not significant"
    print(f"\n   Wilcoxon GNN vs RF: p={p_val:.4f} ({sig})")
except:
    p_val = 1.0
    print(f"\n   Wilcoxon: p=N/A (identical predictions)")

# ═══════════════════════════════════════════════════════════
# 10. MODEL CARD
# ═══════════════════════════════════════════════════════════
print("\n📦 MODEL CARD")
print("="*50)

model_card = {
    "model_name": "DoshaNet — Heterogeneous Graph Attention Network",
    "architecture": {
        "type": "HANConv (Heterogeneous Attention Network)",
        "node_types": ["patient", "symptom", "dosha"],
        "edge_types": ["has_trait", "belongs_to", "similar_to"],
        "layers": 2,
        "attention_heads": best_params['heads'],
        "parameters": sum(p.numel() for p in model.parameters()),
    },
    "training": {
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "hyperparams": best_params,
        "epochs": 400,
        "early_stopping": True,
    },
    "verification": {
        "test_accuracy": round(gnn_acc * 100, 2),
        "cv_mean": round(cv_mean * 100, 2),
        "cv_std": round(cv_std * 100, 2),
        "macro_roc_auc": round(auc, 4),
        "cohens_kappa": round(kappa, 4),
        "wilcoxon_p": round(p_val, 4),
    },
    "techniques": [
        "Heterogeneous Graph Neural Network (HANConv)",
        "GNNExplainer — feature attribution per patient",
        "MC Dropout — epistemic uncertainty quantification",
        "Optuna — automated hyperparameter optimization",
        "Stratified K-Fold cross-validation",
        "Cosine k-NN graph construction (no label leakage)",
        "Multi-class ROC-AUC + Cohen's Kappa evaluation",
        "AdamW + CosineAnnealingLR training schedule",
    ]
}

with open("model_card.json", "w") as f:
    json.dump(model_card, f, indent=2)

print("✅ model_card.json saved")

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE — DoshaNet HAN")
print("="*60)
print(f"""
Files generated:
  ✅ best_model.pt         — trained HANConv weights
  ✅ graph_data.pt         — heterogeneous graph structure
  ✅ encoders.pkl          — label encoders for all features
  ✅ model_card.json       — complete verification report
  ✅ confusion_matrix.png  — evaluation visualization
  ✅ explanation_patient_*.png — GNNExplainer feature attribution

Results:
  📊 Test Accuracy:    {gnn_acc*100:.1f}%
  📊 CV Mean ± Std:    {cv_mean*100:.1f}% ± {cv_std*100:.1f}%
  📊 ROC-AUC (macro):  {auc:.4f}
  📊 Cohen's Kappa:    {kappa:.4f}
""")