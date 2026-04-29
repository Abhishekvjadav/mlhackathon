import torch, pickle, json
import numpy as np

# Load everything
with open("encoders.pkl","rb") as f:
    encoders = pickle.load(f)
gd = torch.load("graph_data.pt", map_location="cpu", weights_only=False)

with open("model_results.json") as f:
    results = json.load(f)

print("✅ encoders.pkl loaded")
print("✅ graph_data.pt loaded")
print(f"✅ model_results.json: GNN={results['test_accuracy']}%")
print(f"✅ Graph: {gd['x'].shape[0]} patients, {len(gd['feature_cols'])} features")
print(f"✅ Dosha classes: {list(gd['dosha_names'])}")
print("\n🎉 All files OK — ready for app.py!")