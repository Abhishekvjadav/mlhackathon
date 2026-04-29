# DoshaNet - Heterogeneous GNN for Ayurvedic Dosha Classification

DoshaNet is a heterogeneous graph attention network (HANConv) that predicts Ayurvedic dosha types from patient symptom features and links predictions to Ayurvedic remedies. The project includes training, evaluation, explainability outputs, and an inference script that prints per-patient predictions and remedy suggestions.

## Project structure

- `train_gnn.py` - End-to-end training pipeline with Optuna tuning, k-NN graph construction, evaluation, and explainability.
- `predict.py` - Loads a trained model and graph, runs predictions, and prints dosha probabilities and remedies.
- `test_predict.py` - Quick sanity checks for saved assets.
- `prakriti_clean.json` - Patient feature dataset with `Dosha` label.
- `ayurgenixai_clean.json` - Remedy lookup table by dosha.
- `best_model.pt` - Trained model weights.
- `graph_data.pt` - Saved graph data and metadata from training.
- `encoders.pkl` - Label encoders saved during training.
- `model_card.json` - Architecture, hyperparameters, and evaluation metrics.
- `confusion_matrix.png`, `explanation_patient_*.png` - Training outputs and explanations.

## Model overview

- Heterogeneous graph with node types: patient, symptom, dosha.
- Edge types: patient->symptom (has_trait), patient->dosha (belongs_to), patient<->patient (similar_to via cosine k-NN).
- Two-layer HANConv with dropout and batch norm.
- Training: AdamW + CosineAnnealingLR, early stopping, Optuna hyperparameter search.
- Evaluation: stratified split, CV, macro ROC-AUC, Cohen's kappa, Wilcoxon test.

## Environment setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torch-geometric scikit-learn pandas numpy optuna matplotlib seaborn scipy
```

Note: `torch-geometric` may require additional wheels depending on your system and CUDA. If install fails, follow the official PyG install guide.

## Train the model

```bash
python train_gnn.py
```

This will:

- Encode the dataset and save `encoders.pkl`.
- Build a heterogeneous graph and save `graph_data.pt`.
- Run Optuna tuning and train a HANConv model.
- Save the best model to `best_model.pt`.
- Write plots such as `confusion_matrix.png` and explanation images.

## Run inference

```bash
python predict.py
```

This script loads `best_model.pt` and `graph_data.pt`, runs predictions for sample patients, and prints:

- Predicted dosha and confidence.
- Probability distribution across dosha classes.
- Remedy guidance from `ayurgenixai_clean.json`.

## Validate saved assets

```bash
python test_predict.py
```

## Model card

See `model_card.json` for a concise summary of architecture, hyperparameters, and evaluation metrics.

## Notes

- The datasets are JSON tables; the label column is `Dosha`.
- The model uses cosine k-NN to connect similar patients without leaking labels.
- Output plots and explanations are generated in the project root.
