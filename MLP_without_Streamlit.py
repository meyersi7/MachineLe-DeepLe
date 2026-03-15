import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc, f1_score,
)
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import time

# ===========================================================================================
# Daten laden — Simulierte Daten
# ===========================================================================================
print("=" * 60)
print("MLP Binary Classification")
print("=" * 60)

df = None

# Simulierte Daten generieren
np.random.seed(42)
n_rows = 1000
age = np.random.randint(18, 70, n_rows)
income = np.random.normal(50000, 15000, n_rows)
spend_score = np.random.uniform(0, 100, n_rows)
membership_years = np.random.randint(0, 15, n_rows)
gender = np.random.choice(["Male", "Female", "Other"], n_rows)
region = np.random.choice(["North", "South", "East", "West"], n_rows)
device = np.random.choice(["Mobile", "Desktop", "Tablet"], n_rows)
plan_type = np.random.choice(["Basic", "Premium", "Gold"], n_rows)
target = (age * 0.5 + spend_score * 0.8 > 60).astype(int)

df = pd.DataFrame({
    "Age": age, "Income": income, "Spend_Score": spend_score,
    "Membership_Years": membership_years, "Gender": gender,
    "Region": region, "Device": device, "Plan_Type": plan_type,
    "Target": target,
})

# ===========================================================================================
# Zielvariable und Features wählen
# ===========================================================================================
target_col = "Target"

# Verfügbare Features (alles ausser Zielvariable)
available_features = [c for c in df.columns if c != target_col]
selected_features = available_features  # alle Features verwenden

# ===========================================================================================
# MLP Hyperparameter
# ===========================================================================================
n_layers = 2
neurons_per_layer = []
for i in range(n_layers):
    n = max(64 // (2 ** i), 16)
    neurons_per_layer.append(n)

learning_rate = 0.001
dropout_rate = 0.2
batch_size = 64
num_epochs = 100
test_size = 0.2

# ===========================================================================================
# Daten-Vorschau
# ===========================================================================================
print("\n📊 Daten-Vorschau")
print(df.head(10))
print(f"\nShape: {df.shape}")
print("\nTarget-Verteilung:")
print(df[target_col].value_counts(normalize=True))

# ===========================================================================================
# Daten vorbereiten
# ===========================================================================================
def prepare_data(df, selected_features, target_col, test_size):
    """Daten vorbereiten: Encoding, Scaling, Split."""
    work_df = df[selected_features + [target_col]].copy().dropna()

    # Numerische und kategorische Spalten identifizieren
    num_cols = work_df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = work_df[selected_features].select_dtypes(exclude=[np.number]).columns.tolist()

    # Skalieren
    if num_cols:
        scaler = StandardScaler()
        work_df[num_cols] = scaler.fit_transform(work_df[num_cols])

    # Kategorisch encodieren
    if cat_cols:
        ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        work_df[cat_cols] = ord_encoder.fit_transform(work_df[cat_cols])

    X = work_df[selected_features].values
    y = work_df[target_col].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    return X_train, X_test, y_train, y_test, selected_features


X_train, X_test, y_train, y_test, feature_names = prepare_data(
    df, selected_features, target_col, test_size,
)

# pos_weight berechnen
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight_val = n_neg / max(n_pos, 1)

print(
    f"\nTrain: {len(y_train)} Samples | Test: {len(y_test)} Samples | "
    f"pos_weight: {pos_weight_val:.2f} (Positiv: {int(n_pos)}, Negativ: {int(n_neg)})"
)

# Tensoren und DataLoader
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)


# ===========================================================================================
# MLP Modell — dynamisch nach Einstellungen
# ===========================================================================================
def build_mlp(input_size, neurons_per_layer, dropout_rate):
    """Dynamisches MLP basierend auf Einstellungen."""
    layers = []
    in_size = input_size
    for n_units in neurons_per_layer:
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_size = n_units
    layers.append(nn.Linear(in_size, 1))
    return nn.Sequential(*layers)


# ===========================================================================================
# Training starten
# ===========================================================================================
print("\n" + "=" * 60)
print("Training")
print("=" * 60)

input_size = X_train.shape[1]
model = build_mlp(input_size, neurons_per_layer, dropout_rate)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Architektur anzeigen
print("\nModell-Architektur:")
print(model)

# Training
loss_history = []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= len(train_dataset)
    loss_history.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} — Loss: {epoch_loss:.4f}")

train_time = time.time() - start_time
print(f"\nTraining abgeschlossen in {train_time:.1f}s")

# Loss-Kurve
fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
ax_loss.plot(loss_history, color="steelblue", linewidth=1.5)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Training Loss über Epochen")
ax_loss.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =======================================================================================
# Evaluation
# =======================================================================================
print("\n" + "=" * 60)
print("Evaluation")
print("=" * 60)

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.sigmoid(logits).numpy().flatten()
    predicted = (probs > 0.5).astype(int)
    y_true = y_test_t.numpy().flatten().astype(int)

accuracy = (predicted == y_true).mean()
f1 = f1_score(y_true, predicted)
fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

print(f"\nAccuracy: {accuracy * 100:.1f}%")
print(f"F1-Score: {f1:.3f}")
print(f"AUC-ROC:  {roc_auc:.3f}")

# Confusion Matrix und ROC Kurve nebeneinander
fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_true, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Klasse 0", "Klasse 1"])
disp.plot(cmap="Blues", ax=ax_cm)
ax_cm.set_title("Confusion Matrix")

# ROC-Kurve
ax_roc.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {roc_auc:.3f}")
ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC-Kurve")
ax_roc.legend()
ax_roc.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
report = classification_report(y_true, predicted, target_names=["Klasse 0", "Klasse 1"])
print(report)

# =======================================================================================
# Permutation Importance
# =======================================================================================
print("=" * 60)
print("Permutation Importance")
print("=" * 60)

class PyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, torch_model):
        self.torch_model = torch_model
        self.torch_model.eval()
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict(self, X):
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            logits = self.torch_model(inputs)
            return (torch.sigmoid(logits) > 0.5).numpy().flatten().astype(int)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

print("Permutation Importance wird berechnet...")
wrapper = PyTorchWrapper(model)
perm_result = permutation_importance(
    wrapper, X_test, y_test,
    n_repeats=20, random_state=42, scoring="f1",
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm_result.importances_mean,
    "Std": perm_result.importances_std,
}).sort_values("Importance", ascending=False)

fig_pi, ax_pi = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
sorted_df = importance_df.sort_values("Importance", ascending=True)
colors = ["steelblue" if v > 0 else "lightcoral" for v in sorted_df["Importance"]]
ax_pi.barh(sorted_df["Feature"], sorted_df["Importance"],
            xerr=sorted_df["Std"], color=colors)
ax_pi.set_xlabel("Mean F1-Score Drop")
ax_pi.set_title("Permutation Importance")
ax_pi.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.show()

print("\nPermutation Importance Tabelle:")
print(importance_df.to_string(index=False))

# Manuelles F1 merken für Optuna-Vergleich
manual_f1 = f1

print("\nTraining und Evaluation abgeschlossen!")

# ===========================================================================================
# OPTUNA — Automatische Hyperparameter-Suche
# ===========================================================================================
print("\n" + "=" * 60)
print("Optuna — Hyperparameter-Suche")
print("=" * 60)

print(
    "Optuna testet automatisch verschiedene Kombinationen von Hyperparametern "
    "(Anzahl Layers, Neuronen, Learning Rate, Dropout, Batch Size) und findet "
    "die Kombination mit dem besten F1-Score. Jeder 'Trial' trainiert ein neues Modell."
)

n_trials = 30
optuna_epochs = 50

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

input_size = X_train.shape[1]
trial_results = []

def objective(trial):
    # Hyperparameter vorschlagen
    n_layers = trial.suggest_int("n_layers", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    bs = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Dynamisches Modell bauen
    layers = []
    in_size = input_size
    for i in range(n_layers):
        out_size = trial.suggest_int(f"n_units_{i}", 16, 128)
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_size = out_size
    layers.append(nn.Linear(in_size, 1))

    trial_model = nn.Sequential(*layers)
    trial_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trial_optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr)
    trial_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    # Training
    trial_model.train()
    for epoch in range(int(optuna_epochs)):
        for batch_X, batch_y in trial_loader:
            outputs = trial_model(batch_X)
            loss = trial_criterion(outputs, batch_y)
            trial_optimizer.zero_grad()
            loss.backward()
            trial_optimizer.step()

    # Evaluation
    trial_model.eval()
    with torch.no_grad():
        logits = trial_model(X_test_t)
        preds = (torch.sigmoid(logits) > 0.5).numpy().flatten().astype(int)
    score = f1_score(y_test, preds)

    # Progress updaten
    trial_results.append({"trial": trial.number + 1, "f1": score})
    print(f"Trial {trial.number + 1}/{n_trials} — F1: {score:.4f}")

    return score

# Study starten
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=int(n_trials))

print(f"\nOptuna abgeschlossen! Bester F1: {study.best_value:.4f}")

# Ergebnisse anzeigen
print(f"\nBester F1-Score: {study.best_value:.4f}")
diff = study.best_value - manual_f1
print(f"Vergleich zu manuellem Modell: {diff:+.4f} F1")

print("\nBeste Hyperparameter:")
params_df = pd.DataFrame(
    [{"Parameter": k, "Wert": v} for k, v in study.best_params.items()]
)
print(params_df.to_string(index=False))

# Trial-Verlauf
results_df = pd.DataFrame(trial_results)
fig_trials, ax_trials = plt.subplots(figsize=(6, 4))
ax_trials.scatter(results_df["trial"], results_df["f1"],
                  alpha=0.6, color="steelblue", s=30)
ax_trials.axhline(y=study.best_value, color="red", linestyle="--",
                  label=f"Best: {study.best_value:.4f}")
ax_trials.set_xlabel("Trial")
ax_trials.set_ylabel("F1-Score")
ax_trials.set_title("F1-Score pro Trial")
ax_trials.legend()
ax_trials.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Beste Architektur anzeigen
print("\nBeste Modell-Architektur:")
best = study.best_params
arch_lines = []
arch_lines.append(f"Input: {input_size} Features")
for i in range(best["n_layers"]):
    arch_lines.append(f"  → Linear({best[f'n_units_{i}']}) + ReLU + Dropout({best['dropout']:.2f})")
arch_lines.append(f"  → Linear(1) — Output")
arch_lines.append(f"\nLearning Rate: {best['lr']:.6f}")
arch_lines.append(f"Batch Size: {best['batch_size']}")
print("\n".join(arch_lines))

print("\nOptuna Hyperparameter-Suche abgeschlossen!")
