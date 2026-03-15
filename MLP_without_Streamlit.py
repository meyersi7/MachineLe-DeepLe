import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc, f1_score,
)
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1) Device — GPU wenn verfügbar
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 2) Daten aus dem bestehenden ML-Split übernehmen
# ============================================================
# X_train1, X_test1, y_train1, y_test1 kommen aus der Zelle oben
# (train_test_split mit test_size=0.25, random_state=0)

# Train/Val Split (aus X_train1 einen Validation-Split erzeugen)
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train1, y_train1, test_size=0.2, random_state=42, stratify=y_train1,
)

# In numpy float32 umwandeln
X_train = X_train_nn.values.astype(np.float32)
X_val   = X_val_nn.values.astype(np.float32)
X_test  = X_test1.values.astype(np.float32)

y_train = y_train_nn.values.astype(np.float32)
y_val   = y_val_nn.values.astype(np.float32)
y_test  = y_test1.values.astype(np.float32)

# Feature-Namen für Permutation Importance
FEATURE_NAMES = list(X_multi.columns)

# pos_weight für Class Imbalance
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)

print(f"\nTrain: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")
print(f"Klasse 1: {int(n_pos)}  |  Klasse 0: {int(n_neg)}  |  pos_weight: {pos_weight.item():.2f}")
print(f"Features: {X_train.shape[1]}")

# ============================================================
# 6) Tensoren & DataLoader
# ============================================================
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).unsqueeze(1).to(device)
X_val_t   = torch.tensor(X_val).to(device)
y_val_t   = torch.tensor(y_val).unsqueeze(1).to(device)
X_test_t  = torch.tensor(X_test).to(device)
y_test_t  = torch.tensor(y_test).unsqueeze(1).to(device)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)


# ============================================================
# 7) MLP-Klasse — Schicht für Schicht, mit BatchNorm
# ============================================================
class MLP(nn.Module):
    """MLP für binäre Klassifikation — jede Schicht explizit definiert."""

    def __init__(self, input_dim, dropout=0.2):
        super().__init__()

        self.fc1   = nn.Linear(input_dim, 64)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2   = nn.Linear(64, 32)
        self.bn2   = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.out   = nn.Linear(32, 1)

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.out(x)
        return x


# ============================================================
# 8) Train-Funktion mit Early Stopping
# ============================================================
def train_model(model, train_loader, val_loader=None,
                epochs=100, lr=1e-3, patience=10, verbose=True):
    """
    Trainiert das Modell.
    Early Stopping: stoppt wenn Val-Loss 'patience' Epochen nicht sinkt.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_history = []
    val_history   = []
    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)

        avg_train = running_loss / len(train_loader.dataset)
        train_history.append(avg_train)

        # --- Validation ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(model(xb), yb).item() * len(xb)
            avg_val = val_loss / len(val_loader.dataset)
            val_history.append(avg_val)

            # Early Stopping Check
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = model.state_dict().copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  Early Stopping bei Epoch {epoch} (patience={patience})")
                    break

            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch:>3}/{epochs}  Train: {avg_train:.4f}  Val: {avg_val:.4f}")
        else:
            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch:>3}/{epochs}  Loss: {avg_train:.4f}")

    # Bestes Modell laden (wenn Early Stopping aktiv war)
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_history, val_history


# ============================================================
# 9) Evaluate-Funktion (mit silent-Modus für Optuna)
# ============================================================
@torch.no_grad()
def evaluate(model, X_t, y_t, plot=True, verbose=True):
    model.eval()
    probs = torch.sigmoid(model(X_t)).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)
    true  = y_t.cpu().numpy().flatten().astype(int)

    acc = (preds == true).mean()
    f1  = f1_score(true, preds)
    fpr, tpr, _ = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)

    if verbose:
        print(f"\n  Accuracy: {acc:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUC-ROC:  {roc_auc:.3f}")
        print()
        print(classification_report(true, preds, target_names=["Klasse 0", "Klasse 1"]))

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        ConfusionMatrixDisplay(
            confusion_matrix(true, preds),
            display_labels=["Klasse 0", "Klasse 1"],
        ).plot(cmap="Blues", ax=ax1)
        ax1.set_title("Confusion Matrix")

        ax2.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax2.set(xlabel="FPR", ylabel="TPR", title="ROC-Kurve")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return f1


# ============================================================
# 10) Training & Evaluation — manuelles Modell
# ============================================================
print("\n" + "=" * 50)
print("Manuelles Modell")
print("=" * 50)

INPUT_DIM  = X_train.shape[1]
DROPOUT    = 0.2
LR         = 1e-3
EPOCHS     = 1000         # höher setzen — Early Stopping bricht ab wenn nötig
PATIENCE   = 15
BATCH_SIZE = 64

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

model = MLP(INPUT_DIM, DROPOUT).to(device)
print(f"\nArchitektur:\n{model}\n")

train_hist, val_hist = train_model(
    model, train_loader, val_loader,
    epochs=EPOCHS, lr=LR, patience=PATIENCE,
)

# Finale Evaluation auf TEST-Set (nicht Val!)
print("\nEvaluation auf Test-Set:")
manual_f1 = evaluate(model, X_test_t, y_test_t)

# Loss-Kurve (Train + Val)
plt.figure(figsize=(7, 3))
plt.plot(train_hist, color="steelblue", label="Train")
if val_hist:
    plt.plot(val_hist, color="coral", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 11) Permutation Importance
# ============================================================
print("\n" + "=" * 50)
print("Permutation Importance")
print("=" * 50)


class PyTorchWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper damit sklearn permutation_importance mit PyTorch funktioniert."""

    def __init__(self, torch_model):
        self.torch_model = torch_model
        self.torch_model.eval()
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict(self, X):
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(device)
            return (torch.sigmoid(self.torch_model(t)) > 0.5).cpu().numpy().flatten().astype(int)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


wrapper = PyTorchWrapper(model)
perm_result = permutation_importance(
    wrapper, X_test, y_test,
    n_repeats=20, random_state=42, scoring="f1",
)

importance_df = pd.DataFrame({
    "Feature":    FEATURE_NAMES,
    "Importance": perm_result.importances_mean,
    "Std":        perm_result.importances_std,
}).sort_values("Importance", ascending=False)

print("\n", importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, max(4, len(FEATURE_NAMES) * 0.35)))
sorted_df = importance_df.sort_values("Importance", ascending=True)
colors = ["steelblue" if v > 0 else "lightcoral" for v in sorted_df["Importance"]]
ax.barh(sorted_df["Feature"], sorted_df["Importance"],
        xerr=sorted_df["Std"], color=colors)
ax.set_xlabel("Mean F1-Score Drop")
ax.set_title("Permutation Importance")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.show()


# ============================================================
# 12) Optuna — Hyperparameter-Suche (auf Validation-Set!)
# ============================================================
print("\n" + "=" * 50)
print("Optuna Hyperparameter-Suche")
print("=" * 50)

N_TRIALS      = 30
OPTUNA_EPOCHS = 100
OPTUNA_PATIENCE = 10


class MLPOptuna(nn.Module):
    """Dynamisches MLP für Optuna (variable Architektur nötig)."""

    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def optuna_objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden   = [trial.suggest_int(f"h{i}", 16, 128) for i in range(n_layers)]
    dropout  = trial.suggest_float("dropout", 0.0, 0.5)
    lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    bs       = trial.suggest_categorical("batch_size", [32, 64, 128])

    m = MLPOptuna(INPUT_DIM, hidden, dropout).to(device)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

    # Train mit Early Stopping auf Validation
    train_model(m, loader, val_loader,
                epochs=OPTUNA_EPOCHS, lr=lr, patience=OPTUNA_PATIENCE, verbose=False)

    # Evaluieren auf VALIDATION-Set (nicht Test!)
    f1 = evaluate(m, X_val_t, y_val_t, plot=False, verbose=False)
    print(f"  Trial {trial.number + 1:>2}/{N_TRIALS}  F1={f1:.4f}  {hidden}")
    return f1


study = optuna.create_study(direction="maximize")
study.optimize(optuna_objective, n_trials=N_TRIALS)

print(f"\nBester Val-F1:  {study.best_value:.4f}")

# Finales Modell mit besten Parametern auf TEST-Set evaluieren
print("\nBestes Optuna-Modell auf Test-Set:")
best = study.best_params
best_hidden = [best[f"h{i}"] for i in range(best["n_layers"])]
best_model = MLPOptuna(INPUT_DIM, best_hidden, best["dropout"]).to(device)
best_loader = DataLoader(train_ds, batch_size=best["batch_size"], shuffle=True)
train_model(best_model, best_loader, val_loader,
            epochs=OPTUNA_EPOCHS, lr=best["lr"], patience=OPTUNA_PATIENCE, verbose=False)
optuna_f1 = evaluate(best_model, X_test_t, y_test_t, plot=False, verbose=True)

print(f"\nVergleich auf Test-Set:  Manuell={manual_f1:.4f}  Optuna={optuna_f1:.4f}"
      f"  Δ={optuna_f1 - manual_f1:+.4f}")

print("\nBeste Parameter:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
