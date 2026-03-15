import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc, f1_score,
)
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1) Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 2) Daten aus dem bestehenden ML-Split übernehmen
# ============================================================
# X_train1, X_test1, y_train1, y_test1 kommen aus der Zelle oben

X_train = X_train1.values.astype(np.float32)
X_test  = X_test1.values.astype(np.float32)
y_train = y_train1.values.astype(np.float32)
y_test  = y_test1.values.astype(np.float32)

FEATURE_NAMES = list(X_train1.columns)

# pos_weight für Class Imbalance
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)

print(f"Train: {len(y_train)}  |  Test: {len(y_test)}")
print(f"Klasse 1: {int(n_pos)}  |  Klasse 0: {int(n_neg)}  |  pos_weight: {pos_weight.item():.2f}")

# ============================================================
# 3) Tensoren & DataLoader
# ============================================================
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).unsqueeze(1).to(device)
X_test_t  = torch.tensor(X_test).to(device)
y_test_t  = torch.tensor(y_test).unsqueeze(1).to(device)

train_ds     = TensorDataset(X_train_t, y_train_t)
BATCH_SIZE   = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


# ============================================================
# 4) MLP-Klasse
# ============================================================
class MLP(nn.Module):
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
# 5) Training
# ============================================================
INPUT_DIM = X_train.shape[1]
LR        = 1e-3
EPOCHS    = 100

model     = MLP(INPUT_DIM).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nArchitektur:\n{model}\n")

loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)

    avg_loss = running_loss / len(train_ds)
    loss_history.append(avg_loss)

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:>3}/{EPOCHS}  Loss: {avg_loss:.4f}")


# ============================================================
# 6) Evaluation
# ============================================================
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)
    true  = y_test_t.cpu().numpy().flatten().astype(int)

f1  = f1_score(true, preds)
acc = (preds == true).mean()
fpr, tpr, _ = roc_curve(true, probs)
roc_auc = auc(fpr, tpr)

print(f"\nAccuracy: {acc:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"AUC-ROC:  {roc_auc:.3f}")
print()
print(classification_report(true, preds, target_names=["Klasse 0", "Klasse 1"]))

# Confusion Matrix + ROC
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

# Loss-Kurve
plt.figure(figsize=(7, 3))
plt.plot(loss_history, color="steelblue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

manual_f1 = f1


# ============================================================
# 7) Optuna
# ============================================================
print("\n" + "=" * 50)
print("Optuna Hyperparameter-Suche")
print("=" * 50)

N_TRIALS      = 30
OPTUNA_EPOCHS = 50


class MLPOptuna(nn.Module):
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
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = torch.optim.Adam(m.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

    m.train()
    for ep in range(OPTUNA_EPOCHS):
        for xb, yb in loader:
            loss = crit(m(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    m.eval()
    with torch.no_grad():
        p = (torch.sigmoid(m(X_test_t)) > 0.5).cpu().numpy().flatten().astype(int)
    score = f1_score(y_test, p)
    print(f"  Trial {trial.number + 1:>2}/{N_TRIALS}  F1={score:.4f}  {hidden}")
    return score


study = optuna.create_study(direction="maximize")
study.optimize(optuna_objective, n_trials=N_TRIALS)

print(f"\nBester F1:  {study.best_value:.4f}  (manuell: {manual_f1:.4f},"
      f" Δ = {study.best_value - manual_f1:+.4f})")
print("\nBeste Parameter:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
