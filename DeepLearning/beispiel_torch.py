"""
Beispiel: Einfaches neuronales Netz mit PyTorch (torch)
=========================================================
Dieses Skript zeigt den grundlegenden Aufbau eines linearen Modells,
das Training mit einem synthetischen Datensatz und die Auswertung.
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ── 1. Synthetische Trainingsdaten erstellen ──────────────────────────────────
torch.manual_seed(42)

X = torch.rand(100, 1) * 10          # 100 Datenpunkte im Bereich [0, 10]
y = 2 * X + 1 + torch.randn(100, 1)  # Ziel: y = 2x + 1 + Rauschen

# ── 2. Modell definieren ──────────────────────────────────────────────────────
model = nn.Linear(in_features=1, out_features=1)

# ── 3. Verlustfunktion und Optimierer ─────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ── 4. Training ───────────────────────────────────────────────────────────────
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}  |  Loss: {loss.item():.4f}")

# ── 5. Ergebnis anzeigen ──────────────────────────────────────────────────────
weight = model.weight.item()
bias = model.bias.item()
print(f"\nGelerntes Modell: y = {weight:.4f} * x + {bias:.4f}")
print("(Erwartet: y ≈ 2 * x + 1)")
