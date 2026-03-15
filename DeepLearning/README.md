# Deep Learning

Dieser Ordner enthält alle Projekte und Experimente im Bereich **Deep Learning**.

## Verwendete Bibliothek

- **PyTorch** (`torch`) – die primäre Bibliothek für den Aufbau und das Training von neuronalen Netzwerken.

## Installation

```bash
pip install torch torchvision torchaudio
```

## Inhalt

| Datei | Beschreibung |
|---|---|
| `beispiel_torch.py` | Einfaches Beispiel: lineares Modell mit PyTorch |

## Schnellstart

```python
import torch
import torch.nn as nn

# Einfaches lineares Modell
model = nn.Linear(1, 1)
print(model)
```
