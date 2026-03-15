# Machine Learning

Dieser Ordner enthält alle Projekte und Experimente im Bereich **Machine Learning**.

## Verwendete Bibliothek

- **Scikit-learn** (`sklearn`) – die primäre Bibliothek für klassische Machine-Learning-Algorithmen.

## Installation

```bash
pip install scikit-learn
```

## Inhalt

| Datei | Beschreibung |
|---|---|
| `beispiel_sklearn.py` | Einfaches Beispiel: Klassifikation mit Scikit-learn |

## Schnellstart

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```
