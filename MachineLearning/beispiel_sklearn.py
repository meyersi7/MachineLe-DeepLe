"""
Beispiel: Klassifikation mit Scikit-learn
==========================================
Dieses Skript zeigt den grundlegenden Arbeitsablauf für ein
Klassifikationsproblem: Daten laden, Modell trainieren und bewerten.
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Datensatz laden ────────────────────────────────────────────────────────
data = load_iris()
X, y = data.data, data.target
klassen = data.target_names

print(f"Datensatz: Iris")
print(f"Features:  {data.feature_names}")
print(f"Klassen:   {list(klassen)}")
print(f"Datenpunkte: {len(X)}\n")

# ── 2. Trainings- und Testdaten aufteilen ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Modell trainieren ──────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 4. Modell bewerten ────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Genauigkeit (Accuracy): {accuracy * 100:.1f}%\n")
print("Detaillierter Bericht:")
print(classification_report(y_test, y_pred, target_names=klassen))
