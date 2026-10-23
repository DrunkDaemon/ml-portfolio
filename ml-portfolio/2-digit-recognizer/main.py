"""Handwritten digit recognition: compare classifiers, tune the best, show mistakes."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=42)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}
for name, m in models.items():
    print(f"{name:20s} CV accuracy: {cross_val_score(m, X_train, y_train, cv=5).mean():.3f}")

grid = GridSearchCV(models["SVM"], {"svc__C": [1, 5, 10, 50], "svc__gamma": ["scale", 0.001, 0.01]}, cv=5)
grid.fit(X_train, y_train)
print("\nBest SVM params:", grid.best_params_)

pred = grid.predict(X_test)
print(f"TEST accuracy: {accuracy_score(y_test, pred):.4f}\n")
print(classification_report(y_test, pred))

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Confusion matrix"); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

wrong = (pred != y_test).nonzero()[0][:8]
fig, axes = plt.subplots(1, max(len(wrong), 1), figsize=(2 * max(len(wrong), 1), 2.4))
axes = [axes] if len(wrong) <= 1 else axes
for ax, i in zip(axes, wrong):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r"); ax.axis("off")
    ax.set_title(f"true {y_test[i]}\npred {pred[i]}", fontsize=8)
plt.savefig("mistakes.png", dpi=120, bbox_inches="tight"); plt.close()
joblib.dump(grid.best_estimator_, "digit_model.joblib")
print("Saved digit_model.joblib, confusion_matrix.png, mistakes.png")
