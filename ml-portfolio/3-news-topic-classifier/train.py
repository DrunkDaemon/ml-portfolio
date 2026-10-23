"""NLP: classify news posts into topics with TF-IDF + linear models."""
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

CATS = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast", "rec.autos"]
strip = ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", categories=CATS, remove=strip)
test = fetch_20newsgroups(subset="test", categories=CATS, remove=strip)
labels = [c.split(".")[-1] for c in train.target_names]

def tfidf():
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)

candidates = {
    "Naive Bayes": make_pipeline(tfidf(), MultinomialNB(alpha=0.1)),
    "Logistic Regression": make_pipeline(tfidf(), LogisticRegression(max_iter=1000, C=5)),
}
best_name, best_model, best_acc = None, None, 0
for name, model in candidates.items():
    model.fit(train.data, train.target)
    acc = (model.predict(test.data) == test.target).mean()
    print(f"{name:20s} test accuracy: {acc:.3f}")
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

print(f"\nBest: {best_name}\n")
pred = best_model.predict(test.data)
print(classification_report(test.target, pred, target_names=labels))
ConfusionMatrixDisplay.from_predictions(test.target, pred, display_labels=labels, xticks_rotation=30)
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

# Most predictive words per class (only for the logistic regression model)
lr = candidates["Logistic Regression"]
vec, clf = lr.steps[0][1], lr.steps[1][1]
names = np.array(vec.get_feature_names_out())
print("Top words per topic (Logistic Regression):")
for i, lab in enumerate(labels):
    print(f"  {lab:15s}", ", ".join(names[np.argsort(clf.coef_[i])[-8:][::-1]]))

joblib.dump({"model": best_model, "labels": labels}, "model.joblib")
print("\nSaved model.joblib and confusion_matrix.png")
