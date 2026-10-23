"""Type any text and get the predicted topic. Run train.py first."""
import joblib

bundle = joblib.load("model.joblib")
model, labels = bundle["model"], bundle["labels"]
print("Type some text (or 'quit'):")
while True:
    text = input("> ").strip()
    if text.lower() in {"quit", "exit", ""}:
        break
    probs = model.predict_proba([text])[0]
    ranked = sorted(zip(labels, probs), key=lambda t: -t[1])
    print("  ->", ", ".join(f"{l} {p:.0%}" for l, p in ranked[:3]))
