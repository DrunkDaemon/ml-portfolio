# ML Portfolio

Three machine learning projects in Python using scikit-learn.

## Projects

1. **House Price Predictor** (`1-house-price-predictor`)
   Predicts California house prices. Compares four models with cross-validation, tunes the best one, explains which features matter, and includes a Streamlit app for live predictions.

2. **Digit Recognizer** (`2-digit-recognizer`)
   Classifies handwritten digits. Compares four classifiers, tunes an SVM with GridSearchCV, and shows which images the model gets wrong.

3. **News Topic Classifier** (`3-news-topic-classifier`)
   Classifies news posts into five topics using TF-IDF features with Naive Bayes and Logistic Regression, and shows the most predictive words per topic.

## How to Run

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Then run each project from its folder:

```bash
# Project 1
cd 1-house-price-predictor
python train.py
streamlit run app.py

# Project 2
cd 2-digit-recognizer
python main.py

# Project 3
cd 3-news-topic-classifier
python train.py
python predict.py
```

Each training script prints its results and saves charts. Model files are created when you run the scripts.
