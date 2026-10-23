# Handwritten Digit Recognizer
Compares 4 classifiers on the scikit-learn digits dataset using 5-fold cross-validation,
tunes an SVM with GridSearchCV, and analyses misclassified images.

## Run
    pip install -r ../requirements.txt
    python main.py

## Results (fill in after running)
Test accuracy: 98.33%  |  Best params: {'svc__C': 50, 'svc__gamma': 0.01}
