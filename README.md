# Fake Review Detector
by Muhammad Aminu Umar(WebCodeLab)

A complete end-to-end machine learning project to detect fake product reviews using natural language processing (NLP) and classical classification models.

This tool enables:
- Training and testing of a fake/real review classifier
- Real-time prediction for single reviews
- Batch prediction from CSV uploads
- Saving/loading model and vectorizer
- Web interface using Gradio
- Model explainability with ELI5

---

##  Features Overview

###  Data Processing
- Predefined sample dataset of labeled real/fake reviews
- Supports training/test splitting and CSV input/output

###  NLP + Vectorization
- Texts are preprocessed and vectorized using **TF-IDF**
- Model-ready format with `scikit-learn`

###  Classifier Training
- Supports:
  - Logistic Regression (default)
  - Random Forest (optional)
- Easily extendable to other models

###  Model Evaluation
- Outputs: Accuracy, Precision, Recall, F1
- Displays confusion matrix and classification report

###  Single Prediction
- Predict whether a single user review is **real** or **fake**
- Can be used directly in code or via Gradio UI

###  Batch Prediction
- Upload CSV file of reviews
- Returns labeled predictions
- Saves results in a new `batch_predictions.csv`

###  Save/Load Model
- Model and vectorizer are saved using `joblib`
- Reloadable for future use without retraining

###  Gradio Web Interface
- Simple browser UI to test reviews manually
- Optional CSV upload directly from UI

###  Explainability (ELI5)
- Highlight influential words in prediction
- View which tokens drive fake/real classification

