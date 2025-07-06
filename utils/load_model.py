import joblib

# Load the model and vectorizer
loaded_model = joblib.load('logistic_regression_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

print("Model and vectorizer loaded.")

# You can now use loaded_model and loaded_vectorizer for predictions
# Example:
# new_review = "This is a test review."
# loaded_review_vectorized = loaded_vectorizer.transform([new_review])
# loaded_prediction = loaded_model.predict(loaded_review_vectorized)
# print(f"Prediction using loaded model: {'Fake' if loaded_prediction[0] == 1 else 'Real'}")

