def predict_single_review(review):
    """Predicts whether a single review is real or fake."""
    # Ensure the vectorizer is available
    if 'vectorizer' not in globals():
        print("Vectorizer not found. Please run the vectorization step first.")
        return None
    # Ensure the model is available
    if 'model' not in globals():
        print("Model not found. Please run the model training step first.")
        return None

    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    return "Fake" if prediction[0] == 1 else "Real"

# Example usage:
new_review = "This is the best product ever!"
print(f"Review: '{new_review}'")
print(f"Prediction: {predict_single_review(new_review)}")

new_review_2 = "This product is a complete waste of money and a scam."
print(f"\nReview: '{new_review_2}'")
print(f"Prediction: {predict_single_review(new_review_2)}")

