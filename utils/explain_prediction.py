import eli5
from IPython.display import display, HTML

def explain_prediction(review):
    """Explains the prediction for a single review by highlighting important words."""
    if 'loaded_vectorizer' not in globals() or 'loaded_model' not in globals():
        return "Model or vectorizer not loaded. Please run the saving and loading steps."

    # To show explanation for a specific instance, use eli5.show_prediction
    # This requires the vectorized input and the model
    explained_instance = eli5.show_prediction(
        loaded_model,
        review,
        vec=loaded_vectorizer,
        feature_names=loaded_vectorizer.get_feature_names_out(),
        target_names=["Real", "Fake"]
    )

    # eli5.show_prediction returns an HTML object, so we just return it
    return explained_instance

# Example usage:
new_review_to_explain = "This product is a complete waste of money and a scam."
print(f"Explaining review: '{new_review_to_explain}'")
display(explain_prediction(new_review_to_explain))

new_review_to_explain_2 = "Highly recommend to everyone."
print(f"\nExplaining review: '{new_review_to_explain_2}'")
display(explain_prediction(new_review_to_explain_2))

