# main.py

from utils.data_loader import load_sample_data
from utils.vectorizer import vectorize_data
from utils.train_model import train_model
from utils.evaluate_model import evaluate_model
from utils.model_io import save_model, load_model
from utils.predict_single import predict_single_review
from utils.batch_predict import predict_from_csv
from app.gradio_app import launch_gradio_interface

def main():
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_sample_data()

    # Step 2: Vectorize
    vectorizer, X_train_vec, X_test_vec = vectorize_data(X_train, X_test)

    # Step 3: Train model
    model = train_model(X_train_vec, y_train)

    # Step 4: Evaluate
    evaluate_model(model, X_test_vec, y_test)

    # Step 5: Save model
    save_model(model, vectorizer)

    # Optional: Predict single review
    review = "The product was fake and didn't arrive"
    print("Prediction:", predict_single_review(review, model, vectorizer))

    # Optional: Batch prediction from CSV
    predict_from_csv("data/sample_reviews.csv", model, vectorizer)

    # Launch Gradio
    launch_gradio_interface(model, vectorizer)

if __name__ == "__main__":
    main()
