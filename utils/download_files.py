from google.colab import files

# List of files to download
files_to_download = [
    'sample_reviews.csv',
    'logistic_regression_model.pkl',
    'tfidf_vectorizer.pkl',
    # Uncomment the line below if you have run the batch prediction cell and want to download the output
    # 'batch_predictions.csv'
]

print("Downloading files:")
for file_name in files_to_download:
    try:
        files.download(file_name)
        print(f"- {file_name} downloaded.")
    except Exception as e:
        print(f"- Could not download {file_name}: {e}")

print("\nDownload process complete.")

