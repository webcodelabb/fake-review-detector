import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import io

# 1. Upload CSV file (with Colab file upload widget)
from google.colab import files
uploaded = files.upload()

# 2. Load uploaded file (assumes single file upload)
filename = list(uploaded.keys())[0]

# Read CSV, assuming no header, and use the first column as text
try:
    # Try reading without a header first
    df = pd.read_csv(io.StringIO(uploaded[filename].decode('utf-8')), header=None)
except Exception as e:
    print(f"Error reading CSV without header: {e}")
    # If that fails, try reading with a header
    try:
        df = pd.read_csv(io.StringIO(uploaded[filename].decode('utf-8')))
    except Exception as e:
        print(f"Error reading CSV with header: {e}")
        raise ValueError("Could not read the CSV file. Please ensure it's a valid CSV.")

# Assume the first column is the text and rename it for clarity
df.rename(columns={df.columns[0]: 'text'}, inplace=True)

# 3. Check if the text column is present and not empty
if 'text' not in df.columns or df['text'].empty:
    raise ValueError("CSV must contain at least one column with text data.")

# 4. Preprocess and vectorize the reviews using the SAME vectorizer you used during training
# Fill potential NaN values with empty strings before vectorizing
X_batch = vectorizer.transform(df['text'].fillna(''))

# 5. Make predictions
predictions = model.predict(X_batch)

# 6. Add predictions to the DataFrame
df['prediction'] = predictions
df['prediction'] = df['prediction'].map({0: 'Real', 1: 'Fake'})

# 7. Display result
display(df.head())

# 8. Downloadable result
df.to_csv("batch_predictions.csv", index=False)
files.download("batch_predictions.csv")

