import pandas as pd

# Expanded dataset
sample_data = {
    'text': [
        "This product is amazing!, I would buy it again.",
        "Absolutely terrible. waste of money.",
        "Best purchase I've made this year.",
        "Fake product. Never received my order.",
        "Highly recommend to everyone.",
        "I love it! It works exactly as described.",
        "Worst experience ever. Do not buy.",
        "This is a scam. I'm reporting this seller.",
        "Fantastic quality and fast shipping.",
        "Too good to be true. Looks fake.",
        "Genuine product. Met all my expectations.",
        "Did not arrive. Probably a fake listing.",
        "Extremely satisfied. Will recommend to friends.",
        "The packaging was fake and item was broken.",
        "Really happy with my purchase.",
        "Fraudulent seller. Avoid at all costs.",
        "Excellent build quality and fast delivery.",
        "I got an empty box. What a scam!",
        "Everything was as described. Very happy.",
        "Product is not as advertised. Total rip-off.",
    ],
    'label': [
        0,  # Real
        1,  # Fake
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1
    ]
}

df = pd.DataFrame(sample_data)

from sklearn.model_selection import train_test_split

X = df['text']       # input text
y = df['label']      # labels (real = 0, fake = 1)

# Split the data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df_sample = pd.DataFrame(sample_data)
df_sample.to_csv("sample_reviews.csv", index=False)

from google.colab import files
files.download("sample_reviews.csv")

