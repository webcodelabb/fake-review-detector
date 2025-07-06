from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Assuming df and train_test_split are already defined and executed
X = df['text']       # input text
y = df['label']      # labels (real = 0, fake = 1)

# Split the data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
