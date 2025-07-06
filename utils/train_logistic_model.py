from sklearn.linear_model import LogisticRegression

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_vectorized, y_train)

