from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

y_pred = model.predict(X_test_vectorized)

print("📊 Accuracy: ", round(accuracy_score(y_test, y_pred), 2))
print("🎯 Precision:", round(precision_score(y_test, y_pred), 2))
print("🔁 Recall:   ", round(recall_score(y_test, y_pred), 2))
print("\n🧾 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

