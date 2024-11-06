import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv("diabetes.csv")
X = data.drop(columns=["class"])
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(6, 3), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
y_pred = model.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)

fp = confusion[0, 1]
fn = confusion[1, 0]
print(f"False Positives (FP): {fp}, False Negatives (FN): {fn}")

alternative_model = MLPClassifier(hidden_layer_sizes=(12, 9), activation='tanh', max_iter=500, random_state=42)
alternative_model.fit(X_train, y_train)
alt_accuracy = alternative_model.score(X_test, y_test)
y_alt_pred = alternative_model.predict(X_test)
alt_confusion = confusion_matrix(y_test, y_alt_pred)
print(f"\n\nAlternative Model Accuracy: {alt_accuracy:.2f}")
print("Alternative Confusion Matrix:\n", alt_confusion)

fp = alt_confusion[0, 1]
fn = alt_confusion[1, 0]
print(f"For Alternative Matrix: False Positives (FP): {fp}, False Negatives (FN): {fn}")
# FN errors are worse as they imply undiagnosed diabetes, leading to a risk of delayed treatment.
