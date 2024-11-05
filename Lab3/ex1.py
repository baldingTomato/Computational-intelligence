from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model_1 = MLPClassifier(hidden_layer_sizes=(2,), max_iter=3000, random_state=42)
model_1.fit(X_train, y_train)
accuracy_1 = model_1.score(X_test, y_test)
print(f"Model with 1 hidden layer (2 neurons) accuracy: {accuracy_1:.2f}")

model_2 = MLPClassifier(hidden_layer_sizes=(3,), max_iter=3000, random_state=42)
model_2.fit(X_train, y_train)
accuracy_2 = model_2.score(X_test, y_test)
print(f"Model with 1 hidden layer (3 neurons) accuracy: {accuracy_2:.2f}")

model_3 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=4000, random_state=42)
model_3.fit(X_train, y_train)
accuracy_3 = model_3.score(X_test, y_test)
print(f"Model with 2 hidden layers (3 neurons each) accuracy: {accuracy_3:.2f}")

best_accuracy = max(accuracy_1, accuracy_2, accuracy_3)
if best_accuracy == accuracy_1:
    print("Best model: 1 hidden layer with 2 neurons.")
elif best_accuracy == accuracy_2:
    print("Best model: 1 hidden layer with 3 neurons.")
else:
    print("Best model: 2 hidden layers with 3 neurons each.")
