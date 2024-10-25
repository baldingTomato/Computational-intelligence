import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_classifier(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
        
    print(f"{name} - Accuracy: {accuracy * 100:.2f}%")
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
        
    return accuracy

def main():
    df = pd.read_csv("iris.csv")

    X = df.drop(columns=["variety"])
    y = df["variety"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classifiers = {
        "3NN": KNeighborsClassifier(n_neighbors=3),
        "5NN": KNeighborsClassifier(n_neighbors=5),
        "11NN": KNeighborsClassifier(n_neighbors=11),
        "Naive Bayes": GaussianNB()
    }  

    # Trening i ewaluacja klasyfikatorów
    accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        accuracies[name] = evaluate_classifier(name, clf, X_test, y_test)

    best_classifier = max(accuracies, key=accuracies.get)
    print("\nPorównanie dokładności:")
    for name, acc in accuracies.items():
        print(f"{name}: {acc * 100:.2f}%")
    print(f"\nNajlepszy klasyfikator: {best_classifier} z dokładnością {accuracies[best_classifier] * 100:.2f}%")


if __name__ == "__main__":
    main()
