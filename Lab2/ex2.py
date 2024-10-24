import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    df = pd.read_csv("iris.csv")

    train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

    print("Training Set:")
    print(train_set)
    print("\nTest Set:")
    print(test_set)

    train_inputs = train_set.iloc[:, 0:4]
    train_labels = train_set.iloc[:, 4]
    test_inputs = test_set.iloc[:, 0:4]
    test_labels = test_set.iloc[:, 4]

    clf = DecisionTreeClassifier()

    clf.fit(train_inputs, train_labels)

    tree_text = tree.export_text(clf, feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
    print("\nDecision Tree:\n")
    print(tree_text)

    plt.figure(figsize=(13,8))
    tree.plot_tree(clf, feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], class_names=clf.classes_, filled=True)
    plt.show()

    predictions = clf.predict(test_inputs)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    conf_matrix = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":
    main()
