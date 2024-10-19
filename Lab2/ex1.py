import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return("Setosa")
    elif pl <= 5:
        return("Virginica")
    else:
        return("Versicolor")


def main():
    df = pd.read_csv("iris.csv")

    (train_set, test_set) = train_test_split(df.values, test_size=0.7, random_state=275542)


    good_predictions = 0
    len = test_set.shape[0]
    for i in range(len):
        if classify_iris(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == test_set[i, 4]:
            good_predictions = good_predictions + 1
    print(good_predictions)
    print(good_predictions / len * 100, "%")

    #print("Liczba próbek: ", test_set.shape[0])

    #train_inputs = train_set[:, 0:4]
    #train_classes = train_set[:, 4]
    #test_inputs = test_set[:, 0:4]
    #test_classes = test_set[:, 4]

    #print("Dane treningowe:\n")
    #print(train_inputs)

    #print("Gatunki treningowe:\n")
    #print(train_classes)

    #print("Dane testowe:\n")
    #print(test_inputs)

    #print("Gatunki testowe:\n")
    #print(test_classes)


if __name__ == "__main__":
    main()
