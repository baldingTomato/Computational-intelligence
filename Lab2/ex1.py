import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def classify_iris(sl, sw, pl, pw):
    if pl < 2:
        return("Setosa")
    elif pl >= 5:
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


if __name__ == "__main__":
    main()
