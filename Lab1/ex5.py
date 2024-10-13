import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def plot_iris_data(iris_df, title, ax, colors):
    ax.scatter(iris_df['sepal.length'], iris_df['sepal.width'], c=iris_df['variety'].map(colors))
    ax.set_title(title)
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.grid(True)

def main():
    iris_df = pd.read_csv("iris.csv")

    colors = {'Setosa': 'red', 'Versicolor': 'green', 'Virginica': 'blue'}

    print("Original Data Statistics:")
    print(iris_df[['sepal.length', 'sepal.width']].describe())

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plot_iris_data(iris_df, "Original Data", axs[0], colors)

    # Min-Max Normalization
    min_max_scaler = MinMaxScaler()
    iris_min_max = iris_df.copy()
    iris_min_max[['sepal.length', 'sepal.width']] = min_max_scaler.fit_transform(iris_df[['sepal.length', 'sepal.width']])
    plot_iris_data(iris_min_max, "Min-Max Normalized Data", axs[1], colors)

    # Z-score Normalization
    z_score_scaler = StandardScaler()
    iris_z_score = iris_df.copy()
    iris_z_score[['sepal.length', 'sepal.width']] = z_score_scaler.fit_transform(iris_df[['sepal.length', 'sepal.width']])
    plot_iris_data(iris_z_score, "Z-Score Scaled Data", axs[2], colors)

    plt.tight_layout()
    plt.show()

    print("\nMin, Max, Mean, Standard Deviation for Original Data:")
    print("Sepal Length:")
    print(f"Min: {iris_df['sepal.length'].min()}, Max: {iris_df['sepal.length'].max()}, Mean: {iris_df['sepal.length'].mean()}, Std: {iris_df['sepal.length'].std()}")
    
    print("Sepal Width:")
    print(f"Min: {iris_df['sepal.width'].min()}, Max: {iris_df['sepal.width'].max()}, Mean: {iris_df['sepal.width'].mean()}, Std: {iris_df['sepal.width'].std()}")

if __name__ == "__main__":
    main()
