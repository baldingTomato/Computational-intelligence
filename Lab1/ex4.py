import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    iris_df = pd.read_csv("iris.csv")

    features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    X = iris_df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = explained_variance_ratio.cumsum()
    print("Explained variance ratios:", explained_variance_ratio)
    print("Cumulative variance:", cumulative_variance)

    # Use the first two principal components
    X_pca_2d = X_pca[:, :2]

    iris_df_pca = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
    iris_df_pca['variety'] = iris_df['variety']

    plt.figure(figsize=(8, 6))
    colors = {'Setosa': 'red', 'Versicolor': 'green', 'Virginica': 'blue'}
    plt.scatter(iris_df_pca['PC1'], iris_df_pca['PC2'], c=iris_df_pca['variety'].map(colors))

    plt.title('Iris Dataset - PCA (2 components)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    # Legend
    for species, color in colors.items():
        plt.scatter([], [], color=color, label=species)
    plt.legend(title="Variety")

    plt.show()


if __name__ == "__main__":
    main()
