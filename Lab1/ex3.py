import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("iris_with_errors.csv")

    # a) Count missing or incomplete data
    df.replace(to_replace=['-', ' ', 'N/A', 'n/a', 'None', '__', '_'], value=np.nan, inplace=True)
    
    missing_data_count = df.isnull().sum()
    total_missing = df.isnull().sum().sum()

    print("Missing data count per column:")
    print(missing_data_count)
    print(f"\nTotal missing values: {total_missing}\n")
    print("\nDatabase statistics with errors:")
    print(df.describe())

    # b) Check the valid range
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for column in numeric_columns:
        valid_mean = df[(df[column] >= 0) & (df[column] <= 15)][column].mean()
        print(f"Mean for {column}: {valid_mean}")
        df[column] = df[column].mask((df[column] < 0) | (df[column] > 15), valid_mean)

    print(df)

    # c) Check variety column
    valid_variety = ['Setosa', 'Versicolor', 'Virginica']

    invalid_variety = df[~df['variety'].isin(valid_variety)]['variety']
    print("\nIncorrect variety entries found:")
    print(invalid_variety)

    variety_corrections = {
        'setosa': 'Setosa',
        'Versicolour': 'Versicolor',
        'virginica': 'Virginica'
    }

    df['variety'] = df['variety'].replace(variety_corrections)

    print("\nCorrected variety column entries:")
    print(df['variety'].unique())


if __name__ == "__main__":
    main()
