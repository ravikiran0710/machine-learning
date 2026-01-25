import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def task1_find_item_costs(file_path):
    print("\n--- A1: Purchase Data Analysis ---")

    df = pd.read_excel(file_path, sheet_name="Purchase data")

    features = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    target = df['Payment (Rs)'].values

    print("Feature Matrix:\n", features)
    print("Target Vector:\n", target)

    print("Vector Space Dimension:", features.shape[1])
    print("Total Records:", features.shape[0])

    print("Rank of Feature Matrix:", np.linalg.matrix_rank(features))

    pseudo_inv = np.linalg.pinv(features)
    item_costs = pseudo_inv.dot(target)

    print("Estimated Item Costs [Candies, Mangoes, Milk]:")
    print(item_costs)


def task2_label_customers(file_path):
    print("\n--- A2: Rich / Poor Classification ---")

    df = pd.read_excel(file_path, sheet_name="Purchase data")

    df['Class'] = df['Payment (Rs)'].apply(lambda val: "RICH" if val > 200 else "POOR")

    print(df[['Payment (Rs)', 'Class']])
    print("Rule Applied: Payment > 200 = RICH, else POOR")


def task3_irctc_statistical_analysis(file_path):
    print("\n--- A3: IRCTC Stock Analysis ---")

    df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
    prices = df['Price'].values

    print("Mean Price:", np.mean(prices))
    print("Variance:", np.var(prices))

    def mean_manual(arr):
        s = 0
        for x in arr:
            s += x
        return s / len(arr)

    def variance_manual(arr):
        m = mean_manual(arr)
        s = 0
        for x in arr:
            s += (x - m) ** 2
        return s / len(arr)

    print("Custom Mean:", mean_manual(prices))
    print("Custom Variance:", variance_manual(prices))

    wednesday_rows = df[df['Day'] == 'Wed']
    print("Wednesday Mean:", wednesday_rows['Price'].mean())

    april_rows = df[df['Month'] == 'Apr']
    print("April Mean:", april_rows['Price'].mean())

    prob_loss = len(df[df['Chg%'] < 0]) / len(df)
    print("Probability of Loss:", prob_loss)

    wed_profit = wednesday_rows[wednesday_rows['Chg%'] > 0]
    print("Probability of Profit on Wednesday:", len(wed_profit) / len(df))

    print("Conditional P(Profit | Wednesday):", len(wed_profit) / len(wednesday_rows))

    plt.scatter(df['Day'], df['Chg%'])
    plt.title("Change % vs Day")
    plt.xlabel("Day")
    plt.ylabel("Change %")
    plt.show()


def task4_thyroid_basic_info(file_path):
    print("\n--- A4: Thyroid Data Exploration ---")

    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    print("Column Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nStats Summary:\n", df.describe())


def task5_similarity_jaccard_smc(file_path):
    print("\n--- A5: Jaccard and SMC ---")

    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)

    binary_df = (numeric_df != 0).astype(int)

    vec1 = binary_df.iloc[0].values
    vec2 = binary_df.iloc[1].values

    f11 = np.sum((vec1 == 1) & (vec2 == 1))
    f10 = np.sum((vec1 == 1) & (vec2 == 0))
    f01 = np.sum((vec1 == 0) & (vec2 == 1))
    f00 = np.sum((vec1 == 0) & (vec2 == 0))

    den_j = (f01 + f10 + f11)
    if den_j == 0:
        jaccard = 1.0
    else:
        jaccard = f11 / den_j

    den_s = (f00 + f01 + f10 + f11)
    if den_s == 0:
        smc = 0.0
    else:
        smc = (f11 + f00) / den_s

    print("f00 =", f00, "f01 =", f01, "f10 =", f10, "f11 =", f11)
    print("Jaccard Coefficient:", jaccard)
    print("Simple Matching Coefficient:", smc)


def task6_cosine_similarity_score(file_path):
    print("\n--- A6: Cosine Similarity ---")

    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    numeric_data = df.select_dtypes(include=[np.number]).fillna(0)

    vec1 = numeric_data.iloc[0].values
    vec2 = numeric_data.iloc[1].values

    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denominator == 0:
        cos_value = 0.0
    else:
        cos_value = np.dot(vec1, vec2) / denominator

    print("Cosine Similarity:", cos_value)


def task7_draw_similarity_heatmap(file_path):
    print("\n--- A7: Similarity Heatmap ---")

    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    numeric_data = df.select_dtypes(include=[np.number]).fillna(0).iloc[:20]

    sim_matrix = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            a = numeric_data.iloc[i].values
            b = numeric_data.iloc[j].values
            denom = np.linalg.norm(a) * np.linalg.norm(b)

            if denom == 0:
                sim_matrix[i][j] = 0.0
            else:
                sim_matrix[i][j] = np.dot(a, b) / denom

    sns.heatmap(sim_matrix, annot=False)
    plt.title("Cosine Similarity Heatmap")
    plt.show()


def run_all_tasks():
    file_path = "Lab2 Session Data.xlsx"

    task1_find_item_costs(file_path)
    task2_label_customers(file_path)
    task3_irctc_statistical_analysis(file_path)
    task4_thyroid_basic_info(file_path)
    task5_similarity_jaccard_smc(file_path)
    task6_cosine_similarity_score(file_path)
    task7_draw_similarity_heatmap(file_path)


if __name__ == "__main__":
    run_all_tasks()
