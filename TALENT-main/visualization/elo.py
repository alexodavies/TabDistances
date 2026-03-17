import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(42)
np.random.seed(42)

names_dict = {
    "dummy": "Dummy",
    "LogReg": "LR",
    "LinearRegression": "LR",
    "NCM": 'NCM',
    "NaiveBayes": "NB",
    "knn": "KNN",
    "svm": "SVM",
    "xgboost": "XGB",
    "catboost": "CatB",
    "RandomForest": "RForest",
    "lightgbm": "LightG",
    "tabpfn": "TabPFN",
    "mlp": "MLP",
    "resnet": "ResNet",
    "node": "NODE",
    "switchtab": "SwitchT",
    "tabnet": "TabNet",
    "tabcaps": "TabCaps",
    "tangos": "TANGOS",
    "danets": "DANets",
    "ftt": "FT-T",
    "autoint": "AutoInt",
    "dcn2": "DCNv2",
    "snn": "SNN",
    "tabtransformer": "TabT",
    "ptarl": "PTaRL",
    "grownet": "GrowNet",
    "tabr": "TabR",
    "dnnr": "DNNR",
    "realmlp": "RealMLP",
    "mlp_plr": "MLP-PLR",
    "excelformer": "ExcelF",
    "modernNCA": "MNCA",
    "tabm": "TabM",
}

method_types={
    "A": ["Dummy"],
    "B": ["LR", "NCM", "NB", "KNN", "SVM", "DNNR"],
    "C": ["XGB", "CatB", "RForest", "LightG"],
    "D": ["MLP", "SNN", "MLP-PLR", 'RealMLP', "ResNet"],
    "E": ["DCNv2", "DANets", "TabCaps"],
    "F": ["TabNet", "NODE", "GrowNet"],
    "G": ["TabPFN", "MNCA", "TabR"],
    "H": ["FT-T", "AutoInt", "ExcelF", "TabT"],
    "I": ["SwitchT", "TANGOS", "PTaRL"]
}

colors_type = {
    "A": "#42A5F5",  # Light Blue
    "B": "#FF7043",  # Coral Orange
    "C": "#66BB6A",  # Vibrant Green
    "D": "#EF5350",  # Rich Red
    "E": "#5C6BC0",  # Soft Indigo
    "F": "#26A69A",  # Emerald Teal
    "G": "#AB47BC",  # Vivid Purple
    "H": "#26C6DA",  # Bright Cyan
    "I": "#D4E157"   # Fresh Lime
}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['Arial']


def calculate_elo_scores_with_missing_values(df, k_factor=32, initial_elo=1500):
    """
    Calculate ELO scores for multiple methods based on their performance across datasets,
    skipping rows with missing values for any method pair.

    Parameters:
    - df: DataFrame where each row is a dataset and each column is a method's performance.
    - k_factor: K-factor for the ELO calculation.
    - initial_elo: Initial ELO score for each method.

    Returns:
    - elo_scores: A dictionary of final ELO scores for each method.
    """
    methods = df.columns
    elo_scores = {method: initial_elo for method in methods}
    
    for _, row in df.iterrows():
        # Skip rows with missing values
        if row.isnull().any():
            continue
        performances = row
        
        # Compare each pair of methods
        for method_a in methods:
            for method_b in methods:
                if method_a == method_b:
                    continue

                rating_a = elo_scores[method_a]
                rating_b = elo_scores[method_b]
                performance_a = performances[method_a]
                performance_b = performances[method_b]

                # Determine the match result
                if performance_a > performance_b:
                    result_a, result_b = 1, 0
                elif performance_a < performance_b:
                    result_a, result_b = 0, 1
                else:
                    result_a, result_b = 0.5, 0.5

                # Calculate expected scores
                expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

                # Update ELO scores
                elo_scores[method_a] += k_factor * (result_a - expected_a)
                elo_scores[method_b] += k_factor * (result_b - expected_b)

    return elo_scores


def calculate_average_elo_scores(df, k_factor=32, initial_elo=1500, num_shuffles=30):
    """
    Calculate average ELO scores for multiple methods across multiple randomized runs.

    Parameters:
    - df: DataFrame where each row is a dataset and each column is a method's performance.
    - k_factor: K-factor for the ELO calculation.
    - initial_elo: Initial ELO score for each method.
    - num_shuffles: Number of times to shuffle the datasets and calculate ELO.

    Returns:
    - avg_elo_scores: A dictionary of average ELO scores for each method across all shuffles.
    """
    methods = df.columns
    accumulated_elo_scores = {method: 0 for method in methods}

    for _ in range(num_shuffles):
        # Shuffle the datasets (rows)
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        
        # Calculate ELO scores for this shuffled order
        elo_scores = calculate_elo_scores_with_missing_values(shuffled_df, k_factor, initial_elo)
        
        # Accumulate ELO scores from this run
        for method in methods:
            accumulated_elo_scores[method] += elo_scores[method]

    # Calculate the average ELO scores
    avg_elo_scores = {method: score / num_shuffles for method, score in accumulated_elo_scores.items()}
    return avg_elo_scores


def plot_elo_scores(elo_scores, path='pics/elo_scores.pdf'):
    """
    Plot ELO scores as a horizontal bar chart.

    Parameters:
    - elo_scores: Dictionary of ELO scores for each method.
    """
    # Sort the ELO scores in descending order for better visualization
    sorted_elo_scores = dict(sorted(elo_scores.items(), key=lambda item: item[1], reverse=True))
    
    # Extract methods and scores
    methods = list(sorted_elo_scores.keys())
    scores = list(sorted_elo_scores.values())
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.xlim(min(scores) - 50, max(scores) + 100)  # Set x-axis limit slightly above the maximum score
    colors = []
    for model in methods:
        for key, value in method_types.items():
            if model in value:
                colors.append(colors_type[key])
                break

    plt.barh(methods, scores, color=colors, edgecolor='black', alpha=0.5)
    plt.xlabel('ELO Score', fontsize=18)
    plt.ylabel('Method', fontsize=18)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest scores at the top

    # Annotate each bar with its ELO score
    for index, value in enumerate(scores):
        plt.text(value + 5, index, f"{value:.2f}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(path)


# Example usage
# Assume `df` is your DataFrame where rows are datasets and columns are methods.
# Example DataFrame to demonstrate
df_bin = pd.read_excel('acc_bin.xlsx')
df_bin = df_bin.rename(columns=names_dict)
df_bin = df_bin[df_bin.columns[1:]]
df_multi = pd.read_excel('acc_multi.xlsx')
df_multi = df_multi.rename(columns=names_dict)
df_multi = df_multi[df_multi.columns[1:]]
df_reg = pd.read_excel('rmse.xlsx')
df_reg = df_reg.rename(columns=names_dict)
df_reg = df_reg[df_reg.columns[1:]]
df_reg = df_reg.applymap(lambda x: -x if isinstance(x, float) else x)

df_all = pd.read_excel('merged_result.xlsx')
df_all = df_all.rename(columns=names_dict)
df_all = df_all[df_all.columns[1:]]

# Calculate ELO scores
elo_scores = calculate_average_elo_scores(df_bin)
print("Final ELO Scores:", elo_scores)
plot_elo_scores(elo_scores, path='elo_bin.pdf')

elo_scores = calculate_average_elo_scores(df_multi)
print("Final ELO Scores:", elo_scores)
plot_elo_scores(elo_scores, path='elo_multi.pdf')

elo_scores = calculate_average_elo_scores(df_reg)
print("Final ELO Scores:", elo_scores)
plot_elo_scores(elo_scores, path='elo_reg.pdf')

elo_scores = calculate_average_elo_scores(df_all)
print("Final ELO Scores:", elo_scores)
plot_elo_scores(elo_scores, path='elo_all.pdf')