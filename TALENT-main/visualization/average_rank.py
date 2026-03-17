import matplotlib.pyplot as plt
import pandas as pd

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
    "D": ["MLP", "SNN", "MLP-PLR", 'RealMLP', "ResNet", 'TabM'],
    "E": ["DCNv2", "DANets", "TabCaps"],
    "F": ["TabNet", "NODE", "GrowNet"],
    "G": ["TabPFN", "MNCA", "TabR", "MncaPFN-1-3000"],
    "H": ["FT-T", "AutoInt", "ExcelF", "TabT"],
    "I": ["SwitchT", "TANGOS", "PTaRL"]
}

colors_type = {
    "A": "#FFFFFF",  # White
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


def calculate_ranks_for_each_dataset(file_path):
    """
    Calculate ranks for each model across datasets, using mean scores and standard deviations.
    
    Parameters:
    file_path (str): Path to the Excel file with 'Mean' and 'Std' sheets.
    
    Returns:
    pd.DataFrame: A DataFrame with ranks calculated for each model across datasets.
    """
    # Load the data from Excel sheets
    mean_df = pd.read_excel(file_path, index_col=0)
    mean_df = mean_df.applymap(lambda x: -x if isinstance(x, float) else x)
    
    ranks_df = mean_df
    ranks_df = ranks_df.rank(axis=1, ascending=False)
    ranks_df = ranks_df.fillna(ranks_df.mean())
    print(ranks_df)
    return ranks_df


def plot_average_ranks(ranks_df, path):
    """
    Plots a horizontal bar chart of the average rank for each model.
    
    Parameters:
    ranks_df (pd.DataFrame): DataFrame where each entry is the rank of a model for a given dataset.
    """
    # Calculate the average rank for each model
    average_ranks = ranks_df.mean().sort_values(ascending=False)
    print(average_ranks)
    
    plt.figure(figsize=(10, 8))
    colors = []

    for model in average_ranks.index:
        for key, value in method_types.items():
            if model in value:
                colors.append(colors_type[key])
                print(model, key)
                break
    # Plotting
    plt.figure(figsize=(10, 8))
    print(len(colors))
    bars = plt.barh(average_ranks.index, average_ranks.values, color=colors, edgecolor='black', alpha=0.5)

    # Add rank labels to each bar
    for bar in bars:
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.2f}', va='center', fontsize=10)
    plt.xlim(5, 32)
    plt.xlabel("Average Rank", fontsize=20)
    plt.ylabel("Methods", fontsize=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)


ranks_df = calculate_ranks_for_each_dataset('merged_result.xlsx')
ranks_df = ranks_df.rename(columns=names_dict)
plot_average_ranks(ranks_df, path='average_rank.pdf')