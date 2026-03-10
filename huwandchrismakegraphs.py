import pandas as pd
import numpy as np
from tabdistances.graph.builder import build_graph
from tabdistances.data.tabfm.dataset import TabFMDataset
from tabdistances.data.priors.tabicl.dataset import SCMPrior

def preprocess_df(df):
    """
    Bulletproof cleaner: forces every single column into a numeric 
    format (float/int) and removes all NaNs.
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.astype(str)
    
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = df_clean[col].astype(str).astype('category').cat.codes
            
    return df_clean

def process_tabfm():
    print("--- Processing TabFM (Real Data) ---")
    tabfm_graphs = {}
    dataset = TabFMDataset(data_path="data/tabfm")
    
    for i, name, df in dataset.iter_tables():
        print(f"[{i+1}/338] Building TabFM graph: {name}")
        
        # Clean the dataframe, then build the graph
        df_clean = preprocess_df(df)
        G = build_graph(df_clean)
        
        # Store in memory using the dataset name as the key
        tabfm_graphs[name] = G
        
    print(f"Success: Built {len(tabfm_graphs)} TabFM graphs in memory.\n")
    return tabfm_graphs

def process_tabicl(num_tables=338):
    print(f"--- Processing TabICL (Synthetic Data: {num_tables} tables) ---")
    tabicl_graphs = {}
    prior = SCMPrior(batch_size=1, prior_type="mix_scm")
    
    for i in range(num_tables):
        X, y, d, seq_lens, train_sizes = prior.get_batch()
        
        # Convert tensor to DataFrame
        X_data = X[0].cpu().numpy() if hasattr(X[0], 'cpu') else X[0]
        col_names = [f"feature_{j}" for j in range(X_data.shape[1])]
        df = pd.DataFrame(X_data, columns=col_names)
        
        if y is not None:
            y_data = y[0].cpu().numpy() if hasattr(y[0], 'cpu') else y[0]
            df['target'] = y_data
            
        print(f"[{i+1}/{num_tables}] Building TabICL synthetic graph")
        
        # TabICL data is naturally numeric, so we don't need to preprocess it
        G = build_graph(df) 
        
        # Store in memory using a generated name as the key
        tabicl_graphs[f"tabicl_synthetic_{i+1}"] = G
            
    print(f"Success: Built {len(tabicl_graphs)} TabICL graphs in memory.\n")
    return tabicl_graphs

# --- Run the pipeline ---
if __name__ == "__main__":
    # These two dictionaries now hold all your NetworkX graph objects!
    my_tabfm_graphs = process_tabfm()
    my_tabicl_graphs = process_tabicl(num_tables=338)
    
    
    print("All done! You can now analyze `my_tabfm_graphs` and `my_tabicl_graphs`.")