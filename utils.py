import pandas as pd
import os
from sentence_transformers import InputExample

COL_TOK = "[COL]"
VAL_TOK = "[VAL]"

def serialize_row(row, columns=None):
    serialized_parts = []
    cols_to_process = columns if columns else row.index
    for col in cols_to_process:
        if col not in row.index: continue
        val = row[col]
        # Skip ID column in the text serialization to avoid bias
        if 'id' == str(col).lower() or pd.isna(val) or str(val).strip() == "": continue
        
        clean_col = str(col).strip().lower()
        clean_val = str(val).strip()
        serialized_parts.append(f"{COL_TOK} {clean_col} {VAL_TOK} {clean_val}")
    return " ".join(serialized_parts)

def load_data_pandas(folder_path, left_file, right_file, matches_file, 
                     encoding='utf-8', 
                     match_left_col='id1', match_right_col='id2', sep_truth=",", sep=","):
    
    path_l = os.path.join(folder_path, left_file)
    path_r = os.path.join(folder_path, right_file)
    path_m = os.path.join(folder_path, matches_file)

    # Read CSVs
    if "test_dblp" in path_l:
       COLS_TO_USE = ["id","author1","author2","title","year"]
       df_left = pd.read_csv(path_l, encoding=encoding, names=COLS_TO_USE,sep=sep, on_bad_lines='skip')
       df_right = pd.read_csv(path_r, encoding=encoding, names=COLS_TO_USE,sep=sep,on_bad_lines='skip')
       df_matches = pd.read_csv(path_m, encoding=encoding, sep=sep_truth)
    else:
       df_left = pd.read_csv(path_l, encoding=encoding, sep=sep)
       df_right = pd.read_csv(path_r, encoding=encoding, sep=sep)
       df_matches = pd.read_csv(path_m, encoding=encoding, sep=sep_truth)

    # --- FIX: Smartly find the ID column ---
    def find_id_column(df, filename):
        # 1. Look for explicit 'id' column (case-insensitive)
        for col in df.columns:
            if str(col).lower().strip() == 'id':
                return col
        
        # 2. Fallback: Use the first column if no 'id' found
        print(f"[WARNING] No 'id' column found in {filename}. Using first column '{df.columns[0]}' as ID.")
        return df.columns[0]

    left_id_col = find_id_column(df_left, left_file)
    right_id_col = find_id_column(df_right, right_file)

    # Convert IDs to string
    df_left[left_id_col] = df_left[left_id_col].astype(str)
    df_right[right_id_col] = df_right[right_id_col].astype(str)
    
    # Rename to standard 'id' and set as index
    df_left.rename(columns={left_id_col: 'id'}, inplace=True)
    df_right.rename(columns={right_id_col: 'id'}, inplace=True)
    
    df_left = df_left.drop_duplicates(subset=['id']).set_index('id')
    df_right = df_right.drop_duplicates(subset=['id']).set_index('id')

    # --- Standardize Matches File ---
    # Ensure match columns exist
    if match_left_col not in df_matches.columns or match_right_col not in df_matches.columns:
        raise ValueError(f"Match columns '{match_left_col}'/'{match_right_col}' not found in {matches_file}. Found: {df_matches.columns.tolist()}")

    df_matches[match_left_col] = df_matches[match_left_col].astype(str)
    df_matches[match_right_col] = df_matches[match_right_col].astype(str)
    
    # Rename matches to standard 'left_id', 'right_id'
    df_matches = df_matches.rename(columns={match_left_col: 'left_id', match_right_col: 'right_id'})

    # --- FIX: Filter Invalid Matches ---
    # Only keep matches where the IDs actually exist in the data files
    valid_left_ids = set(df_left.index)
    valid_right_ids = set(df_right.index)
    
    original_len = len(df_matches)
    df_matches = df_matches[
        df_matches['left_id'].isin(valid_left_ids) & 
        df_matches['right_id'].isin(valid_right_ids)
    ]
    
    if len(df_matches) < original_len:
        print(f"Dropped {original_len - len(df_matches)} pairs because IDs were missing in source files.")

    return df_left, df_right, df_matches

def create_input_examples(df_left, df_right, df_matches, left_cols=None, right_cols=None):
    examples = []
    # Use standard 'left_id' and 'right_id'
    for _, row in df_matches.iterrows():
        lid, rid = row['left_id'], row['right_id']
        # The DataFrame is already filtered in load_data_pandas, so we can trust these IDs exist
        text_a = serialize_row(df_left.loc[lid], columns=left_cols)
        text_b = serialize_row(df_right.loc[rid], columns=right_cols)
        examples.append(InputExample(texts=[text_a, text_b], label=1))
            
    return examples
