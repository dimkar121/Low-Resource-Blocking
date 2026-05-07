import pandas as pd
import torch
import gc
import time
import os
import random
import numpy as np
import faiss 
from sentence_transformers import SentenceTransformer, losses, models, InputExample, util
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Import from utils (or fallback) ---
try:
    from utils import load_data_pandas, serialize_row
except ImportError:
    print("Warning: utils.py not found. Using fallback functions.")
    def serialize_row(row, columns):
        return " ".join([f"{col} {val}" for col, val in row.items() if pd.notna(val)])

# --- CONFIGURATION ---
DATA_PATH = "./data/"
BATCH_SIZE = 16
RESULTS_FILE = "results_blocking.csv"

TRAIN_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30]
NUM_EPOCHS = 4
NEG_RATIO = 5
TOP_K_RETRIEVAL = 5  # Number of neighbors FAISS will retrieve

# List of Models
EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "Arctic": "Snowflake/snowflake-arctic-embed-xs",
    "GTE": "thenlper/gte-small",
    "BGE": "BAAI/bge-small-en-v1.5",
    "E5": "intfloat/e5-small-v2"
}

TASKS = [
    {
     "name":"ABT-BUY",
     "left": "Abt.csv", "right": "Buy.csv", "matches": "truth_abt_buy.csv",
     "encoding": "unicode_escape", "sep_truth": ",", "sep" :",",
     "match_left_col": "idAbt", "match_right_col": "idBuy",
     "left_cols": ["name", "description"], "right_cols": ["name","description"],
     "left_cols_tfidf": ["name"], "right_cols_tfidf": ["name"],
    },
    {
     "name":"ACM-DBLP",
     "left": "ACM.csv", "right": "DBLP.csv", "matches": "truth_ACM_DBLP.csv",
     "encoding": "unicode_escape", "sep_truth": ",", "sep" :",",
     "match_left_col": "idACM", "match_right_col": "idDBLP",
     "left_cols": ["title", "authors","venue"], "right_cols": ["title","authors","venue"]
    },    
    {
     "name": "Amazon-Google",
     "left": "Amazon.csv", "right": "GoogleProducts.csv", "matches": "truth_amazon_google.csv",
     "encoding": "unicode_escape", "sep_truth": ",", "sep" :",",
     "match_left_col": "idAmazon", "match_right_col": "idGoogle",
     "left_cols": ["name", "description"], "right_cols": ["name","description"]
    },
    {
     "name": "Amazon-Walmart",
     "left": "amazon_products.csv", "right": "walmart_products.csv", "matches": "truth_amazon_walmart.tsv",
     "encoding": "unicode_escape", "sep_truth": "\t", "sep" :",",
     "match_left_col": "id1", "match_right_col": "id2",
     "left_cols": ["longdescr", "shortdescr", "title"], "right_cols": ["longdescr", "shortdescr", "title"]
    },
    {
     "name": "Scholar-DBLP",
     "left": "Scholar.csv", "right": "DBLP.csv", "matches": "truth_Scholar_DBLP.csv",
     "encoding": "unicode_escape", "sep_truth": ",", "sep" :",",
     "match_left_col": "idScholar", "match_right_col": "idDBLP",
     "left_cols": ["authors", "title","venue","year"], "right_cols": ["authors","title","venue","year"]
    },
    {  "name": "DBLP",
       "left": "test_dblp_A.txt", "right": "test_dblp_B.txt", "matches": "truth_DBLP.csv",
       "encoding": "utf-8", "sep_truth": ",", "sep" :",",
       "match_left_col": "id1", "match_right_col": "id2",
       "left_cols": ["author1", "author2","title", "year"], "right_cols": ["author1","author2","title","year"]
  },

]

# --- HELPER: Create Pos+Neg InputExamples AND track explicitly generated pairs ---
def create_examples_and_track(df_left, df_right, df_matches, left_cols, right_cols, vectorizer, tfidf_matrix, tfidf_left_cols=None, neg_ratio=1):
    examples = []
    pairs_used = set()
    if tfidf_left_cols is None:
          tfidf_left_cols = left_cols
          
    corpus_ids = list(df_right.index)
    
    # Cache the FULL Left texts for training, and NARROW Left texts for searching
    left_cache_full = {}
    left_cache_tfidf = {}
    for lid in df_matches['left_id'].unique():
        if lid in df_left.index:
            left_cache_full[lid] = serialize_row(df_left.loc[lid], columns=left_cols)
            left_cache_tfidf[lid] = serialize_row(df_left.loc[lid], columns=tfidf_left_cols)
    
    for _, row in df_matches.iterrows():
        lid, rid = row['left_id'], row['right_id']
        
        if lid not in left_cache_full or rid not in df_right.index:
            continue
            
        text_l_full = left_cache_full[lid]
        text_r_full = serialize_row(df_right.loc[rid], columns=right_cols)
        
        # 1. Positive Pair
        examples.append(InputExample(texts=[text_l_full, text_r_full], label=1))
        pairs_used.add((lid, rid))
        
        # 2. Hard Negative Pairs (Searching using PRE-BUILT Index)
        text_l_narrow = left_cache_tfidf[lid]
        query_vec = vectorizer.transform([text_l_narrow])
        cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
        
        top_indices = cosine_similarities.argsort()[-(neg_ratio + 5):][::-1]
        
        negatives_added = 0
        for idx in top_indices:
            hard_neg_rid = corpus_ids[idx]
            
            if hard_neg_rid == rid: 
                continue 
            
            text_neg_full = serialize_row(df_right.loc[hard_neg_rid], columns=right_cols)
            examples.append(InputExample(texts=[text_l_full, text_neg_full], label=0))
            pairs_used.add((lid, hard_neg_rid))
            
            negatives_added += 1
            if negatives_added >= neg_ratio:
                break
            
    return examples, pairs_used

# --- HELPER: Evaluation using FAISS HNSW ---
def evaluate_with_faiss_hnsw(model, df_l, df_r, test_matches_df, seen_pairs, left_cols, right_cols, k=5):
    """ Pure Top-K Evaluation for Blocking """
    
    # 1. Prepare Corpus & Queries
    corpus_ids = list(df_r.index)
    corpus_texts = [serialize_row(df_r.loc[rid], columns=right_cols) for rid in corpus_ids]
    
    query_ids = list(df_l.index)
    query_texts = [serialize_row(df_l.loc[qid], columns=left_cols) for qid in query_ids]
    
    # 2. Encode
    t_enc_start = time.time()
    corpus_embeddings = model.encode(corpus_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    query_embeddings = model.encode(query_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    enc_time = time.time() - t_enc_start
    
    # 3. Build Index & Retrieve
    t_res_start = time.time()
    d = corpus_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT) 
    index.hnsw.efConstruction = 40
    index.add(corpus_embeddings)
    index.hnsw.efSearch = 64
    
    actual_k = min(k, len(corpus_ids))
    scores, indices = index.search(query_embeddings, actual_k)
    res_time = time.time() - t_res_start
    
    # 4. Evaluate Pure @K metrics
    test_matches_set = set(zip(test_matches_df['left_id'], test_matches_df['right_id']))
    unique_test_queries = set(test_matches_df['left_id'])
    
    TP, FP = 0, 0
    
    for i, qid in enumerate(query_ids):
        if qid not in unique_test_queries:
            continue # We only evaluate queries that have a true match
            
        for j in range(actual_k):
            rid = corpus_ids[indices[i][j]]
            pair = (qid, rid)
            
            if pair in seen_pairs: # Leakage filter
                continue
                
            if pair in test_matches_set:
                TP += 1
            else:
                FP += 1

    FN = len(test_matches_set) - TP
    
    recall_at_k = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision_at_k = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0
    
    del index
    gc.collect()
    
    return recall_at_k, precision_at_k, f1_at_k, enc_time, res_time

# --- CORE EXPERIMENT CYCLE ---
def run_training_cycle(model_name, base_model, df_l, df_r, df_train_pool, df_test, task_info, vectorizer, tfidf_matrix):
    results = []
    
    for fraction in TRAIN_FRACTIONS:
        print(f"    >> Training Fraction: {fraction*100}%")
        
        if fraction == 0.0:
            seen_pairs = []
            r_at_k, p_at_k, f1_at_k, enc_time, res_time = evaluate_with_faiss_hnsw(
                base_model, df_l, df_r, df_test, seen_pairs,  
                task_info['left_cols'], task_info['right_cols'], k=TOP_K_RETRIEVAL
            )
            
            print(f"       [0-Shot] R@{TOP_K_RETRIEVAL}: {r_at_k:.4f} | P@{TOP_K_RETRIEVAL}: {p_at_k:.4f} | F1@{TOP_K_RETRIEVAL}: {f1_at_k:.4f}")
            
            results.append({
                "Dataset": task_info['name'],
                "Model": model_name,
                "Train Fraction": 0.0,
                f"Recall@{TOP_K_RETRIEVAL}": r_at_k,
                f"Precision@{TOP_K_RETRIEVAL}": p_at_k,
                f"F1@{TOP_K_RETRIEVAL}": f1_at_k,
                "Training Time": 0.0,
                "Encoding Time": enc_time,
                "Retrieval Time": res_time
            })
            continue 
       
        # 1. Sample Training Data
        n_samples = int(len(df_train_pool) * fraction)
        if n_samples < 2: n_samples = 2
        df_train_sub = df_train_pool.sample(n=n_samples, random_state=42)

        tf_left_cols = task_info.get("left_cols_tfidf", None)
        
        # Use the PRE-BUILT vectorizer and matrix
        train_examples, train_pairs = create_examples_and_track(
            df_l, df_r, df_train_sub, task_info['left_cols'], task_info['right_cols'], 
            vectorizer, tfidf_matrix, tfidf_left_cols=tf_left_cols, neg_ratio=NEG_RATIO
        )
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
        seen_pairs = train_pairs
        
        # 2. Train
        from copy import deepcopy
        current_model = deepcopy(base_model)
        train_loss = losses.OnlineContrastiveLoss(current_model, distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE, margin=0.5)
        warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)
        
        t_start = time.time()
        current_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=NUM_EPOCHS, warmup_steps=warmup_steps, show_progress_bar=False
        )
        train_time = time.time() - t_start
        
        # 3. Evaluate Pure @K
        r_at_k, p_at_k, f1_at_k, enc_time, res_time = evaluate_with_faiss_hnsw(
            current_model, df_l, df_r, df_test, seen_pairs, 
            task_info['left_cols'], task_info['right_cols'], k=TOP_K_RETRIEVAL
        )
        
        print(f"       [Result] R@{TOP_K_RETRIEVAL}: {r_at_k:.4f} | P@{TOP_K_RETRIEVAL}: {p_at_k:.4f} | F1@{TOP_K_RETRIEVAL}: {f1_at_k:.4f}")
        
        results.append({
            "Dataset": task_info['name'],
            "Model": model_name,
            "Train Fraction": fraction,
            f"Recall@{TOP_K_RETRIEVAL}": r_at_k,
            f"Precision@{TOP_K_RETRIEVAL}": p_at_k,
            f"F1@{TOP_K_RETRIEVAL}": f1_at_k,
            "Training Time": train_time,
            "Encoding Time": enc_time,
            "Retrieval Time": res_time
        })
        
        del current_model; gc.collect(); torch.cuda.empty_cache()
        
    return results

# --- MAIN EXECUTION ---
all_results = []

for task in TASKS:
    print(f"\n>>> PROCESSING TASK: {task['name']}")
    
    try:
        df_l, df_r, df_m = load_data_pandas(DATA_PATH, task['left'], task['right'], task['matches'], encoding=task['encoding'], sep_truth=task['sep_truth'], match_left_col=task['match_left_col'], match_right_col=task['match_right_col'], sep=task['sep'])
    except Exception as e:
        print(f"Error loading {task['name']}: {e}")
        continue

    # Clean datasets
    if task["name"] == "IMDB-DBPEDIA":
        print("len dfm",len(df_m), "left", len(df_l)  , "right", len(df_r))
        df_l.reset_index(inplace=True); df_r.reset_index(inplace=True)
        df_l = df_l.dropna(subset=['title']); df_r = df_r.dropna(subset=['title'])           
        df_l['id'] = pd.to_numeric(df_l['id'], errors='coerce'); df_r['id'] = pd.to_numeric(df_r['id'], errors='coerce')
        df_m["left_id"] = df_m["left_id"].astype(int); df_m["right_id"] = df_m["right_id"].astype(int)
        valid_d1_ids = set(df_l['id'].values); valid_d2_ids = set(df_r['id'].values)
        mask_to_keep = df_m['left_id'].isin(valid_d1_ids) & df_m['right_id'].isin(valid_d2_ids)
        df_m = df_m[mask_to_keep].copy()
        df_l.set_index('id', inplace=True); df_r.set_index('id', inplace=True)
        print("len dfm",len(df_m), "left", len(df_l)  , "right", len(df_r))

    if task["name"] == "Amazon-Walmart":
        df_l.reset_index(inplace=True); df_r.reset_index(inplace=True)
        df_l['id'] = pd.to_numeric(df_l['id'], errors='coerce'); df_l.dropna(subset=['id'], inplace=True); df_l['id'] = df_l['id'].astype(int)
        df_r['id'] = pd.to_numeric(df_r['id'], errors='coerce'); df_r.dropna(subset=['id'], inplace=True); df_r['id'] = df_r['id'].astype(int)
        df_m["left_id"] = pd.to_numeric(df_m["left_id"], errors='coerce'); df_m["right_id"] = pd.to_numeric(df_m["right_id"], errors='coerce')
        df_m.dropna(subset=["left_id", "right_id"], inplace=True)
        df_m["left_id"] = df_m["left_id"].astype(int); df_m["right_id"] = df_m["right_id"].astype(int)
        valid_ids_a = set(df_l['id'].values); valid_ids_b = set(df_r['id'].values)
        df_m = df_m[df_m["right_id"].isin(valid_ids_b) & df_m["left_id"].isin(valid_ids_a)]
        df_l.set_index('id', inplace=True); df_r.set_index('id', inplace=True)

    # --- 1. PRE-BUILD GLOBAL TF-IDF INDEX ---
    print("   Building Global TF-IDF Index for Hard-Negative Mining...")
    tf_right_cols = task.get("right_cols_tfidf", task["right_cols"])
    corpus_ids = list(df_r.index)
    corpus_texts_for_tfidf = [serialize_row(df_r.loc[rid], columns=tf_right_cols) for rid in corpus_ids]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus_texts_for_tfidf)
    # ----------------------------------------

    df_train_pool, df_test = train_test_split(df_m, test_size=0.4, random_state=42)
    print(f"   Matches Split -> Pool: {len(df_train_pool)} |  Test: {len(df_test)}")

    for model_name, model_path in EMBEDDING_MODELS.items():
        print(f"\n--- Model: {model_name} ---")
        
        word_embedding_model = models.Transformer(model_path)
        word_embedding_model.max_seq_length = 128
        word_embedding_model.tokenizer.add_tokens(["[COL]", "[VAL]"], special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        base_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        # Pass the pre-built vectorizer and matrix into the cycle
        all_results.extend(run_training_cycle(f"Raw-{model_name}", base_model, df_l, df_r, df_train_pool, df_test, task, vectorizer, tfidf_matrix))
        
pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
print(f"\n\nSuccess! All results saved to {RESULTS_FILE}")
