import pandas as pd
import torch
import gc
import time
import os
import random
import numpy as np
import faiss  # <-- NEW: Required for ANNS index
from sentence_transformers import SentenceTransformer, losses, models, InputExample, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Import from utils (or fallback) ---
try:
    from utils import load_data_pandas, serialize_row
except ImportError:
    print("Warning: utils.py not found. Using fallback functions.")
    def serialize_row(row, columns):
        # Fallback serialization if utils is missing
        return " ".join([f"{col} {val}" for col, val in row.items() if pd.notna(val)])

# --- CONFIGURATION ---
UNIVERSAL_MODEL_PATH = "output/universal_model"
DATA_PATH = "./data/"
BATCH_SIZE = 16
RESULTS_FILE = "results1.csv"

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
      "name":"IMDB-DBPEDIA",
      "left": "imdb.csv", "right": "dbpedia.csv", "matches": "truth_imdb_dbpedia.csv",
      "encoding": "utf-8", "sep_truth": "|", "sep" :"|",
      "match_left_col": "D1", "match_right_col": "D2",
      "left_cols": ["title", "starring"], "right_cols": ["title","starring"]
    },
    {
      "name": "DBLP",
      "left": "test_dblp_A.txt", "right": "test_dblp_B.txt", "matches": "truth_DBLP.csv",
      "encoding": "utf-8", "sep_truth": ",", "sep" :",",
      "match_left_col": "id1", "match_right_col": "id2",
      "left_cols": ["author1", "author2","title", "year"], "right_cols": ["author1","author2","title","year"]
    },
    {
     "name": "Scholar-DBLP",
     "left": "Scholar.csv", "right": "DBLP.csv", "matches": "truth_Scholar_DBLP.csv",
     "encoding": "unicode_escape", "sep_truth": ",", "sep" :",",
     "match_left_col": "idScholar", "match_right_col": "idDBLP",
     "left_cols": ["authors", "title","venue","year"], "right_cols": ["authors","title","venue","year"]
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
    }
]

# --- HELPER: Create Pos+Neg InputExamples AND track explicitly generated pairs ---
def create_examples_and_track(df_left, df_right, df_matches, left_cols, right_cols, tfidf_left_cols=None, tfidf_right_cols=None, neg_ratio=1):
    examples = []
    pairs_used = set()
    if tfidf_left_cols is None:
          tfidf_left_cols = left_cols
          tfidf_right_cols = right_cols
    print(f"Left TF-IDF columns={tfidf_left_cols}, right TF-IDF columns={tfidf_right_cols}")
    corpus_ids = list(df_right.index)
    
    # 1. Prepare Corpus specifically for TF-IDF (e.g., ONLY the 'name' column)
    corpus_texts_for_tfidf = [serialize_row(df_right.loc[rid], columns=tfidf_right_cols) for rid in corpus_ids]
    
    # 2. Build TF-IDF Index using the narrow attributes
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus_texts_for_tfidf)
    
    # 3. Cache the FULL Left texts for training, and NARROW Left texts for searching
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
            
        # These are the FULL texts (Name + Description) that the model will actually learn from
        text_l_full = left_cache_full[lid]
        text_r_full = serialize_row(df_right.loc[rid], columns=right_cols)
        
        # 1. Positive Pair (using full text)
        examples.append(InputExample(texts=[text_l_full, text_r_full], label=1))
        pairs_used.add((lid, rid))
        
        # 2. Hard Negative Pairs (Searching using ONLY the narrow text)
        text_l_narrow = left_cache_tfidf[lid]
        query_vec = vectorizer.transform([text_l_narrow])
        cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
        
        top_indices = cosine_similarities.argsort()[-(neg_ratio + 5):][::-1]
        
        negatives_added = 0
        for idx in top_indices:
            hard_neg_rid = corpus_ids[idx]
            
            if hard_neg_rid == rid: 
                continue 
            
            # The model must learn from the FULL text of the negative item, not just its name!
            text_neg_full = serialize_row(df_right.loc[hard_neg_rid], columns=right_cols)
            examples.append(InputExample(texts=[text_l_full, text_neg_full], label=0))
            pairs_used.add((lid, hard_neg_rid))
            
            negatives_added += 1
            if negatives_added >= neg_ratio:
                break
            
    return examples, pairs_used



# --- HELPER: Evaluation using FAISS HNSW ---
def evaluate_with_faiss_hnsw(model, df_l, df_r, test_matches_df, seen_pairs, valid_evaluator, left_cols, right_cols, k=100):
    """
    1. Extract optimal threshold from the validation evaluator.
    2. Encode Corpus (Right dataset) and Queries (Left dataset).
    3. Build FAISS HNSW index and retrieve Top-K candidates.
    4. Filter out any candidate pair that was in `seen_pairs` (training/validation).
    5. Classify the remaining pairs using the optimal threshold.
    """
    # 1. Get Optimal Threshold from Validation set
    #if hasattr(valid_evaluator, "compute_metrices"):
    #    val_metrics = valid_evaluator.compute_metrices(model)
    #else:
    #    val_metrics = valid_evaluator.compute_metrics(model)
     
    val_metrics = valid_evaluator(model)
        
    #optimal_threshold = val_metrics.get('valid-eval_cossim_f1_threshold', 0.5)
    # DEBUG: Print the keys so you can see exactly what your version outputs
    print(f"    [DEBUG] Available metrics: {list(val_metrics.keys())}")
        
    # 2. Dynamically hunt down the correct threshold key
    optimal_threshold = 0.5 # Default fallback
    for key, value in val_metrics.items():
        # Look for any key that contains both "f1_threshold" and "cos" (catches cosine or cossim)
        if 'f1_threshold' in key and 'cos' in key:
            optimal_threshold = float(value)
            print(f"    [DEBUG] Successfully extracted threshold: {optimal_threshold:.4f} (from key: '{key}')")
            break

    

    # 2. Prepare Corpus (Right Dataset)
    corpus_ids = list(df_r.index)
    corpus_texts = [serialize_row(df_r.loc[rid], columns=right_cols) for rid in corpus_ids]
    
    # 3. Prepare Queries (Left Dataset)
    query_ids = list(df_l.index)
    query_texts = [serialize_row(df_l.loc[qid], columns=left_cols) for qid in query_ids]
    
    # 4. Measure Encoding Time & Encode with Normalization (required for FAISS Inner Product)
    t_enc_start = time.time()
    corpus_embeddings = model.encode(corpus_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    query_embeddings = model.encode(query_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    enc_time = time.time() - t_enc_start
    
    # 5. Build FAISS HNSW Index
    t_res_start = time.time()
    d = corpus_embeddings.shape[1]
    # Inner product on normalized vectors is equal to cosine similarity
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT) 
    index.hnsw.efConstruction = 40
    index.add(corpus_embeddings)
    index.hnsw.efSearch = 64
    
    # Ensure K doesn't exceed corpus size
    actual_k = min(k, len(corpus_ids))
    
    # 6. Retrieve Top-K Candidates
    scores, indices = index.search(query_embeddings, actual_k)
    res_time = time.time() - t_res_start
    
    # 7. Evaluate Formulated Pairs
    test_matches_set = set(zip(test_matches_df['left_id'], test_matches_df['right_id']))
    
    TP, FP = 0, 0
    hits_at_k = 0 # Track how many true matches were retrieved in Top-K
    
    for i, qid in enumerate(query_ids):
        # We only care about Recal@K if this query actually has a ground-truth match in the test set
        has_test_match = any(qid == tm[0] for tm in test_matches_set)
        found_in_top_k = False
        
        for j in range(actual_k):
            rid = corpus_ids[indices[i][j]]
            score = scores[i][j]
            pair = (qid, rid)
            
            # --> LEAKAGE FILTER: If pair was explicitly in Train/Val, skip it!
            if pair in seen_pairs:
                continue
            
            # Mark if the true pair was successfully retrieved by FAISS
            if pair in test_matches_set:
                found_in_top_k = True
                
            # --> MATCHING FILTER: Classify using the validation threshold
            if score >= optimal_threshold:
                if pair in test_matches_set:
                    TP += 1
                else:
                    FP += 1 # It passed the threshold but is not a true test match
        
        if has_test_match and found_in_top_k:
            hits_at_k += 1

    # False Negatives: True test matches that were either NOT retrieved by FAISS, OR scored below threshold
    FN = len(test_matches_set) - TP
    
    # Calculate Final Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    unique_test_queries = len(set(test_matches_df['left_id']))
    recall_at_k = hits_at_k / unique_test_queries if unique_test_queries > 0 else 0.0
    
    # Clean up memory
    del index
    gc.collect()
    
    return recall, precision, f1, recall_at_k, optimal_threshold, enc_time, res_time


# --- CORE EXPERIMENT CYCLE ---
def run_training_cycle(model_name, base_model, df_l, df_r, df_train_pool, df_valid, df_test, task_info):
    results = []
    
    print(f"    Preparing Validation set & Evaluator...")
    # Extract validation pairs and explicitly track them
    valid_examples, valid_pairs = create_examples_and_track(df_l, df_r, df_valid, task_info['left_cols'], task_info['right_cols'], 
                                  tfidf_left_cols=task_info["left_cols_tfidf"], tfidf_right_cols=task_info["right_cols_tfidf"], neg_ratio=NEG_RATIO)
    valid_evaluator = BinaryClassificationEvaluator.from_input_examples(valid_examples, batch_size=BATCH_SIZE, name='valid-eval')

    for fraction in TRAIN_FRACTIONS:
        print(f"    >> Training Fraction: {fraction*100}%")
        
        if fraction == 0.0:
            print("       Skipping training (Zero-Shot baseline).")
            # For 0-shot, seen_pairs is only the validation set
            seen_pairs = valid_pairs 
            recall, prec, f1, r_at_k, best_thr, enc_time, res_time = evaluate_with_faiss_hnsw(
                base_model, df_l, df_r, df_test, seen_pairs, valid_evaluator, 
                task_info['left_cols'], task_info['right_cols'], k=TOP_K_RETRIEVAL
            )
            
            print(f"       [Result] F1: {f1:.4f} | R: {recall:.4f} | P: {prec:.4f} | R@{TOP_K_RETRIEVAL}: {r_at_k:.4f} | Thr: {best_thr:.2f}")
            
            results.append({
                "Dataset": task_info['name'],
                "Model": model_name,
                "Train Fraction": 0.0,
                "Recall": recall,
                "Precision": prec,
                "F1": f1,
                f"Recall@{TOP_K_RETRIEVAL}": r_at_k,
                "Best Threshold": best_thr,
                "Training Time": 0.0,
                "Encoding Time": enc_time,
                "Resolution Time": res_time
            })
            continue 
       
        # 1. Sample Training Data
        n_samples = int(len(df_train_pool) * fraction)
        if n_samples < 2: n_samples = 2
        
        df_train_sub = df_train_pool.sample(n=n_samples, random_state=42)
        train_examples, train_pairs = create_examples_and_track(df_l, df_r, df_train_sub, task_info['left_cols'], task_info['right_cols'], 
        tfidf_left_cols=task_info["left_cols_tfidf"], tfidf_right_cols=task_info["right_cols_tfidf"],
        neg_ratio=NEG_RATIO)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
        
        # Merge Train and Val pairs to prevent leakage during FAISS search
        seen_pairs = valid_pairs.union(train_pairs)
        
        # 2. Reset Model
        from copy import deepcopy
        current_model = deepcopy(base_model)
        
        # 3. Setup Loss & Warmup
        train_loss = losses.OnlineContrastiveLoss(current_model, distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE, margin=0.5)
        warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)
        
        # 4. Train
        t_start = time.time()
        current_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=NUM_EPOCHS,
            warmup_steps=warmup_steps,
            show_progress_bar=False
        )
        train_time = time.time() - t_start
        
        # 5. Evaluate using FAISS HNSW
        recall, prec, f1, r_at_k, best_thr, enc_time, res_time = evaluate_with_faiss_hnsw(
            current_model, df_l, df_r, df_test, seen_pairs, valid_evaluator, 
            task_info['left_cols'], task_info['right_cols'], k=TOP_K_RETRIEVAL
        )
        
        print(f"       [Result] F1: {f1:.4f} | R: {recall:.4f} | P: {prec:.4f} | R@{TOP_K_RETRIEVAL}: {r_at_k:.4f} | Thr: {best_thr:.2f}")
        
        results.append({
            "Dataset": task_info['name'],
            "Model": model_name,
            "Train Fraction": fraction,
            "Recall": recall,
            "Precision": prec,
            "F1": f1,
            f"Recall@{TOP_K_RETRIEVAL}": r_at_k,
            "Best Threshold": best_thr,
            "Training Time": train_time,
            "Encoding Time": enc_time,
            "Resolution Time": res_time
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
        df_l.reset_index(inplace=True); df_r.reset_index(inplace=True)
        df_l = df_l.dropna(subset=['title']); df_r = df_r.dropna(subset=['title'])           
        df_l['id'] = pd.to_numeric(df_l['id'], errors='coerce'); df_r['id'] = pd.to_numeric(df_r['id'], errors='coerce')
        df_m["left_id"] = df_m["left_id"].astype(int); df_m["right_id"] = df_m["right_id"].astype(int)
        valid_d1_ids = set(df_l['id'].values); valid_d2_ids = set(df_r['id'].values)
        mask_to_keep = df_m['left_id'].isin(valid_d1_ids) & df_m['right_id'].isin(valid_d2_ids)
        df_m = df_m[mask_to_keep].copy()
        df_l.set_index('id', inplace=True); df_r.set_index('id', inplace=True)

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

    # Split Data: Test (20%), Valid (10%), Train Pool (70%)
    df_train_full, df_test = train_test_split(df_m, test_size=0.2, random_state=42)
    df_train_pool, df_valid = train_test_split(df_train_full, test_size=0.125, random_state=42)
    
    print(f"   Matches Split -> Pool: {len(df_train_pool)} | Valid: {len(df_valid)} | Test: {len(df_test)}")

    for model_name, model_path in EMBEDDING_MODELS.items():
        print(f"\n--- Model: {model_name} ---")
        
        word_embedding_model = models.Transformer(model_path)
        word_embedding_model.max_seq_length = 128
        word_embedding_model.tokenizer.add_tokens(["[COL]", "[VAL]"], special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        base_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        all_results.extend(run_training_cycle(f"Raw-{model_name}", base_model, df_l, df_r, df_train_pool, df_valid, df_test, task))
        
pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
print(f"\n\nSuccess! All results saved to {RESULTS_FILE}")
