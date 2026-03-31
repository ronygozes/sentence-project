
import torch

import gc

import numpy as np

import pandas as pd

from sentence_transformers import SentenceTransformer, CrossEncoder, util

from scipy.optimize import linear_sum_assignment

# ============================================================

# GPU MEMORY CLEANUP

# ============================================================


def clear_vram():

    """Frees VRAM for the next model."""

    gc.collect()

    torch.cuda.empty_cache()

 
# ============================================================

# NORMALIZATION FUNCTIONS

# ============================================================


def min_max_normalize(matrix):

    """

    Min-Max normalization to the range [0,1].

    Safe even if many values cluster near each other.

    """

    m = matrix.astype(float)

    min_val = np.min(m)
    max_val = np.max(m)

    if max_val == min_val:

        # Degenerate matrix: all values equal → return zeros

        return np.zeros_like(m)

    return (m - min_val) / (max_val - min_val)


def apply_threshold(matrix_norm, threshold=0.4, negative_fill=-1e9):

    """

    Applies threshold BEFORE matching.

    Values below threshold become a huge negative value,

    which ensures greedy & Hungarian avoid them.

    """

    m = matrix_norm.copy()

    m[m < threshold] = negative_fill

    return m

 
# ============================================================

# ENCODER FUNCTIONS

# ============================================================


def run_bi_encoder(model_name, group_a, group_b, prefix_a="", prefix_b=""):

    """

    Runs a Bi-Encoder (E5, BGE-M3, etc.) and returns raw cosine similarity matrix.

    """

    print(f"\nLoading Bi-Encoder: {model_name}")

    model = SentenceTransformer(model_name, device="cuda")

    a_proc = [f"{prefix_a}{s}" for s in group_a]
    b_proc = [f"{prefix_b}{s}" for s in group_b]

    emb_a = model.encode(a_proc, convert_to_tensor=True, show_progress_bar=True)
    emb_b = model.encode(b_proc, convert_to_tensor=True, show_progress_bar=True)

    cosine_matrix = util.cos_sim(emb_a, emb_b).cpu().numpy()

    del model, emb_a, emb_b
    clear_vram()

    return cosine_matrix


def run_cross_encoder(model_name, group_a, group_b):

    """

    Runs a Cross-Encoder (BGE-Reranker etc.) and returns raw score matrix.

    """

    print(f"\nLoading Cross-Encoder: {model_name}")

    model = CrossEncoder(model_name, device="cuda")

    pairs = [[s1, s2] for s1 in group_a for s2 in group_b]

    scores = model.predict(pairs, batch_size=32, show_progress_bar=True)

    scores = np.array(scores)
 
    # If output is 2D (logits), keep final column

    if scores.ndim == 2:

        scores = scores[:, -1]

    matrix = scores.reshape(len(group_a), len(group_b))

    del model
    clear_vram()

    return matrix
 

# ============================================================

# MATCHING ALGORITHMS

# ============================================================

 

def solve_greedy(matrix):

    """

    Greedy max-matching with unmatched handling.

    Avoids negative-fill (-1e9) pairs naturally.

    """

    rows, cols = matrix.shape
    flat = []

    for r in range(rows):
        for c in range(cols):
            flat.append((matrix[r, c], r, c))

    flat.sort(key=lambda x: x[0], reverse=True)

    used_r, used_c = set(), set()
    results = []

    for score, r, c in flat:
        # if score < -1e8:  # ignore negative-fill
        #     continue
        if r not in used_r and c not in used_c:
            results.append({"row": r, "col": c, "score": float(score)})
            used_r.add(r)
            used_c.add(c)

    # Unmatched rows
    for r in range(rows):
        if r not in used_r:
            results.append({"row": r, "col": None, "score": None})

    # Unmatched cols
    # for c in range(cols):
    #     if c not in used_c:
    #         results.append({"row": None, "col": c, "score": None})

    return results
 

def solve_hungarian(matrix):

    """

    Hungarian max-matching with unmatched handling.

    Negative-fill avoids low-quality matches.

    """

    row_ind, col_ind = linear_sum_assignment(-matrix)
    used_r = set(row_ind)
    used_c = set(col_ind)
    results = []

    # Matched pairs
    for r, c in zip(row_ind, col_ind):
        score = matrix[r, c]
        # if score < -1e8:
        #     score = None
        #     c = None
        results.append({"row": int(r), "col": int(c), "score": float(score)})

    # Unmatched rows
    for r in range(matrix.shape[0]):
        if r not in used_r:
            results.append({"row": r, "col": None, "score": None})

    # Unmatched cols
    # for c in range(matrix.shape[1]):
    #     if c not in used_c:
    #         results.append({"row": None, "col": c, "score": None})

    return results


# ============================================================

# FULL PIPELINE (FOR ANY MODEL)

# ============================================================


def create_model_df(matrix_raw, mat_name):
    """
    Takes a raw similarity matrix and returns:
      - normalized matrix
      - greedy matches
      - hungarian matches
    """
    # Normalize
    matrix_norm = min_max_normalize(matrix_raw)

    greedy_res = solve_greedy(matrix_norm)
    greedy_dict = {item['row']: item['col'] for item in greedy_res}
    greedy_sr = pd.Series(greedy_dict, name=f'{mat_name}_greedy')

    hungarian_res = solve_hungarian(matrix_norm)
    hungarian_dict = {item['row']: item['col'] for item in hungarian_res}
    hungarian_sr = pd.Series(hungarian_dict, name=f'{mat_name}_hungarian')

    df = pd.concat([greedy_sr, hungarian_sr], axis=1)
    return df


# ============================================================

# EXAMPLE USAGE

# ============================================================

 
def create_chapter_df(group_a, group_b):

    # ---- 1) BI-ENCODER EXAMPLE (E5-LARGE) ----

    raw_e5 = run_bi_encoder("intfloat/multilingual-e5-large", group_a, group_b, prefix_a="passage: ", prefix_b="passage: ")

    results_e5 = create_model_df(raw_e5, 'e5-large')

    # ---- 2) BI-ENCODER EXAMPLE (BGE-M3) ----

    raw_bge = run_bi_encoder("BAAI/bge-m3", group_a, group_b)

    results_bge = create_model_df(raw_bge, "BAAI/bge-m3")

    # ---- 3) CROSS-ENCODER EXAMPLE (BGE-RERANKER-V2-M3) ----

    raw_rerank = run_cross_encoder("BAAI/bge-reranker-v2-m3", group_a, group_b)

    results_rerank = create_model_df(raw_rerank, "BAAI/bge-reranker-v2-m3")

    df = pd.concat([results_e5, results_bge, results_rerank], axis=1).astype('Int64')

    return df


if __name__ == "__main__":
    values_b = [
        "The company announced a new AI feature during yesterday's meeting.",
        "My neighbor adopted a small dog that barks all night.",
        "I forgot my umbrella and got completely soaked in the rain.",
        "The restaurant near my office serves amazing vegan dishes.",
        "She spent the whole weekend organizing her photo albums.",
        "The train was delayed again because of technical issues.",
        "He bought a new laptop to improve his workflow.",
        "The museum opened a new exhibition about ancient civilizations.",
        "I tried a new recipe yesterday and it turned out delicious.",
        "The local basketball team won their game by a huge margin."
    ]

    values_a = [
        "I don't know what to write",
        "Jerusalem is stupid",
        "Pipe size 3.5*20 inch",
        "A fresh exhibit about ancient cultures just opened at the city museum.",
        "I ended up drenched because I left my umbrella at home.",
        "The tech company revealed an AI upgrade in their latest briefing.",
        "She reorganized thousands of old photos over the weekend.",
        "The regional basketball squad dominated their last match.",
        "He purchased a faster laptop to boost his productivity.",
        "A new vegan restaurant opened near my workplace and the food is fantastic.",
        "The commuter train experienced another delay due to a malfunction.",
        "My friend adopted a tiny dog that keeps barking through the night.",
        "I experimented with a new dish yesterday and it came out great.",
        "Just a test sentence to unbalance the groups"
    ]
    create_chapter_df(values_a, values_b, '34')