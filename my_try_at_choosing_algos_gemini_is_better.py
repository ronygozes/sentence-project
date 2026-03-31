# import torch
# import gc
# from sentence_transformers import SentenceTransformer, CrossEncoder, util

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def greedy_algorithm(input_mat: np.ndarray) -> tuple:
    mat = input_mat.copy()

    best_matches = []

    while mat.max() > 0:
        idx = np.unravel_index(mat.argmax(), mat.shape)
        best_matches.append(idx)
        mat[idx[0], :] = 0  # Zero out the row
        mat[:, idx[1]] = 0  # Zero out the col

    best_rows = np.array([i for (i, j) in best_matches])
    best_cols = np.array([j for (i, j) in best_matches])
    return tuple([best_rows, best_cols])


def hungarian_algorithm(mat: np.ndarray) -> tuple:
    return linear_sum_assignment(1-mat)


def make_matches_series(matches, group_a, group_b):
    """
    assuming group_a is the larger list of texts
    """
    matches_dict = {}
    for i in range(len(matches[0])):
        matches_dict[group_a[matches[0][i]]] = group_b[matches[1][i]]

    for i in range(len(group_a)):
        if i not in matches[0]:
            matches_dict[group_a[i]] = None

    return pd.Series(matches_dict)


def create_matches_df(mat, mat_name, group_a, group_b):
    mat = (mat - mat.min()) / (mat.max() - mat.min())
    print(mat)
    greedy_matches = greedy_algorithm(mat)
    print(greedy_matches)
    hungarian_matches = hungarian_algorithm(mat)
    print(hungarian_matches)

    greedy_series = make_matches_series(greedy_matches, group_a, group_b)
    greedy_series.name = f'greedy_{mat_name}'
    hungarian_series = make_matches_series(hungarian_matches, group_a, group_b)
    hungarian_series.name = f'hungarian_{mat_name}'

    df = pd.concat([hungarian_series, greedy_series], axis=1)
    print(df)


if __name__ == "__main__":
    arr = np.random.randint(1, 100, size=(5, 4))
    group_a = ['a0', 'a1', 'a2', 'a3', 'a4']
    group_b = ['b0', 'b1', 'b2', 'b3']
    create_matches_df(arr, 'e5', group_a, group_b)


