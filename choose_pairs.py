import numpy as np

def greedy_max_matching(sim_matrix, threshold=0.55):
    n, m = sim_matrix.shape

    # Flatten into list of (score, i, j)
    pairs = [
        (sim_matrix[i, j], i, j)
        for i in range(n)
        for j in range(m)
    ]

    # Sort by score descending
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_i = set()
    matched_j = set()
    matches = []

    # Greedy selection
    for score, i, j in pairs:
        if score < threshold:
            break
        if i not in matched_i and j not in matched_j:
            matches.append((i, j, score))
            matched_i.add(i)
            matched_j.add(j)

    # Add unmatched rows
    for i in range(n):
        if i not in matched_i:
            matches.append((i, None, None))

    for j in range(m):
        if j not in matched_j:
            matches.append((None, j, None))

    return matches