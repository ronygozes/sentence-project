from create_scores_matrix import compute_similarity_matrix
from choose_pairs import greedy_max_matching


def recreate_df(chapters_headers, chapters1, chapters2):
    final_rows = []
    for ch in chapters_headers:
        df1 = chapters1.get(ch, None)
        df2 = chapters2.get(ch, None)

        # Case 1: chapter exists in both → match
        if df1 is not None and df2 is not None:
            desc1 = df1["תאור"].tolist()
            desc2 = df2["תאור"].tolist()

            sim_matrix = compute_similarity_matrix(model, desc1, desc2)
            matches = greedy_max_matching(sim_matrix, threshold=0.55)

            for i, j, score in matches:
                row1 = df1.iloc[i] if i is not None else None
                row2 = df2.iloc[j] if j is not None else None

                final_rows.append({
                    "chapter": ch,
                    f"סעיף_{file1}": row1["סעיף"] if row1 is not None else None,
                    f"תיאור_{file1}": row1["תאור"] if row1 is not None else None,
                    f"מחיר_{file1}": row1["מחיר"] if row1 is not None else None,
                    f"סעיף_{file2}": row2["סעיף"] if row2 is not None else None,
                    f"תיאור_{file2}": row2["תאור"] if row2 is not None else None,
                    f"מחיר_{file2}": row2["מחיר"] if row2 is not None else None,
                    "score": score,
                })

        # Case 2: chapter only in df1 → append all df1 rows with NaN for df2
        elif df1 is not None:
            for _, row1 in df1.iterrows():
                final_rows.append({
                    "chapter": ch,
                    f"סעיף_{file1}": row1["סעיף"],
                    f"תיאור_{file1}": row1["תאור"],
                    f"מחיר_{file1}": row1["מחיר"],
                    f"סעיף_{file2}": None,
                    f"תיאור_{file2}": None,
                    f"מחיר_{file2}": None,
                    "score": None,
                })

        # Case 3: chapter only in df2 → append all df2 rows with NaN for df1
        elif df2 is not None:
            for _, row2 in df2.iterrows():
                final_rows.append({
                    "chapter": ch,
                    f"סעיף_{file1}": None,
                    f"תיאור_{file1}": None,
                    f"מחיר_{file1}": None,
                    f"סעיף_{file2}": row2["סעיף"],
                    f"תיאור_{file2}": row2["תאור"],
                    f"מחיר_{file2}": row2["מחיר"],
                    "score": None,
                })

    final = pd.DataFrame(final_rows)
