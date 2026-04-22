import pandas as pd

def assemble_chapter(df_A, df_B, llm_results, chapter_name, col_id="סעיף", col_text="תיאור", col_price="מחיר"):
    """
    Build the final aligned DataFrame for a single chapter.
    df_A: Group A DataFrame
    df_B: Group B DataFrame
    llm_results: list/dict of LLM outputs (same order as df_A)
    chapter_name: string
    """

    # Normalize indices
    df_A = df_A.reset_index(drop=True)
    df_B = df_B.reset_index(drop=True)

    # Track which B items were used
    used_B_indices = set()

    rows = []

    # ---------------------------------------------------------
    # 1. Build rows for each A item (matched or unmatched)
    # ---------------------------------------------------------
    for i, row_A in df_A.iterrows():
        A_id = row_A.get(col_id)
        A_text = row_A.get(col_text)
        A_price = row_A.get(col_price)

        result = llm_results[i]
        idx = result.get("best_match_index")

        # Default B fields
        B_text = None
        B_price = None

        # If the model returned a valid index, try to use it
        if idx is not None:
            try:
                idx = int(idx)
                if 0 <= idx < len(df_B):
                    row_B = df_B.iloc[idx]
                    B_text = row_B.get(col_text)
                    B_price = row_B.get(col_price)
                    used_B_indices.add(idx)
            except:
                pass  # leave B fields empty

        rows.append({
            "A_id": A_id,
            "A_text": A_text,
            "A_price": A_price,
            "B_text": B_text,
            "B_price": B_price,
            "reason": result.get("reason"),
            "confidence": result.get("confidence"),
            "match_found": B_text is not None
        })

    # ---------------------------------------------------------
    # 2. Add unused B items
    # ---------------------------------------------------------
    all_B_indices = set(range(len(df_B)))
    unused_B = all_B_indices - used_B_indices

    for idx in sorted(unused_B):
        row_B = df_B.iloc[idx]
        rows.append({
            "A_id": None,
            "A_text": None,
            "A_price": None,
            "B_text": row_B.get(col_text),
            "B_price": row_B.get(col_price),
            "reason": "Unused candidate",
            "confidence": None,
            "match_found": False
        })

    # ---------------------------------------------------------
    # 3. Build DataFrame
    # ---------------------------------------------------------
    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # 4. Sort by A_id (Group A order), B-only items at bottom
    # ---------------------------------------------------------
    df["sort_key"] = df["A_id"].fillna("ZZZ")
    df = df.sort_values("sort_key", kind="stable").drop(columns=["sort_key"])

    # ---------------------------------------------------------
    # 5. Add chapter header + blank line
    # ---------------------------------------------------------
    header = pd.DataFrame([{"A_text": chapter_name}])
    blank = pd.DataFrame([{"A_text": ""}])

    df = pd.concat([header, blank, df, blank, blank], ignore_index=True)

    return df
