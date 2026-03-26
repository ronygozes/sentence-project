import pandas as pd
from load_excel import load_clean_split
from create_scores_matrix import load_model
from recreate_df import recreate_df
    

def main():
    """
    models: "bi_minilm", "cross_minilm", "e5_large"
    choosing_algorithm: "greedy", "hungarian"
    """
    file1 = "נוף הגליל"
    file2 = "נצרת השלום"
    path1 = rf"/mnt/c/Users/Rony/Downloads/{file1}.xlsx"
    path2 = rf"/mnt/c/Users/Rony/Downloads/{file2}.xlsx"
    output_path = rf"/mnt/c/Users/Rony/Downloads/output.xlsx"

    models = {
                "e5_large": {"model": "intfloat/multilingual-e5-large", "type": "bi_encoder", "prefix": "query: "},
                "bge_encoder": {"model": "BAAI/bge-m3", "type": "cross-encoder", "prefix": ""},
                "bge_reranker": {"model": "BAAI/bge-reranker-v2-m3" }
              }
    
    choosing_algorithm = {"greedy", "hungarian"}


    model = load_model()

    chapters1 = load_clean_split(path1)
    chapters2 = load_clean_split(path2)

    chapters_headers = sorted(set(chapters1.keys()) | set(chapters2.keys()))

    final_df = recreate_df(chapters_headers, chapters1, chapters2)

    
    final_df.to_excel(output_path, index=False)
    print(final_df[[f"תיאור_{file1}", f"תיאור_{file2}", "score"]].head(30))
    print(final_df.shape)


if __name__ == "__main__":
    main()
