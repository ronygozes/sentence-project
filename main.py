import pandas as pd
from load_excel import load_clean_split
# from recreate_df import recreate_df
from sentence_transformers_models import create_chapter_df

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)

def main():
    """
    models: "bi_minilm", "cross_minilm", "e5_large"
    choosing_algorithm: "greedy", "hungarian"
    """
    file1 = "נוף הגליל"
    file2 = "נצרת השלום"
    input1 = rf"/home/rony/projects/sentence-project/excel_files/{file1}.xlsx"
    input2 = rf"/home/rony/projects/sentence-project/excel_files/{file2}.xlsx"

    chapters1 = load_clean_split(input1)
    chapters2 = load_clean_split(input2)

    chapter_name = '08'
    df1 = chapters1[chapter_name]
    df2 = chapters2[chapter_name]
    group_a = df1['תאור'].tolist()
    group_b = df2['תאור'].tolist()
    df = create_chapter_df(group_a, group_b)

    print(df)
    df.to_excel(rf'/home/rony/projects/sentence-project/dfs/{chapter_name}.xlsx')


if __name__ == "__main__":
    main()
