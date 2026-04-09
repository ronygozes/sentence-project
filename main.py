import os
import pandas as pd
from configs import *

from load_excel import load_clean_split
from sentence_transformers_models import create_transformer_df
from llm import llm_pipeline

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)


def create_index_matches(chapters1, chapters2):
    os.makedirs(f'{data_dir}/index_matches', exist_ok=True)

    # create matching index df per chapter using sentence-transformers
    transformer_files = [name.split('.')[0] for name in os.listdir(rf'{data_dir}/{matches_dir}')]
    for chapter in chapters1:
        if chapter not in transformer_files:
            df1 = chapters1[chapter]
            df2 = chapters2[chapter]
            group_a = df1['תאור'].tolist()
            group_b = df2['תאור'].tolist()
            df = create_transformer_df(group_a, group_b)
            df.to_excel(rf'{data_dir}/{matches_dir}/{chapter}_matches.xlsx')


def recreate_items_df():
    pass

    # llm_df = pd.read_excel(f"{data_dir}/qwen35_best/{chapter}.xlsx", index_col=0)
    # best_match_index_sr = llm_df['best_match_index']
    # df2_matches = []
    # for i in range(transformer_df.shape[0]):
    #     matches_col = best_match_index_sr.iloc[i]
    #
    #     if not np.isnan(matches_col):
    #         df2_matches.append(transformer_df.iloc[i, int(matches_col)])
    #     else:
    #         df2_matches.append(np.nan)
    # transformer_df['qwen3.5'] = df2_matches
    # transformer_df.to_excel(f"{data_dir}/matches_with_first_llm/{chapter}.xlsx")
    # print('Finished with chapter ', chapter)
    # print(transformer_df, '\n\n\n\n\n')


def main():
    """
    transformer models: "intfloat/multilingual-e5-large", "BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"
    choosing algorithm for transformers: "greedy", "hungarian"
    llm models: "qwen3.5:latest", "deepseek-r1:8b", "deepseek-r1:14b-qwen-distill-q4_K_M"
    """

    input1 = rf"{data_dir}/input_output/{file1}.xlsx"
    input2 = rf"{data_dir}/input_output/{file2}.xlsx"

    chapters1, headers1 = load_clean_split(input1)
    chapters2, headers2 = load_clean_split(input2)

    create_index_matches(chapters1, chapters2)

    llm_pipeline(chapters1, chapters2)


if __name__ == "__main__":
    main()
