import os
import numpy as np
import pandas as pd
from configs import *

from load_excel import load_clean_split
from sentence_transformers_models import create_transformer_df
from llm import run_llm

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)


def create_index_matches(chapters1, chapters2):
    # create matching index df per chapter using sentence-transformers
    transformer_files = [name.split('.')[0] for name in os.listdir(rf'{data_dir}/index_matches')]
    for chapter in chapters1:
        if chapter not in transformer_files:
            df1 = chapters1[chapter]
            df2 = chapters2[chapter]
            group_a = df1['תאור'].tolist()
            group_b = df2['תאור'].tolist()
            df = create_transformer_df(group_a, group_b)
            df.to_excel(rf'{data_dir}/index_matches/{chapter}.xlsx')


def create_best_llm_matches(chapters1, chapters2, llm_dir, model, use_previous_results=False, previous_llm_dir=None):
    files = [name.split('.')[0] for name in os.listdir(rf'{data_dir}/{llm_dir}')]

    for chapter in chapters1:
        results_dict = {}
        transformer_df = pd.read_excel(rf'{data_dir}/index_matches/{chapter}.xlsx', index_col=0)

        if chapter in files:
            continue

        df1 = chapters1[chapter]
        df2 = chapters2[chapter]
        group_a = df1['תאור'].tolist()
        group_b = df2['תאור'].tolist()
        print(chapter)
        print([item[:10] for item in group_a])
        print([item[:10] for item in group_b])

        for i, sr in transformer_df.iterrows():
            results_dict[group_a[sr.name]] = [group_b[idx] for idx in sr.values.tolist()]

        if use_previous_results:
            previous_results = pd.read_excel(f"{data_dir}/{previous_llm_dir}/{chapter}.xlsx").to_dict(orient='index')

        else:
            previous_results = None

        llm_df = run_llm(items=results_dict, model=model, previous_results=previous_results)
        llm_df.to_excel(f"{data_dir}/{llm_dir}/{chapter}.xlsx")


def main():
    """
    transformer models: "intfloat/multilingual-e5-large", "BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"
    choosing algorithm for transformers: "greedy", "hungarian"
    llm models: "qwen3.5:latest", "deepseek-r1:14b-qwen-distill-q4_K_M"
    """

    input1 = rf"{data_dir}/input_output/{file1}.xlsx"
    input2 = rf"{data_dir}/input_output/{file2}.xlsx"

    chapters1, headers1 = load_clean_split(input1)
    chapters2, headers2 = load_clean_split(input2)

    create_index_matches(chapters1, chapters2)

    create_best_llm_matches(chapters1, chapters2, llm_dir="qwen35_best", model=small_llm)
    exit()

    create_best_llm_matches(chapters1, chapters2, llm_dir="deepseek-r1_best", model=large_llm,
                            use_previous_results=True, previous_llm_dir="qwen35_best")
    qwen35_files = [name.split('.')[0] for name in os.listdir(rf'/{data_dir}/qwen35_best')]
    print(qwen35_files)
    for chapter in chapters1:
        results_dict = {}
        transformer_df = pd.read_excel(rf'{data_dir}/index_matches/{chapter}.xlsx', index_col=0)

        if chapter not in qwen35_files:
            df1 = chapters1[chapter]
            df2 = chapters2[chapter]
            group_a = df1['תאור'].tolist()
            group_b = df2['תאור'].tolist()
            print(chapter)
            print(group_a)
            print(group_b)

            for i, sr in transformer_df.iterrows():
                results_dict[group_a[sr.name]] = [group_b[idx] for idx in sr.values.tolist()]

            llm_df = run_llm(items=results_dict, model=first_llm_model_params)
            llm_df.to_excel(f"{data_dir}/qwen35_best/{chapter}.xlsx")


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




if __name__ == "__main__":
    main()
