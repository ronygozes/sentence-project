import os

import ollama
import json
import time
import re

import pandas as pd

from configs import data_dir
from model_rules import models_data
from llm_tests import test_cases


def normalize_for_llm(text: str) -> str:
    # Replace inch symbol with safe text
    text = re.sub(r'(\d)\s*"', r'\1 inch', text)
    text = text.replace('מ"מ', 'ממ')  # Hebrew mm
    text = text.replace('"', '')      # fallback: remove any remaining quotes
    return text


def check_json(original_content):
    # Remove <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", original_content, flags=re.DOTALL).strip()

    # Remove code fences
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Normalize common LLM mistakes
    cleaned = cleaned.replace("True", "true").replace("False", "false")
    cleaned = cleaned.replace("None", "null")

    # Replace smart quotes
    cleaned = cleaned.replace("“", "\"").replace("”", "\"")

    # Extract the last JSON object (most reliable)
    matches = re.findall(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if not matches:
        print("No JSON found. Returning fallback.")
        print("original_content:\n", original_content)
        return None
    return matches


def parse_llm_response(response) -> dict:
    original_content = response["message"]["content"].strip()
    matches = check_json(original_content)
    if not matches:
        print("No JSON found. Returning fallback.")
        print("original_content:\n", original_content)
        return {"best_match_index": None, "confidence": "low", "reason": "parse error - no json"}

    json_str = matches[-1]  # last one is usually the correct one

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("JSON decode error. Returning fallback.")
        print("original_content:\n", original_content)
        return {"best_match_index": None, "confidence": "low", "reason": "parse error"}

    # Validate schema
    if "best_match_index" not in data:
        data["best_match_index"] = None
    if data.get("confidence") not in ["high", "medium", "low"]:
        data["confidence"] = "low"
    if "unit_conversion_applied" not in data:
        data["unit_conversion_applied"] = False

    return data


def parse_verification_response(response):
    original_content = response["message"]["content"].strip()
    matches = check_json(original_content)
    if not matches:
        print("JSON decode error. Returning fallback.")
        print("original_content:\n", original_content)
        return {"is_comparable": False, "reason": "parse error"}
    try:
        return json.loads(matches[-1])
    except:
        print("JSON decode error. Returning fallback.")
        print("original_content:\n", original_content)
        return {"is_comparable": False, "reason": "parse error"}


def build_candidate_lists(transformer_df, group_a, group_b):
    results = {}
    for i, sr in transformer_df.iterrows():
        reference = group_a[sr.name]
        candidates = [group_b[idx] for idx in sr.values.tolist()]
        results[reference] = candidates
    return results


def normalize_items(reference, candidates):
    ref_norm = normalize_for_llm(reference)
    cand_norm = [normalize_for_llm(c) for c in candidates]
    return ref_norm, cand_norm


def create_prompts(reference: str, candidates: list[str] | str, step: str):
    if step == "selection_step_prompts":
        candidates_str = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
        user_prompt = (
            models_data[step]["USER_PROMPT"]
            .replace("reference_str", reference)
            .replace("candidates_str", candidates_str)
            .strip()
        )
    else:
        user_prompt = (
            models_data[step]["USER_PROMPT"]
            .replace("reference_str", reference)
            .replace("chosen_candidate_str", candidates)
            .strip()
        )

    system_prompt = (
        models_data[step]["SYSTEM_PROMPT"]
    )

    return user_prompt, system_prompt


def find_best_material_match(user_prompt: str, system_prompt: str, model: str, step: str) -> dict:
    # Load model-specific settings
    model_params = models_data['models'][model]
    think = model_params["think"]
    num_predict = 1000 if not think else 5000
    print('num_predict', num_predict)
    print('think', think)

    # Call the model
    response = ollama.chat(
        model=model,
        options={"temperature": 0, "num_predict": num_predict},
        think=think,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Parse the result
    if step == "selection_step_prompts":
        result = parse_llm_response(response)
    else:
        result = parse_verification_response(response)
    return result


def create_best_llm_matches(chapters1, chapters2, llm_dir, model):
    files = [name.split('.')[0] for name in os.listdir(rf'{data_dir}/{llm_dir}')]
    for chapter in chapters1:
        if chapter in files:
            continue

        transformer_df = pd.read_excel(rf'{data_dir}/index_matches/{chapter}.xlsx', index_col=0)

        df1 = chapters1[chapter]
        df2 = chapters2[chapter]
        group_a = df1['תאור'].tolist()
        group_b = df2['תאור'].tolist()

        items = build_candidate_lists(transformer_df, group_a, group_b)

        results = {}
        for reference in items:
            print(f'Starting with reference: {reference}')
            start = time.time()

            normalizes_reference, normalized_candidates = normalize_items(reference, items[reference])

            user_prompt, system_prompt = create_prompts(normalizes_reference, normalized_candidates, step='selection_step_prompts')

            result = find_best_material_match(user_prompt, system_prompt, model=model, step='selection_step_prompts')

            # make sure best_match_index is the right index for best_match
            if result.get("best_match_index") is not None:
                try:
                    for i, c in enumerate(normalized_candidates):
                        if c.strip() == result["best_match"].strip():
                            result["best_match_index"] = i
                            break
                except Exception as e:
                    print(f'result for reference: {reference} \nis broken:\n{result}')

            results[reference] = result
            print(f'{reference} \ntook {time.time() - start} seconds\n')
            print(result)
            print('\n\n')
        df = pd.DataFrame.from_dict(results, orient='index').reset_index(drop=True)
        os.makedirs(f'{data_dir}/{llm_dir}', exist_ok=True)
        df.to_excel(f"{data_dir}/{llm_dir}/{chapter}.xlsx")


def create_verified_llm_matches(chapters1, chapters2, first_pass_dir, second_pass_dir, model):
    files = [name.split('.')[0] for name in os.listdir(rf'{data_dir}/{second_pass_dir}')]

    for chapter in chapters1:
        chapter = '07'
        if chapter in files:
            continue

        print(f"\n=== Verifying chapter {chapter} ===")

        transformer_df = pd.read_excel(rf'{data_dir}/index_matches/{chapter}.xlsx', index_col=0)
        first_pass_df = pd.read_excel(rf'{data_dir}/{first_pass_dir}/{chapter}.xlsx', index_col=0)

        df1 = chapters1[chapter]
        df2 = chapters2[chapter]
        group_a = df1['תאור'].tolist()
        group_b = df2['תאור'].tolist()

        items = build_candidate_lists(transformer_df, group_a, group_b)

        verified_results = {}

        for i, (reference, candidates) in enumerate(items.items()):
            print(f'Starting with reference: {reference}')
            start = time.time()

            chosen_index = int(first_pass_df.iloc[i]["best_match_index"])

            ref_norm, cand_norm = normalize_items(reference, candidates)

            # STEP 1 — Verification
            # verify_prompt = create_verifier_prompt(ref_norm, cand_norm[chosen_index], chosen_index)
            # verify_prompt = models_data['verification_step_prompts']['USER_PROMPT']
            verify_user_prompt, verify_system_prompt = create_prompts(ref_norm, cand_norm[chosen_index], step='verification_step_prompts')
            verify_result = find_best_material_match(verify_user_prompt, verify_system_prompt, model=model, step='verification_step_prompts')

            if verify_result.get("is_comparable"):
                # Verified — keep original
                verified_results[reference] = {
                    "verified_index": chosen_index,
                    "reason": verify_result.get("reason", "verified"),
                    "confidence": "high"
                }
                print(f'{reference} \ntook {time.time() - start} seconds\n')
                print(verify_result)
                print('\n\n')
                continue

            # STEP 2 — Re-selection
            # select_prompt = create_second_pass_selection_prompt(ref_norm, cand_norm)
            select_user_prompt, select_system_prompt = create_prompts(ref_norm, cand_norm, step='selection_step_prompts')
            select_result = find_best_material_match(select_user_prompt, select_system_prompt, model=model, step='selection_step_prompts')

            new_index = select_result.get("best_match_index", chosen_index)

            verified_results[reference] = {
                "verified_index": new_index,
                "reason": select_result.get("reason", "override"),
                "confidence": "high"
            }
            print(f'{reference} \ntook {time.time() - start} seconds\n')
            print(select_result)
            print('\n\n')

        df = pd.DataFrame.from_dict(verified_results, orient='index').reset_index(drop=True)
        os.makedirs(f"{data_dir}/{second_pass_dir}", exist_ok=True)
        df.to_excel(f"{data_dir}/{second_pass_dir}/{chapter}.xlsx")
        exit()

