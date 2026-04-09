import os

import ollama
import json
import time
import re

import pandas as pd

from configs import *
from model_rules import models_data
from llm_tests import test_cases

verification_models = [
    {'model': 'qwen3.5:latest', 'think': False, 'name': 'qwen3.5-no-think'},
    {'model': 'qwen3.5:latest', 'think': True, 'name': 'qwen3.5-think'},
    {'model': 'deepseek-r1:14b-qwen-distill-q4_K_M', 'think': False, 'name': 'deepseek-r1-q4-no-think'},
    {'model': 'deepseek-r1:14b-qwen-distill-q4_K_M', 'think': True, 'name': 'deepseek-r1-q4-think'},
    {'model': 'deepseek-r1:14b-qwen-distill-q8_0', 'think': False, 'name': 'deepseek-r1-q8-no-think'},
    {'model': 'deepseek-r1:14b-qwen-distill-q8_0', 'think': True, 'name': 'deepseek-r1-q8-think'},
]


verification_models1 = [
    {'model': 'qwen2.5:7b-instruct-q6_K', 'think': False, 'name': 'qwen2.5-instruct'},
    {'model': 'qwen2.5:7b', 'think': False, 'name': 'qwen2.5-7b'},
    {'model': 'qwen3.5:latest', 'think': False, 'name': 'qwen3.5-no-think'},
    {'model': 'qwen3.5:latest', 'think': True, 'name': 'qwen3.5-think'},
    {'model': 'deepseek-r1:14b-qwen-distill-q4_K_M', 'think': False, 'name': 'deepseek-r1-no-think'},
    {'model': 'deepseek-r1:14b-qwen-distill-q4_K_M', 'think': True, 'name': 'deepseek-r1-think'}
]


def normalize_for_llm(text: str) -> str:
    # Replace inch symbol with safe text
    text = re.sub(r'(\d)\s*"', r'\1 inch', text)
    text = text.replace('מ"מ', 'ממ')  # Hebrew mm
    text = text.replace('"', '')      # fallback: remove any remaining quotes
    return text


def return_data(original_content):
    # Remove <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", original_content, flags=re.DOTALL).strip()

    # Remove code fences
    cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

    # Normalize common LLM mistakes
    cleaned = cleaned.replace("True", "true").replace("False", "false")
    cleaned = cleaned.replace("None", "null")
    cleaned = cleaned.replace("“", "\"").replace("”", "\"")

    # Try to extract the last valid JSON object by scanning for balanced braces
    stack = []
    start = None
    candidates = []

    for i, ch in enumerate(cleaned):
        if ch == '{':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(cleaned[start:i+1])

    # Try parsing candidates from last to first
    for json_str in reversed(candidates):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue

    print("No valid JSON found. Returning None.")
    print("original_content:\n", original_content)
    return None


def return_data_previous(original_content):
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

    try:
        json_str = matches[-1]
        data = json.loads(json_str)
    except IndexError as e:
        print("No JSON found. Returning fallback.")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        print("original_content:\n", original_content)
        data = None
    except json.JSONDecodeError as e:
        print("JSON decode error. Returning fallback.")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        print("original_content:\n", original_content)
        data = None

    return data


def parse_llm_response(data) -> dict:
    # If parsing failed entirely
    if data is None or not isinstance(data, dict):
        return {
            "reason": "No JSON found",
            "confidence": "low",
            "best_match_index": None,
            "best_match": None,
            "reference": None,
            "unit_conversion_applied": False
        }

    # --- Extract and normalize fields (ignore all others) ---

    # reason
    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = str(reason) if reason is not None else ""

    # confidence
    confidence = data.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    # best_match_index
    bmi = data.get("best_match_index")
    try:
        bmi = int(bmi) if bmi is not None else None
    except (ValueError, TypeError):
        bmi = None

    # best_match
    best_match = data.get("best_match")
    if not isinstance(best_match, str):
        best_match = None

    # reference
    reference = data.get("reference")
    if not isinstance(reference, str):
        reference = None

    # unit_conversion_applied
    unit = bool(data.get("unit_conversion_applied", False))

    # Return EXACT schema
    return {
        "reason": reason,
        "confidence": confidence,
        "best_match_index": bmi,
        "best_match": best_match,
        "reference": reference,
        "unit_conversion_applied": unit
    }


def parse_verification_response(data):
    if data is None:
        return {"is_comparable": False, "reason": "No JSON found"}

    # Normalize is_comparable
    val = data.get("is_comparable", False)

    if isinstance(val, str):
        val = val.strip().lower()
        is_comp = val in ("true", "yes", "1", "כן")
    elif isinstance(val, (int, float)):
        is_comp = bool(val)
    else:
        is_comp = bool(val)

    # Normalize reason
    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = str(reason) if reason is not None else ""

    return {
        "is_comparable": is_comp,
        "reason": reason
    }



def build_candidate_dict(matches_path, chapters1, chapters2, chapter):
    matches_df = pd.read_excel(rf'{matches_path}/{chapter}.xlsx', index_col=0)
    df1 = chapters1[chapter]
    df2 = chapters2[chapter]
    group_a = df1['תאור'].tolist()
    group_b = df2['תאור'].tolist()

    items = {}
    for i, sr in matches_df.iterrows():
        reference = group_a[sr.name]
        candidates = [group_b[idx] for idx in sr.values.tolist()]
        items[reference] = candidates
    return items


def normalize_items(reference, candidates):
    ref_norm = normalize_for_llm(reference)
    cand_norm = [normalize_for_llm(c) for c in candidates]
    return ref_norm, cand_norm


def create_prompts(reference: str, candidates: list[str] | str, step: str):
    if step == "selection":
        candidates_str = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
        user_prompt = (
            models_data[step]["USER_PROMPT"]
            .replace("reference_str", reference)
            .replace("candidates_str", candidates_str)
            .strip()
        )
    elif step == 'verification':
        user_prompt = (
            models_data[step]["USER_PROMPT"]
            .replace("reference_str", reference)
            .replace("chosen_candidate_str", candidates)
            .strip()
        )
    else:
        exit('Wrong step')

    system_prompt = (
        models_data[step]["SYSTEM_PROMPT"]
    )

    return user_prompt, system_prompt


def run_llm_model(user_prompt: str, system_prompt: str, model: str, step: str, think: bool) -> dict:
    # Load model-specific settings
    num_predict = 1024  # if not think else 4096
    print('num_predict', num_predict)
    print('think', think)

    # Call the model
    response = ollama.chat(model=model, options={"temperature": 0, "num_predict": num_predict}, think=think,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Parse the result
    original_content = response["message"]["content"].strip()
    data =  return_data(original_content)

    if step == "selection":
        result = parse_llm_response(data)
    else:
        result = parse_verification_response(data)
    return result


# def first_pass(chapters1, chapters2, first_pass_path, matches_path, model):
#     files = [name.split('.')[0] for name in os.listdir(first_pass_path)]
#
#     # loop over a dict of dfs
#     for chapter in chapters1:
#         if chapter in files:
#             continue
#
#         items = build_candidate_dict(matches_path, chapters1, chapters2, chapter)
#
#         # loop over every item in group A
#         results = {}
#         for i, reference in enumerate(items):
#             print(f'{i}. reference: {reference}')
#             start = time.time()
#
#             normalizes_reference, normalized_candidates = normalize_items(reference, items[reference])
#             user_prompt, system_prompt = create_prompts(normalizes_reference, normalized_candidates, step='selection')
#             result = run_llm_model(user_prompt, system_prompt, model=model, step='selection')
#
#             # make sure best_match_index is the right index for best_match
#             try:
#                 if result.get("best_match") is None:
#                     result["best_match_index"] = None
#                 else:
#                     found_best_match = False
#                     for j, c in enumerate(normalized_candidates):
#                         if c.strip() == result.get("best_match").strip():
#                             found_best_match = True
#                             result["best_match_index"] = j
#                             break
#                     if not found_best_match:
#                         raise Exception
#             except:
#                 print(f'Result parse error in first pass:\n{result}')
#                 result['best_match_index'] = None
#                 result['best_match'] = None
#
#
#             results[reference] = result
#             compute_time = time.time()-start
#             print(f'It took {compute_time} seconds\n')
#             print(result)
#             print('\n\n')
#
#         df = pd.DataFrame.from_dict(results, orient='index').reset_index(drop=True)
#         df.to_excel(f"{first_pass_path}/{chapter}_first_pass.xlsx")


def create_verified_llm_matches(chapters1, chapters2, first_pass_path, matches_path, model, verification_pass_path,
                                think):
    files = [name.split('.')[0] for name in os.listdir(verification_pass_path)]

    for chapter in chapters1:
        if chapter in files:
            continue

        print(f"\n=== Verifying chapter {chapter} ===")
        first_pass_df = pd.read_excel(rf'{first_pass_path}/{chapter}_first_pass.xlsx', index_col=0)
        items = build_candidate_dict(matches_path, chapters1, chapters2, chapter)

        verified_results = {}
        for i, (reference, candidates) in enumerate(items.items()):
            print(f'{i} . Starting with reference: {reference}')
            start = time.time()

            raw_index = first_pass_df.iloc[i]["best_match_index"]
            if pd.isna(raw_index):
                verified_results[reference] = {
                    "is_comparable": False,
                    "reason": "No match found for reference"
                }
                continue
            else:
                chosen_index = int(raw_index)

            ref_norm, cand_norm = normalize_items(reference, candidates)

            # STEP 1 — Verification
            verify_user_prompt, verify_system_prompt = create_prompts(ref_norm, cand_norm[chosen_index],
                                                                      step='verification')
            verify_result = run_llm_model(verify_user_prompt, verify_system_prompt, model=model,
                                          step='verification', think=think)

            verified_results[reference] = {
                "is_comparable": verify_result.get("is_comparable", False),
                "reason": verify_result.get("reason", "Unknown")
            }
            print(f'It took {time.time() - start} seconds\n')
            print(verify_result)
            print('\n\n')

        df = pd.DataFrame.from_dict(verified_results, orient='index').reset_index(drop=True)
        df.to_excel(f"{verification_pass_path}/{chapter}_verification_pass.xlsx")


def selection_runs(chapters1, chapters2, first_pass_path, matches_path, model, verification_pass_path=None,
                   second_pass_path=None, second_run=False, think=False):
    if second_run:
        save_path = second_pass_path
        run_num = 'second_pass'
    else:
        save_path = first_pass_path
        run_num = 'first_pass'

    first_pass_df = pd.DataFrame()
    verified_df = pd.DataFrame()

    files = [name.split('.')[0] for name in os.listdir(save_path)]

    # loop over a dict of dfs
    for chapter in chapters1:
        if f'{chapter}_{run_num}' in files:
            continue

        if second_run:
            first_pass_df = pd.read_excel(rf'{first_pass_path}/{chapter}_first_pass.xlsx', index_col=0)
            verified_df = pd.read_excel(rf'{verification_pass_path}/{chapter}_verification.xlsx', index_col=0)
        items = build_candidate_dict(matches_path, chapters1, chapters2, chapter)

        # loop over every item in group A
        results = {}
        for i, reference in enumerate(items):
            print(f'{i}. reference: {reference}')
            start = time.time()

            if second_run:
                verified_item = verified_df.iloc[i]
                is_comparable = verified_item.get("is_comparable")
                if is_comparable:
                    result = first_pass_df.iloc[i].to_dict()
                    results[reference] = result
                    continue

            normalizes_reference, normalized_candidates = normalize_items(reference, items[reference])
            user_prompt, system_prompt = create_prompts(normalizes_reference, normalized_candidates, step='selection')
            result = run_llm_model(user_prompt, system_prompt, model=model, step='selection', think=think)

            # make sure best_match_index is the right index for best_match
            try:
                if result.get("best_match") is None:
                    result["best_match_index"] = None
                else:
                    found_best_match = False
                    for j, c in enumerate(normalized_candidates):
                        if c.strip() == result.get("best_match").strip():
                            found_best_match = True
                            result["best_match_index"] = j
                            break
                    if not found_best_match:
                        raise Exception
            except:
                print(f'Result parse error in {run_num}:\n{result}')
                result['best_match_index'] = None
                result['best_match'] = None

            results[reference] = result
            compute_time = time.time() - start
            print(f'It took {compute_time} seconds\n')
            print(result)
            print('\n\n')

        df = pd.DataFrame.from_dict(results, orient='index').reset_index(drop=True)
        df.to_excel(f"{save_path}/{chapter}_{run_num}.xlsx")


def llm_pipeline(chapters1, chapters2):

    debug_chapters = ['22']  # or None

    if debug_chapters:
        chapters1 = {k: chapters1[k] for k in debug_chapters}
        chapters2 = {k: chapters2[k] for k in debug_chapters}

    first_pass_path = f'{data_dir}/{first_pass_dir}'
    second_pass_path = f'{data_dir}/{second_pass_dir}'
    verification_pass_path = f'{data_dir}/{verification_pass_dir}'

    matches_path = f'{data_dir}/{matches_dir}'
    if not os.path.exists(matches_path):
        exit('Missing matches directory')

    os.makedirs(first_pass_path, exist_ok=True)
    os.makedirs(second_pass_path, exist_ok=True)
    os.makedirs(verification_pass_path, exist_ok=True)

    # first selection pass
    selection_runs(chapters1, chapters2, first_pass_path=first_pass_path, matches_path=matches_path,
                   model=first_pass_model)

    # verification pass
    for params in verification_models:
        model, think, name = params['model'], params['think'], params['name']
        start = time.time()
        print(f'\n\nVerification with params:\n{params}')
        model_verification_pass_path = f'{verification_pass_path}/{name}'
        os.makedirs(model_verification_pass_path, exist_ok=True)
        create_verified_llm_matches(chapters1, chapters2, first_pass_path=first_pass_path,
                                    verification_pass_path=model_verification_pass_path, matches_path=matches_path,
                                    model=model, think=think)
        print(f'model: {model} took {time.time() - start} seconds\n')

    exit()

    # second selection pass
    selection_runs(chapters1, chapters2, first_pass_path=first_pass_path, matches_path=matches_path,
                   model=second_pass_model, verification_pass_path=verification_pass_path,
                   second_pass_path=second_pass_path, second_run=True)