import ollama
import json
import time
import re

import pandas as pd

from model_rules import models_data
from llm_tests import test_cases


def normalize_for_llm(text: str) -> str:
    # Replace inch symbol with safe text
    text = re.sub(r'(\d)\s*"', r'\1 inch', text)
    text = text.replace('מ"מ', 'ממ')  # Hebrew mm
    text = text.replace('"', '')      # fallback: remove any remaining quotes
    return text


def parse_llm_response(response) -> dict:
    original_content = response["message"]["content"].strip()

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


def create_user_prompt(reference: str, candidates: list[str], previous_result: dict | None = None):
    # Build candidate list string
    candidates_str = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))

    # Build the user prompt
    if previous_result is None:
        # First pass: simple prompt
        user_prompt = (
            models_data["USER_PROMPT"]
            .replace("candidates_str", candidates_str)
            .replace("reference_str", reference)
            .strip()
        )
    elif ((previous_result.get("best_match_index") is None) or
          (previous_result.get("best_match") is None) or
          (previous_result.get("confidence") in ["low", "medium"])):
        # Fallback pass: include previous model's output
        previous_json = json.dumps(previous_result, ensure_ascii=False, indent=2)
        user_prompt = (
            models_data["USER_PROMPT_SECOND_RUN"]
            .replace("candidates_str", candidates_str)
            .replace("reference_str", reference)
            .replace("previous_json", previous_json)
            .strip()
        )
    else:
        user_prompt = None

    return user_prompt


def find_best_material_match(user_prompt: str, model: str) -> dict:

    system_prompt = models_data["SYSTEM_PROMPT"]
    
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
    result = parse_llm_response(response)
    return result


def run_llm(items: dict, model: str, previous_results: dict | None = None):
    print('\n\nStarting llm run!!!!!!!!!!!!!!\n')

    results = {}
    for reference in items:
        print(f'Starting with case: {reference}')
        start = time.time()

        normalizes_reference = normalize_for_llm(reference)

        candidates = items[reference]
        normalized_candidates = []
        for candidate in candidates:
            normalized_candidates.append(normalize_for_llm(candidate))

        user_prompt =  create_user_prompt(normalizes_reference, normalized_candidates, previous_results)
        if user_prompt is None:
            results[reference] = {"best_match_index": None, "confidence": "high", "reason": "exists already"}
            continue

        result = find_best_material_match(user_prompt, model=model)

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
        print(f'{reference} \ntook {time.time()-start} seconds\n')
        print(result)
        print('\n\n')
    df = pd.DataFrame.from_dict(results, orient='index').reset_index(drop=True)
    return df


if __name__ == "__main__":
    run_llm(items=test_cases)