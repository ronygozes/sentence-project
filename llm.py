import ollama
import json
import time
import re
from model_rules import models_data
from llm_tests import test_cases


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


def find_best_material_match(reference: str, candidates: list[str], model: str, run_num: int = 0, previous_result: dict | None = None) -> dict:

    # Build candidate list string
    candidates_str = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))

    # Build the user prompt
    if run_num == 0:
        # First pass: simple prompt
        user_prompt = (
            models_data["USER_PROMPT"]
            .replace("candidates_str", candidates_str)
            .replace("reference_str", reference)
            .strip()
        )
    else:
        # Fallback pass: include previous model's output
        previous_json = json.dumps(previous_result, ensure_ascii=False, indent=2)
        user_prompt = (
            models_data["USER_PROMPT_SECOND_RUN"]
            .replace("candidates_str", candidates_str)
            .replace("reference_str", reference)
            .replace("previous_json", previous_json)
            .strip()
        )

    system_prompt = models_data["SYSTEM_PROMPT"]
    
    # Load model-specific settings
    model_params = models_data['models'][model]
    think = model_params["think"]
    num_predict = 200 if not think else 4000
    print(num_predict)

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


def normalize_for_llm(text: str) -> str:
    # Replace inch symbol with safe text
    text = re.sub(r'(\d)\s*"', r'\1 inch', text)
    text = text.replace('מ"מ', 'ממ')  # Hebrew mm
    text = text.replace('"', '')      # fallback: remove any remaining quotes
    return text


def main():
    print('Starting first run!!!!!!!!!!!!!!')
    first_results = {}
    for key in test_cases:
        print(f'Starting with key: {key}')
        start = time.time()
        candidates = test_cases[key]
        normalizes_candidates = []
        for candidate in candidates:
            normalizes_candidates.append(normalize_for_llm(candidate))
        result = find_best_material_match(key, normalizes_candidates, model="qwen3.5:latest", run_num=0)
        if result.get("best_match_index") is not None:
            try:
                for i, c in enumerate(normalizes_candidates):
                    if c.strip() == result["best_match"].strip():
                        result["best_match_index"] = i
                        break
            except:
                print(f'result is broken: {result}')
                exit()
        first_results[key] = {'result': result, 'time': time.time()-start}
        print(f'{key} \ntook {time.time()-start} seconds\n')

    second_results = {}
    print('Starting second run!!!!!!!!!!!!!!')
    for key in test_cases:
        result = first_results[key]
        if not ((result.get("best_match_index") is None) or (result.get("best_match") is None) or (result.get("confidence") in ["low", "medium"])):
            second_results[key] = None
            continue
        
        print(f'Starting with key: {key}')
        start = time.time()
        candidates = test_cases[key]
        normalizes_candidates = []
        for candidate in candidates:
            normalizes_candidates.append(normalize_for_llm(candidate))
        new_result = find_best_material_match(key, normalizes_candidates, model="deepseek-r1:14b-qwen-distill-q4_K_M", run_num=1, previous_result=first_results[key])
        second_results[key] = {'result': new_result, 'time': time.time()-start}

    for key in first_results:
        print(first_results[key])
        print('\n\n')

    for key in second_results:
        print(second_results[key])
        print('\n\n')


if __name__ == "__main__":
    main()