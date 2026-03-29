import ollama
import json
import time
import re
from model_rules import models_data
from llm_tests import test_cases


def parse_llm_response_broken(response) -> dict:
    content = response["message"]["content"].strip()
    original_content = response["message"]["content"].strip()
    
    

    print("original content:\n", content)
    
    # strip <think>...</think> block if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = content.strip()
    
    # strip ```json ... ``` or ``` ... ``` fences
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f'there was an error, the original content is: {original_content}')
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        print(f"parse error. the content:\n{content}\n\n\n")
        return {"best_match_index": None, "confidence": "low", "reason": "parse error"}


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
        return {"best_match_index": None, "confidence": "low", "reason": "parse error"}

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


def find_best_material_match_v1(reference: str, candidates: list[str], model: str="qwen3.5:4b", allow_think: bool=False) -> dict:
    
    candidates_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    model_params = models_data[model]

    user_prompt = model_params["USER_PROMPT"].replace('candidates_str', candidates_str).replace('reference_str', reference)
    system_prompt = model_params["SYSTEM_PROMPT"]
    print(user_prompt)
    exit()
    
    response = ollama.chat(
        model=model,
        options={"temperature": 0, "num_predict": 500},
        think=model_params["think"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    result = parse_llm_response(response)

    return result


def find_best_material_match(
    reference: str,
    candidates: list[str],
    model: str = "qwen3.5:latest",
    allow_think: bool = False,
    previous_result: dict | None = None
) -> dict:

    # Build candidate list string
    candidates_str = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))

    # Load model-specific settings
    model_params = models_data[model]

    # Build the user prompt
    if not allow_think:
        # First pass: simple prompt
        user_prompt = (
            model_params["USER_PROMPT"]
            .replace("candidates_str", candidates_str)
            .replace("reference_str", reference)
        )
    else:
        # Fallback pass: include previous model's output
        previous_json = json.dumps(previous_result, ensure_ascii=False, indent=2)

        user_prompt = f"""
This is a second-pass analysis. The first model produced a low-confidence or null result.

Reference item:
{reference}

Candidate items:
{candidates_str}

Here is the previous model's output:
{previous_json}

Your task:
- Re-evaluate the candidates according to the rules in the system prompt.
- Consider the previous model's reasoning, but do NOT rely on it blindly.
- Perform deeper reasoning, including unit conversions and nominal/actual comparisons.
- If the previous model was wrong, correct it.
- If no match exists, return null.

Return the JSON result in the exact format defined in the system prompt.
""".strip()

    system_prompt = model_params["SYSTEM_PROMPT"]

    # Call the model
    response = ollama.chat(
        model=model,
        options={"temperature": 0, "num_predict": 200},
        think=model_params["think"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Parse the result
    result = parse_llm_response(response)

    # Fallback logic
    if False and not allow_think:
        # Only fallback if the first model was weak
        if result["best_match_index"] is None or result["confidence"] == "low":
            return find_best_material_match(
                reference,
                candidates,
                model="qwen3:8b",
                allow_think=True,
                previous_result=result
            )

    return result


def normalize_for_llm(text: str) -> str:
    # Replace inch symbol with safe text
    text = re.sub(r'(\d)\s*"', r'\1 inch', text)
    text = text.replace('מ"מ', 'ממ')  # Hebrew mm
    text = text.replace('"', '')      # fallback: remove any remaining quotes
    return text

# Example

print('Starting!!!!!!!!!!!!!!')
first_results = {}
for key in test_cases:
    print(f'Starting with key: {key}')
    start = time.time()
    candidates = test_cases[key]
    normalizes_candidates = []
    for candidate in candidates:
        normalizes_candidates.append(normalize_for_llm(candidate))
    result = find_best_material_match(key, normalizes_candidates)
    if result["best_match_index"] is not None:
        for i, c in enumerate(normalizes_candidates):
            if c.strip() == result["best_match"].strip():
                result["best_match_index"] = i
                break
    first_results[key] = {'result': result, 'time': time.time()-start}
    print(f'{key} \ntook {time.time()-start} seconds\n')

for key in first_results:
    print(first_results[key])
    print('\n\n')

# print('\n\n\n\n\n\n\nstarting with secondary results')
# second_results = {}
# for key in first_results:
#     if (first_results[key]['result']['best_match_index'] is not None) and (first_results[key]['result']['confidence'] == 'high'):
#         second_results[key] = None
#     else:
#         print(f'Starting with key: {key}')
#         start = time.time()
#         result = find_best_material_match(key, test_cases[key], model="qwen3:8b")
#         second_results[key] = {'result': result, 'time': time.time()-start}
#         print(f'{key} \ntook {time.time()-start} seconds\n')

# for key in second_results:
#     print(second_results[key])
#     print('\n\n')

# {
#   "best_match_index": 1,
#   "confidence": "high", 
#   "reason": "2 inch PVC pipe equals ~50.8mm, matches candidate 1",
#   "unit_conversion_applied": true
# }