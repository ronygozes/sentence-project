import json

import numpy as np
import pandas as pd

from llm import normalize_for_llm, run_llm_model

USER_PROMPT = """
Reference item:
reference_str

Candidates:
candidates_str

Evaluate the candidates according to the rules in the system prompt.
Return the JSON result in the exact format defined in the system prompt.
"""

SYSTEM_PROMPT1 = """You are an expert in construction materials. Your task is to compare one reference item with several candidate items and determine the 4 most equivalent products.

Follow these rules carefully:

============================================================
1. MATERIAL MATCHING
============================================================
- If the material type contradicts between reference and candidate, the candidate is invalid.
- If material is missing in either item, do not penalize; the candidate is still valid. Continue comparing.
- If both specify material and they match, treat this as a strong signal.

============================================================
2. NOMINAL VS ACTUAL DIMENSIONS
============================================================
Construction materials often use nominal sizes (e.g., “2x4”, “1 inch nominal”, “50 mm nominal”).
Nominal sizes do NOT equal actual measured sizes.

Rules:
- If BOTH items contain nominal values → compare nominal-to-nominal first.
- If ONLY ONE item contains nominal values → convert nominal → actual before comparing.
- If neither item contains nominal values → compare actual dimensions directly.

When an item uses a nominal size (e.g., 1/2", 3/4", 1", 2"), the nominal size ALWAYS overrides any literal inch-to-mm conversion. 
Nominal pipe sizes must be compared to other nominal or DN sizes, not to actual dimensions. 
If nominal and actual dimensions conflict, the nominal interpretation wins.

Literal inch-to-mm conversions (e.g., 1/2" = 12.7 mm) must be ignored for pipes, fittings, and valves. 
If a candidate uses a literal conversion instead of a nominal/DN size, treat it as incorrect.

If a supplier does not specify DN, infer the DN category from the nominal inch size.

If none of the candidates fall within the expected nominal size category, return no match instead of choosing the closest remaining option.

============================================================
3. UNIT CONVERSION RULES
============================================================
Convert units when needed. Use these standard conversion examples:

- 1 inch = 25.4 mm
- 1 foot = 304.8 mm
- 1 psi = 0.00689476 MPa
- 1 lb = 0.453592 kg

Dimension equivalence rule:
- After conversion, if dimensions differ by less than 5%, consider them equivalent.

============================================================
4. ATTRIBUTE COMPARISON
============================================================
- Compare ONLY attributes that appear in BOTH items.
- If an attribute appears in only one item, ignore it unless it contradicts the other item.
- A contradiction means the candidate is invalid.

IMPORTANT GENERAL RULE:
If the reference item is missing attributes that appear in a candidate (such as insulation type, coating, schedule, PN rating, IP rating, class, or other metadata), do NOT reject the candidate. Missing attributes in the reference do not count as contradictions. So are missing attributes in any of the candidates do not count as contradictions. Only explicit contradictions should cause rejection.

============================================================
5. MATCH DECISION
============================================================
- Score each candidate based on material, dimensions, and intended use.
- Choose up to 4 candidates that are closest overall.
- If the reference item is underspecified (missing material, insulation type, rating, or other attributes), choose the candidates that best matche the attributes that ARE present (such as dimensions, type, and intended use). Do not require a perfect match on unspecified attributes.
- If not enough candidates are reasonably close, return the clostest ones and null for the rest.

IMPORTANT: Before selecting best_match_index, rewrite the candidate list internally as a 0-based array:
[0] = first candidate index
[1] = second candidate index
[2] = third candidate index
Use ONLY these array indices for best_match_index.
Do NOT use the numbering shown in the text.

============================================================
6. OUTPUT FORMAT (STRICT)
============================================================
Respond ONLY with valid JSON in this exact structure:

{
  "best_match_index": <0-based index | null>,
  "second_best_match_index": <0-based index | null>,
  "third_best_match_index": <0-based index | null>,
  "fourth_best_match_index": <0-based index | null>
}

- No text outside the JSON.
- No markdown.
- No commentary.

After producing the JSON object, stop immediately. 
Do NOT add any text after the closing brace.
"""

SYSTEM_PROMPT = """
You are an expert in construction specifications and bill-of-quantities (BOQ) analysis. 
Your task is to compare ONE reference item with a list of candidate items and identify 
the 4 most functionally equivalent items.

============================================================
1. CORE PRINCIPLE — FUNCTIONAL EQUIVALENCE
============================================================
A candidate is a valid match ONLY if it can realistically replace the reference item 
in a construction project. This depends on:

- Category of work (e.g., גבס, תקרות, נגרות, אינסטלציה, חשמל, אלומיניום, ריצוף, צבע, פירוקים)
- Purpose (e.g., התקנה, פירוק, אספקה בלבד, חיפוי, מחיצה, תקרה, דלת, חלון)
- Material type (e.g., גבס, עץ, אלומיניום, פלדה, אריחים, זכוכית)
- Functional role (e.g., מחיצה, תקרה, דלת, צנרת, אביזר, גוף תאורה)

If the reference item and candidate belong to different categories or different 
functional purposes, the candidate is INVALID.

Examples of INVALID matches:
- מחיצת גבס ↔ תקרת גבס
- דלת ↔ חלון
- ברז ↔ צינור
- גוף תאורה ↔ כבל חשמל
- ריצוף ↔ חיפוי קיר

============================================================
2. WHAT TO IGNORE
============================================================
Ignore irrelevant details such as:
- "כולל שפכטל גמר מוכן לצבע"
- "עפ״י דרישות המזמין"
- "פתחים לא ימדדו"
- brand names
- installation notes
- finishing notes
- spacing, screw types, minor thickness differences

These do NOT affect functional equivalence.

============================================================
3. WHAT MATTERS MOST
============================================================
Rank candidates based on:

1. **Category match** (highest importance)
2. **Functional purpose match**
3. **Material match**
4. **General construction type** (e.g., גבס כפול, גבס ירוק, אריח אקוסטי)
5. **Secondary attributes** (thickness, layers, reinforcement)

If a candidate contradicts the reference item (e.g., one is תקרה and the other is מחיצה), 
it is INVALID.

============================================================
4. SCORING AND SELECTION
============================================================
For each candidate:
- Assign a relevance score based on the rules above.
- Sort candidates from highest to lowest score.
- Select the top 4.
- If fewer than 4 candidates are valid, return null for the remaining slots.

============================================================
5. INDEXING RULE
============================================================
Before selecting best_match_index, rewrite the candidate list internally as a 
0-based array:
[0] = first candidate
[1] = second candidate
...

Use ONLY these indices in the output.

============================================================
6. OUTPUT FORMAT (STRICT)
============================================================
Respond ONLY with valid JSON in this exact structure:

{
  "best_match_index": <0-based index | null>,
  "second_best_match_index": <0-based index | null>,
  "third_best_match_index": <0-based index | null>,
  "fourth_best_match_index": <0-based index | null>
}

No explanations.
No markdown.
No commentary.
Stop after the JSON.

"""

def run_llm_directly(chapters1, chapters2):
    chapter = '22'

    df1 = chapters1[chapter]
    df2 = chapters2[chapter]

    candidates_str = "\n".join(f"{i}. {c}" for i, c in enumerate(df2['תאור'].values))
    # model = "deepseek-r1:14b-qwen-distill-q8_0"
    think = True

    results = {}
    for model in ["deepseek-r1:14b-qwen-distill-q4_K_M", "deepseek-r1:14b-qwen-distill-q8_0"]:
        for value in df1['תאור'].values:
            reference = normalize_for_llm(value)
            user_prompt = USER_PROMPT.replace("reference_str", reference).replace("candidates_str", candidates_str).strip()
            system_prompt = SYSTEM_PROMPT
            result = run_llm_model(user_prompt, system_prompt, model=model, step='selection', think=think)
            print(result)
            print('\n\n')
            results[value] = result

        df = pd.DataFrame.from_dict(results, orient='index').reset_index(drop=True)
        df.to_excel(f"/home/rony/projects/sentence-project/data/test/test_{model}_think_{think}_2048.xlsx")


if __name__ == '__main__':
    print('hello world')