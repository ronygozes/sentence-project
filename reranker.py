from ollama import chat, generate

# Update this to the exact name you see in 'ollama list'
MODEL = "qwen3:8b" 
MODEL = "material-reranker"

def get_engineering_score(query, doc):
    # We give the model a technical persona so it respects units
    prompt = f"""[Technical Comparison]
                Query: {query}
                Document: {doc}
                Task: Rate technical equivalence from 0.00 to 1.00. 
                Note: 2-inch is ~50mm. Sch40 and PN16 are pressure/wall specs.
                Score:"""

    # response = ollama.generate(
    #     model=MODEL,
    #     prompt=prompt,
    #     options={
    #         "temperature": 0,
    #         "num_predict": 10,
    #         "stop": ["\n", "Reasoning"]
    #     }
    # )

    response = chat(
    model='qwen3:8b',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
    print(response.message.content)
    exit()
    return response #['response'].strip()

# Test Case
target = "2-inch PVC Pipe Sch40"
options = ["50mm PVC Pipe PN16", "40mm PVC Pipe", "10mm Copper"]

print(f"--- Engineering Rank for: {target} ---")
for item in options:
    score = get_engineering_score(target, item)
    print(score['context'])
    exit()
    