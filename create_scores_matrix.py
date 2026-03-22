import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from huggingface_hub import login
import os
import logging

# Silence SentenceTransformers warnings
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Silence HuggingFace Transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Silence PyTorch warnings
logging.getLogger("torch").setLevel(logging.ERROR)

def testing_model(sentences):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  

    embeddings = []
    scores = []

    for i in range(len(sentences)):
        embeddings.append(model.encode(sentences[i], convert_to_tensor=True))

    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            score = util.cos_sim(embeddings[i], embeddings[j]).item()
            scores.append(f"Cosine similarity for sentences {i} and {j}: {score}")
    return scores


def medium_model(sentences):
    # Load the multilingual cross-encoder
    model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    scores = []

    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            # Predict similarity score
            score = model.predict([(sentences[i], sentences[j])])[0]
            scores.append(f"Cosine similarity for sentences {i} and {j}: {score}")
    return scores


# Only use when ready for real work, will take long time
def prod_model(sentences):
    model = SentenceTransformer("intfloat/multilingual-e5-large") 

    embeddings = []
    scores = []

    for i in range(len(sentences)):
        embeddings.append(model.encode("query: " + sentences[i], convert_to_tensor=True))

    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            score = util.cos_sim(embeddings[i], embeddings[j]).item()
            scores.append(f"Cosine similarity for sentences {i} and {j}: {score}")
    return scores


def compute_similarity_matrix(model, list1, list2):
    # Encode both lists
    emb1 = model.encode(list1, convert_to_tensor=True, device="cuda")
    emb2 = model.encode(list2, convert_to_tensor=True, device="cuda")

    # Compute cosine similarity matrix
    sim_matrix = util.cos_sim(emb1, emb2)
    sim_matrix = sim_matrix.cpu().numpy()

    return sim_matrix


def load_model():
    # model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    model = SentenceTransformer("intfloat/multilingual-e5-large") 
    return model