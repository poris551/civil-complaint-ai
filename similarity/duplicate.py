import numpy as np

def find_similar(query_embedding, embeddings, top_k=10, threshold=0.65):
    scores = embeddings @ query_embedding.reshape(-1)
    order = np.argsort(scores)[::-1]
    result = []
    for idx in order[:top_k]:
        if scores[idx] >= threshold:
            result.append((int(idx), float(scores[idx])))
    return result
