import numpy as np

DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"

class ComplaintEmbedder:
    def __init__(self, model_name=DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=64):
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

def cosine_topk(query_embedding, embeddings, k=10):
    scores = np.dot(embeddings, query_embedding.reshape(-1))
    idx = np.argsort(scores)[::-1][:k]
    return idx, scores[idx]
