from sklearn.cluster import HDBSCAN

def cluster_embeddings(embeddings, min_cluster_size=8, min_samples=4):
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )
    labels = model.fit_predict(embeddings)
    return labels, model
