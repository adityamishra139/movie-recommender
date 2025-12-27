from sklearn.metrics.pairwise import cosine_similarity

class SimilarityEngine:
    def __init__(self):
        pass

    def compute_similarity(self, vectors):
        # Convert to float32 to save memory (Render free tier has 512MB limit)
        vectors = vectors.astype('float32')
        return cosine_similarity(vectors)
