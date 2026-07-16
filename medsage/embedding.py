import numpy as np

def cosine_matrix(a, b=None):
    if b is None: b=a
    a=np.asarray(a, dtype='float32'); b=np.asarray(b, dtype='float32')
    a=a/(np.linalg.norm(a, axis=1, keepdims=True)+1e-12)
    b=b/(np.linalg.norm(b, axis=1, keepdims=True)+1e-12)
    return a @ b.T

def cosine_vector(a, b): return cosine_matrix(a, np.asarray(b, dtype='float32').reshape(1,-1)).reshape(-1)

def save_embeddings(path, embeddings):
    import pathlib
    path=pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True); np.save(path, embeddings.astype('float32'))

def load_embeddings(path): return np.load(path).astype('float32')

class SentenceEmbedder:
    def __init__(self, model_path, device='auto', normalize_embeddings=True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, device=None if device == 'auto' else device)
        self.normalize_embeddings = normalize_embeddings
    def encode(self, texts, batch_size=64, show_progress_bar=True):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings).astype('float32')
