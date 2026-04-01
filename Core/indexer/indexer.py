import faiss
import numpy as np
from typing import List, Tuple, Dict

def build_index(embeddings: Dict[int, np.ndarray]) -> faiss.Index:
    if not embeddings:
        raise ValueError("Aucun embedding fourni.")

    # Dimension des vecteurs
    dim = next(iter(embeddings.values())).shape[0]

    ids = np.array(list(embeddings.keys()), dtype=np.int64)

    vectors = np.vstack([
        np.ascontiguousarray(vec, dtype=np.float32)
        for vec in embeddings.values()
    ])
    # Index FAISS
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(vectors, ids)

    return index


def search_top_k(index: faiss.Index, query: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    k = min(k, index.ntotal)
    D, I = index.search(q, k)
    results = [(int(cid), float(score)) for cid, score in zip(I[0], D[0]) if cid != -1]
    print(f"Recherche terminée : {len(results)} résultats retournés (top {k}).")
    return results
