"""State-conditioned clean reference retrieval."""
import numpy as np


def retrieve_k8(states, features, query, k=8):
    states = np.asarray(states, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    mean = states.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(states.std(0, dtype=np.float64), 1e-6).astype(np.float32)
    distance = (((states - mean) / scale) ** 2).sum(-1)
    indices = np.argpartition(distance, k - 1)[:k]
    indices = indices[np.argsort(distance[indices])]
    return np.asarray(features)[indices].mean(0, dtype=np.float32), indices
