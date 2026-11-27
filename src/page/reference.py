"""State-conditioned clean reference retrieval."""

import numpy as np


def _state_distance(states, query):
    states = np.asarray(states, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    mean = states.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(states.std(0, dtype=np.float64), 1e-6).astype(np.float32)
    return (((states - mean) / scale) ** 2).sum(-1)


def retrieve_k8_bank(states, features, query, k=8):
    """Return the K pose-matched clean feature tensors, their mean, and row IDs."""
    distance = _state_distance(states, query)
    indices = np.argpartition(distance, k - 1)[:k]
    indices = indices[np.argsort(distance[indices])]
    bank = np.asarray(features)[indices].astype(np.float32, copy=False)
    return bank, bank.mean(0, dtype=np.float32), indices


def retrieve_k8(states, features, query, k=8):
    """Return the pose-matched K-feature mean and row IDs."""
    _, reference, indices = retrieve_k8_bank(states, features, query, k=k)
    return reference, indices
