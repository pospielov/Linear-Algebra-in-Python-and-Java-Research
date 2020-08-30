import site

import numpy as np


def test():
    embeddings1 = embeddings[0:1]
    embeddings2 = embeddings[1:10001]

    embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    return np.max(dist)


print("started")
