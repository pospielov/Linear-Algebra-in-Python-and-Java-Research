import numpy as np
from scipy.spatial.distance import cdist
import scipy
import timeit

embeddings = np.genfromtxt("embeddings.txt", delimiter=',')


def test():
    embeddings1 = embeddings[0:1]
    embeddings2 = embeddings[1:10001]

    embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)
    dist = cdist(XA=embeddings1, XB=embeddings2, metric="euclidean")
    # print(np.argmax(dist))
    return np.max(dist)


# print(test())
print("started")
print(timeit.repeat("test()", setup="from __main__ import test;  gc.enable();", number=1000, repeat=3))
