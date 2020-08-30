import numpy as np
import timeit

embeddings = np.genfromtxt("embeddings.txt", delimiter=',')


def test():
    embeddings1 = embeddings[0:1]
    embeddings2 = embeddings[1:10001]

    embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    # print(np.argmax(dist));
    return np.max(dist)


# print(test())
print("started")
print(timeit.repeat("test()", setup="from __main__ import test;  gc.enable();", number=1000, repeat=3))
