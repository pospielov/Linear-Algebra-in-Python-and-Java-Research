import numpy as np
import timeit

embeddings = np.genfromtxt("embeddings.txt", delimiter=',')
embeddings2 = embeddings[1:10001]
embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)


def test():
    embeddings1 = embeddings[0:1]

    embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    # print(np.argmax(dist));
    return np.max(dist)


# print(test())
print("started")
print(timeit.repeat("test()", setup="from __main__ import test;  gc.enable();", number=1000, repeat=3))

# [16.546649905999402, 16.559010640999986, 16.606338937000146]
