import dask
import dask.array as da
from dask.distributed import Client, progress

import numpy as np
import timeit

client = Client(processes=False, threads_per_worker=8, n_workers=1, memory_limit='4GB')

embeddings = np.genfromtxt("embeddings.txt", delimiter=',')
embeddings = da.from_array(embeddings, chunks=2048)


def test():
    embeddings1 = embeddings[0:1]
    embeddings2 = embeddings[1:10001]

    embeddings1 = dask.array.divide(embeddings1, dask.array.linalg.norm(embeddings1, axis=1, keepdims=True))
    embeddings2 = dask.array.divide(embeddings2, dask.array.linalg.norm(embeddings2, axis=1, keepdims=True))
    diff = dask.array.subtract(embeddings1, embeddings2)
    dist = dask.array.sum(dask.array.square(diff), 1)
    # print(dask.array.argmax(dist).compute())
    return dask.array.max(dist).compute()


# print(test())
print("started")
print(timeit.repeat("test()", setup="from __main__ import test;  gc.enable();", number=1000, repeat=3))
