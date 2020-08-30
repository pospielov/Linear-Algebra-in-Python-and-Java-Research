import numpy as np
import tensorflow as tf
import timeit

embeddings = np.genfromtxt("embeddings.txt", delimiter=',')
embeddings2 = embeddings[1:10001]
embeddings2 = tf.math.l2_normalize(embeddings2, axis=1)


def test():
    embeddings1 = embeddings[0:1]

    embeddings1 = tf.math.l2_normalize(embeddings1, axis=1)
    diff = tf.subtract(embeddings1, embeddings2)
    dist = tf.reduce_sum(tf.square(diff), axis=1)
    # print(dist)
    # print(tf.argmax(dist))
    return tf.reduce_max(dist)


# print(test())
print("started")
print(timeit.repeat("test()", setup="from __main__ import test;  gc.enable();", number=1000, repeat=3))

# [10.683477842000684, 10.66390071800015, 10.703822994999427]
