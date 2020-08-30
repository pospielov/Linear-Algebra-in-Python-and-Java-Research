import numpy as np
import tensorflow as tf
import timeit

embeddings = np.genfromtxt("embeddings.txt", delimiter=',')


def test():
    embeddings1 = embeddings[0:1]
    embeddings2 = embeddings[1:10001]

    # embeddings1 = tf.divide(embeddings1, tf.norm(embeddings1, axis=1, keepdims=True))
    # embeddings2 = tf.divide(embeddings2, tf.norm(embeddings2, axis=1, keepdims=True))
    embeddings1 = tf.math.l2_normalize(embeddings1, axis=1)
    embeddings2 = tf.math.l2_normalize(embeddings2, axis=1)
    diff = tf.subtract(embeddings1, embeddings2)
    dist = tf.reduce_sum(tf.square(diff), axis=1)
    # print(dist)
    # print(tf.argmax(dist))
    return tf.reduce_max(dist)


# print(test())
print("started")
print(timeit.repeat("test()", setup="from __main__ import test;  gc.enable();", number=1000, repeat=3))

