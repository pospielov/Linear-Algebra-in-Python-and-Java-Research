# Linear Algebra in Python and Java Research

The main idea of this research was to find capabilities of linear algebra calculations in Python and Java.
There are a lot of examples in Python, but almost no in Java, so I hope this will be useful for anyone.

Some performance results you can see on this picture:

![Performance-results](https://user-images.githubusercontent.com/3736126/91655938-3fe65680-eabd-11ea-9afc-efa43e9e6ee8.png)

#### The benchmark
As an example of linear algebra calculation I chose max euclidean distance between 1 and 10000 vectors with previous normalization of them
. `Timing` metric shows time of 1000 repeating calculations on the same vectors (so one benchmark is 10000000 distance calculations). 
In Python I used `timeit` library, but in Java I just used `System.currentTimeMillis()` instead of microbenchmark, but as the most
 calculations were made in C++, I hope it didn't influence a lot for results.


#### Some conclusions

* `Java` has the same performance as `Python` (Actually in both cases the main calculations are made in C++)
* There are too few examples in Java and libraries in beta(`ND4J`) or has non stable API (`Tensorflow`)
* In my opinion `ND4J` is the most ready to use library for `Java`.
* `GraalVM` Python module showed the awful performance (about 86620 seconds in `GraalVM` vs 35 seconds in pure Python `Numpy`)
* No easy way to transform `Python` code to `Java` (Hope `GraalVM` will improve the performance)
* Always use libs compiled with `MKL` if possible (Install `Tensoflow` from `conda`, not `pip`)
* There is always way to optimise algorithm (`tf.math.l2_normalize(embeddings1, axis=1)` works faster then `tf.divide(embeddings1, tf.norm(embeddings1, axis=1, keepdims=True))`)
