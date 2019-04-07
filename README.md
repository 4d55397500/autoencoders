autoencoders
------------


### Single layer autoencoder 
* A single layer encoder & decoder autoencoder without activations [learns projections along principal components](linear_ae_pca.py). For example, training on 2-d plane embedded in 3 dimensions learns projections along that plane.

```
[0.192 -2.652 -0.013] -> [0.204 -2.660 -0.000] 
[1.172 -0.013 -0.054] -> [1.227 -0.041 -0.000]
[0.341 1.030 -0.318] -> [0.660 0.866 -0.000]
```


### Semantic hashing
* [Autoencoder with noise injected](semantic_hashing.py) prior to sigmoid activation forces bit sequence representation preserving information.
Example with one-hot representation of a word set mapping to 3 dimensions.

```
foo -> bit sequence: 000
bar -> bit sequence: 111
car -> bit sequence: 010
jar -> bit sequence: 100
```

