autoencoders
------------


### Single layer autoencoder 
* A single layer encoder & decoder autoencoder without activations [learns the encoder as projections along principal components](linear_ae_pca.py). For example, training on a 2-d plane embedded in 3 dimensions trains the encoder to be projections along that plane.

```
...
Epoch: 396 Epoch loss: 1.3975636287316456e-12
Epoch: 397 Epoch loss: 1.2718209528605508e-12
Epoch: 398 Epoch loss: 2.042226219626126e-12
Epoch: 399 Epoch loss: 5.955678238447372e-12
---------------------------------------------
RECONSTRUCTION ~ PCA PROJECTIONS
[1.290 0.483 2.077] -> [1.997 -1.057 0.000] prj error: 1.6941322492059756
[0.238 0.397 0.000] -> [0.238 0.397 0.000] prj error: 0.00031562439973610134
[-0.492 -1.536 -0.252] -> [-0.578 -1.349 -0.000] prj error: 0.20569515712402547
[-0.845 0.247 0.640] -> [-0.628 -0.227 0.000] prj error: 0.5220061477252258
[0.114 0.088 -1.163] -> [-0.282 0.951 0.000] prj error: 0.94912850831395
[-0.737 0.455 1.332] -> [-0.284 -0.533 0.000] prj error: 1.0868102526308359
[-0.231 -0.654 0.164] -> [-0.175 -0.776 0.000] prj error: 0.13416687617840845
[-0.106 0.693 -0.539] -> [-0.290 1.092 0.000] prj error: 0.43945521851173674
[-0.684 -1.163 0.460] -> [-0.528 -1.504 -0.000] prj error: 0.3749856354477562
[0.406 1.430 -0.287] -> [0.308 1.643 0.000] prj error: 0.23409751458246955

```


### Semantic hashing on categorical elements
* [Autoencoder with noise injected](semantic_hashing.py) prior to sigmoid activation forces bit sequence representation preserving information.
[Example with one-hot representation of a word set](categorical_semantic_hashing.py) mapping to 3 dimensions.

```
...
Epoch: 96 Epoch loss: 8.67664722725749 Noise std: 113.59573078181967
Epoch: 97 Epoch loss: 11.23818837851286 Noise std: 119.27551732091065
Epoch: 98 Epoch loss: 10.360635101795197 Noise std: 125.23929318695619
Epoch: 99 Epoch loss: 12.108762972056866 Noise std: 131.50125784630401
---------------------------------------------
foo -> bit sequence: 000
bar -> bit sequence: 101
car -> bit sequence: 110
jar -> bit sequence: 011
```


### Semantic hashing on bag of words
* [Example on bag of words](bag_of_words_semantic_hashing.py) mapping to 5 dimensions.

```
...
Epoch: 96 Epoch loss: 13.278802040964365 Noise std: 113.59573078181967
Epoch: 97 Epoch loss: 13.692236740142107 Noise std: 119.27551732091065
Epoch: 98 Epoch loss: 9.443496888503432 Noise std: 125.23929318695619
Epoch: 99 Epoch loss: 12.24506833218038 Noise std: 131.50125784630401
---------------------------------------------
this is a cat -> bit sequence: 10100
this is my hat -> bit sequence: 10111
this is a hat -> bit sequence: 10101
this is a big hat -> bit sequence: 00101
this is one fat cat -> bit sequence: 11110
this is a smelly rat -> bit sequence: 11101
that is completely random -> bit sequence: 01000
like night versus day -> bit sequence: 10011
```