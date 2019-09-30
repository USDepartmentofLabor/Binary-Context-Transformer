# BinaryContextTransformer

Efficiently creates two-way interaction terms for sparse, binary data in large datasets and vocabularies.

## Overview

Suppose you are working with a dataset that includes two predictor variables: the text of a message and the type of medium through which it was sent.

| type | message |
|:-|:-|
| text  | text me if ur doing anything 2nite |
| tweet | Holla! Anyone doing anything tonight? |
| email | Sent you a text. What are you doing tonight? |

If you want to distinguish the words in the messages based on the type of medium, you may have to compute every possible combination of words and types. For large datasets that contain many unique words, this is computationally impractical. Moreover, such datasets are usually sparse and most combinations will never occur.

`BinaryContextTransformer` efficiently produces combinations between *context* features (message type) and *base* features (message words) so that they can be used for exploratory analysis or prediction.

### Benefits

- Follows Scikit-Learn `Transformer` format.
- For sparse data, `fit_transform` runs in O(V + NW), where:
	- N = number of records
	- C = number of context features
	- B = number of base features
	- V = number of combinations in resulting vocabulary
		- For sparse interactions, V << CB
	- W = average number of base and context features active per record
		- For sparse interactions when each record belongs to only one context, W << B
- Excludes interaction terms that appear in only one context
- Serialized transform has similar memory size to `CountVectorizer` from Scikit-Learn

### Drawbacks

- Only designed for binary features
- May increase model overfitting
- Must be fit in sequence

## Contents

- `binarycontexttransformer.py`: Python class for transformer.
- **Examples:** Jupyter notebook with example usage.
- **Rare Occupation Classification:** Jupyter notebook with fictional data to illustrate application of binary interaction terms.

## Acknowledgements

Developed by Vinesh Kannan, Coding It Forward Data Science Fellow at the Bureau of Labor Statistics.
