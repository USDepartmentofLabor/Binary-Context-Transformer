# BinaryContextTransformer

Efficiently creates two-way interaction terms for sparse, binary data in large datasets and vocabularies.

## Overview

Suppose you are working with a dataset that includes two variables: the text of a message and the type of medium through which it was sent.

| type | message |
|:-|:-|
| text  | text me if ur doing anything 2nite |
| tweet | Holla! Anyone doing anything tonight? |
| email | Sent you a text. What are you doing tonight? |

If you want to distinguish the words in the messages based on the type of medium, you may have to compute every possible combination of words and types. For large datasets that contain many unique words, this is computationally onerous. Moreover, such datasets are usually sparse and most combinations will never occur.

**Base** features, such as **message words**, are variables that may have different meanings in different contexts. **Context** features, such as**message types**, are indicator variables that denote which context a record belongs to. `BinaryContextTransformer` efficiently produces combinations between context features and base features so that they can be used for exploratory analysis or prediction. 

### Benefits

- Follows Scikit-Learn `Transformer` format.
- Excludes interaction terms that appear in only one context.
- For sparse data, `fit_transform` runs in O(S + V), where:
	- N = number of records, rows in the input matrix
    - B = number of base features, columns in the input matrix
    - C = number of context features, columns in the input matrix
    - S = number of entries in the input matrix
        - For sparse matrices, N < S << NB
	- V = number of combinations in resulting vocabulary
		- For sparse interactions, V << BC
- Serialized transformer has similar file size to `CountVectorizer` from Scikit-Learn.

### Drawbacks

- Only designed for binary features.
- May increase model overfitting.
- Must be fit in sequence after other transformers, such as `CountVectorizer`.
- Input must be split into two matrices: one with base features (`X`) and one with context features (`X_context`).

## Contents

- `binarycontexttransformer.py`: Python class for transformer.
- **Examples:** Jupyter notebook with example usage.
- **Rare Occupation Classification:** Jupyter notebook with fictional data to illustrate application of binary interaction terms.

## Acknowledgements

Developed by Vinesh Kannan, Coding It Forward Data Science Fellow at the Bureau of Labor Statistics.
