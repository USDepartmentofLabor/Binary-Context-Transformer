# BinaryContextTransformer

Efficiently creates two-way interaction terms for sparse, binary data in large datasets and vocabularies.

- [Overview](#overview)
    - [Benefits](#benefits)
    - [Drawbacks](#drawbacks)
    - [Related Tools](#related-tools)
- [Repository Contents](#repository-contents)
- [Example](#example)
- [Acknowledgements](#acknowledgements)

## Overview

Suppose you are working with a dataset that includes two variables: the text of a message and the type of medium through which it was sent.

| type | message |
|:-|:-|
| text  | text me if ur doing anything 2nite |
| tweet | Holla! Anyone doing anything tonight? |
| email | Sent you a text. What are you doing tonight? |

If you want to distinguish the words in the messages based on the type of medium, you may have to compute every possible combination of words and types. For large datasets that contain many unique words, this is computationally onerous. Moreover, such datasets are usually sparse and most combinations will never occur.

**Base** features, such as **message words**, are variables that may have different meanings in different contexts. **Context** features, such as**message types**, are indicator variables that denote which context a record belongs to. `BinaryContextTransformer` efficiently produces combinations between context features and base features so that they can be used for exploratory analysis or prediction.

Examples of binary context features from the table above are `text_x_anything` or `tweet_x_anything`. These combination features may be useful if the meaning of the word `"anything"` differs based on the medium it was sent through.

_**Reminder:** This is a hypothetical example. Emails, texts, and tweets contain personal information. If you are actually analyzing such data, make appropriate considerations for consent and privacy._

### Benefits

- Follows Scikit-Learn `Transformer` format.
- Excludes interaction terms that appear in only one context.
- For sparse data, `fit_transform` runs in `O(S + V)`, where:
	- `N` = number of records, rows in the input matrix
    - `B` = number of base features, columns in the input matrix
    - `C` = number of context features, columns in the input matrix
    - `S` = number of entries in the input matrix
        - For sparse matrices, `N < S << N x B`
	- `V` = number of combinations in resulting vocabulary
		- For sparse interactions, `V << B x C`
- Input matrices will be converted to [compressed sparse column (CSC)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html) format, if not already in that format. The output matrix will also be in CSC format.
- Serialized transformer has similar file size to `CountVectorizer` from Scikit-Learn.
- Accepts a custom progress bar function, such as [tqdm](https://github.com/tqdm/tqdm) or a similar format.

### Drawbacks

- Only designed for binary features.
- May increase model overfitting.
- Must be fit in sequence after other transformers, such as `CountVectorizer`.
- Input must be split into two matrices: one with base features (`X`) and one with context features (`X_context`).

### Related Tools

`BinaryContextTransformer` is similar to [`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) in Scikit-Learn, which supports other variable types. `PolynomialFeatures` can also generate interaction terms of any degree, not just two-way interactions. However, since every possible combination of features is considered, `PolynomialFeatures` runs in polynomial time at the requested degree.

`BinaryContextTransformer` focuses on just binary data and takes advantage of sparsity to compute interaction terms in `O(S + V)` instead of `O(N x (C + B))`, as described above.

## Repository Contents

- [`binarycontexttransformer.py`](binarycontexttransformer.py): Python class for transformer.
- [`Examples.ipynb`](Examples.ipynb): Jupyter notebook with example usage on hypothetical data.
- [`Rare Occupation Classification.ipynb`](Rare%20Occupation%20Classification.ipynb): Jupyter notebook with hypothetical data to illustrate application of binary context terms.

## Example

This example shows how to create the binary context features described above. Usually, other transformers are used to convert input data into matrix form before using `BinaryContextTransformer`.

```python
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from binarycontexttransformer import BinaryContextTransformer
>>> 
>>> 
>>> data = [
...     ("text", "text me if ur doing anything 2nite"),
...     ("tweet", "Holla! Anyone doing anything tonight?"),
...     ("email", "Sent you a text. What are you doing tonight?")
... ]
>>> df = pd.DataFrame(data, columns=["type", "message"])
>>> vzr_type = CountVectorizer(analyzer="word", binary=True)
>>> X_type = vzr_type.fit_transform(df["type"])
>>> vzr_msg = CountVectorizer(analyzer="word", binary=True)
>>> X_msg = vzr_msg.fit_transform(df["message"])
>>> bct = BinaryContextTransformer(
...     features=vzr_msg.get_feature_names(),
...     contexts=vzr_type.get_feature_names()
... )
>>> X_msg_type = bct.fit_transform(X_msg, X_type)
>>> print(X_msg_type.todense())
[[1 0 0 1 0 0 1 0 0]
 [0 1 0 0 1 0 0 0 1]
 [0 0 1 0 0 1 0 1 0]]
>>> bct.get_feature_names()
['text_x_anything',
 'tweet_x_anything',
 'email_x_doing',
 'text_x_doing',
 'tweet_x_doing',
 'email_x_text',
 'text_x_text',
 'email_x_tonight',
 'tweet_x_tonight']
```

For an example discussion of using `BinaryContextTransformer` for a classification task, [read this Jupyter notebook](Rare%20Occupation%20Classification.ipynb).

## Acknowledgements

Developed by Vinesh Kannan, Coding It Forward Data Science Fellow at the Bureau of Labor Statistics.

Thank you to Alex Measure, Brandon Kopp, George Stamas, James Walker, Jennifer Edgar, and Mohamed Moulaye.
