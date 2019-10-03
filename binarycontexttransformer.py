import numpy as np
import scipy as sp
from sklearn.base import TransformerMixin
from scipy.sparse import csc_matrix, csr_matrix


class BinaryContextTransformer(TransformerMixin):
    """
    Expands base features into interaction terms when they appear with
    different context features. Both base features and context features
    must be binary.
    """

    def __init__(self, features, contexts, progress=None):
        """
        Args:
            features: names of base features
            contexts: names of context features
            progress: function of format progress_fn(iter, total) that takes
                an iterable and an integer with the total number of items and
                returns a generator to track progress at each step of the
                iterable (default=None)
        """
        self.features = features
        self.contexts = contexts
        self.col_pairs = []
        self.progress = progress
        self.vocabulary = {}

    def fit(self, X, X_context):
        """
        Args:
            X: matrix of base feature columns
            X_context: matrix of context feature columns
        """
        assert X.shape[1] == len(self.features), "X not same size as base."
        assert X_context.shape[1] == len(
            self.contexts
        ), "X_context not same size as context."
        if not isinstance(X, csc_matrix):
            X = csc_matrix(X)
        if not isinstance(X_context, csc_matrix):
            X_context = csc_matrix(X_context)
        looper = range(X_context.shape[1])
        if self.progress is not None:
            looper = self.progress(looper, total=X_context.shape[1])
        # Find possible interactions from the sparse input matrix.
        blocks = []
        # If each record appears in only one context, the runtime complexity
        # of this loop is O(S), where S = the number of entries in the sparse
        # matrix. Each row will be selected only once and the call to max()
        # for a sparse matrix will only consider nonzero entries in the row.
        # For sparse matrices, N < S << NB.
        for i in looper:
            # Get row indices of records that match context i
            row_list = X_context[:, i].indices
            if len(row_list) > 0:
                # Squash rows into binary mask for each feature
                # 1 if feature and context co-occur, 0 otherwise
                row_vals = X[row_list, :].max(axis=0)
                blocks.append(row_vals)
        # The variable `S` is a matrix where each row is a context and each
        # column is a feature, nonzero entries are possible interactions.
        S = sp.sparse.vstack(blocks)
        # Get column indices of features that occur in at least 2 contexts
        feature_idxs = csr_matrix(S.sum(axis=0) - 1).indices
        S = csc_matrix(S)
        # Make vocabulary
        col_pairs = []
        vocab = {}
        k = 0
        # The runtime complexity of this loop is O(V), where V is the number
        # of interaction terms in the resulting vocabulary. In the worst case,
        # when every feature appears in every context, V = BC. When interactions
        # are sparse, V << BC.
        looper = feature_idxs
        if self.progress is not None:
            looper = self.progress(looper, total=len(feature_idxs))
        for j in looper:
            context_idcs = S[:, j].indices
            for i in context_idcs:
                col_pairs.append((i, j))
                feature_name = self.features[j]
                context_name = self.contexts[i]
                name = context_name + "_x_" + feature_name
                vocab[name] = k
                k += 1
        self.col_pairs = col_pairs
        self.vocabulary = vocab
        # Check that vocabulary is correct size, sizes will not match
        # if features or contexts contain duplicate feature names.
        # This may occur when joining multiple vocabularies to form
        # the base feature names.
        msg_len = (
            "Length of `vocab` does not match `col_pairs`. ",
            "Check for duplicate feature names.",
        )
        assert len(col_pairs) == len(vocab), msg_len
        return self

    def transform(self, X, X_context):
        """
        Args:
            X: matrix of base feature columns
            X_context: matrix of context feature columns
        """
        assert X.shape[1] == len(self.features), "X not same size as base."
        assert X_context.shape[1] == len(
            self.contexts
        ), "X_context not same size as context."
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)
        if not isinstance(X_context, csr_matrix):
            X_context = csr_matrix(X_context)
        n = X.shape[0]
        m = len(self.col_pairs)
        data = []
        cols = []
        rows = []
        val = 1

        # The runtime complexity of this loop is O(V). See `fit` method
        # for notes on V, the size of the fitted vocabulary.
        col_pair_map = {}
        for k, (i, j) in enumerate(self.col_pairs):
            col_pair_map[(i, j)] = k
        looper = range(n)
        if self.progress is not None:
            looper = self.progress(looper, total=n)
        # If each record appears in only one context, the runtime complexity
        # of this loop is O(S) where S is the number of entries in the sparse
        # matrix. See `fit` method for notes on S.
        for r in looper:
            contexts = X_context[r, :].indices
            features = X[r, :].indices
            for i in contexts:
                for j in features:
                    pair = (i, j)
                    if pair in col_pair_map:
                        k = col_pair_map[pair]
                        data.append(val)
                        rows.append(r)
                        cols.append(k)

        mat = csc_matrix((data, (rows, cols)), shape=(n, m), dtype=np.int8)
        return mat

    def fit_transform(self, X, X_context):
        """
        Args:
            X: matrix of base feature columns
            X_context: matrix of context feature columns
        """
        assert X.shape[1] == len(self.features), "X not same size as base."
        assert X_context.shape[1] == len(
            self.contexts
        ), "X_context not same size as context."
        self.fit(X, X_context)
        return self.transform(X, X_context)

    def get_feature_names(self):
        """
        Returns a list of feature names corresponding to column indices.
        """
        vocab = sorted(self.vocabulary.items(), key=lambda p: p[1])
        return [name for name, i in vocab]
