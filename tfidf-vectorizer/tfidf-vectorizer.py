import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    # Write code here
    tokenized = [document.lower().split() for document in documents]
    vocab = sorted({token for tokens in tokenized for token in tokens})
    ix = {token: position for position, token in enumerate(vocab)}
    m = np.zeros((len(documents), len(vocab)), dtype=float)
    df = Counter()

    for tokens in tokenized:
        df.update(set(tokens))

    for row, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for token, count in counts.items():
            tf = count / len(tokens)
            idf = math.log(len(documents) / df[token])
            m[row, ix[token]] = tf * idf

    return {
        "tfidf_matrix": m,
        "vocabulary": vocab
    }