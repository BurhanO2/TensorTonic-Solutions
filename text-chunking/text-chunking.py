def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    # Write code here
    step = chunk_size - overlap
    n = len(tokens)
    chunks = []
    for i in range(0, n, step):
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        if i >= n - chunk_size:
            break
    return chunks