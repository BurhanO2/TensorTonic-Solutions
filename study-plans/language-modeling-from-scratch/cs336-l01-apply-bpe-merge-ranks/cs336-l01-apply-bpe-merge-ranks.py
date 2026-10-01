def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns token IDs after applying the ordered merge rules.
    """
    ids = list(text.encode('utf-8'))

    for (left_id, right_id, new_id) in merges:
        if len(ids) < 2:
            continue

        merged = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == left_id and (ids[i + 1] == right_id):
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1

        ids = merged

    return ids


def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns the text reconstructed from the token bytes.
    """
    raw = []
    for token_id in ids:
        raw.extend(vocab[token_id])
    return bytes(raw).decode('utf-8')
