def wordpiece_tokenize(text: str, vocab: dict,
                       unk_token: str = "[UNK]", max_word_length: int = 100) -> list:
    """
    Returns the WordPiece tokens as a list of strings.
    """
    tokens = []
    for word in text.lower().split():
        if len(word) > max_word_length:
            tokens.append(unk_token)
            continue

        pieces = []
        start = 0

        while start < len(word):
            match = None
            end = len(word)

            while end > start:
                cand = word[start:end]
                if start > 0:
                    cand = "##" + cand
                if cand in vocab:
                    match = cand
                    break
                end -= 1

            if match is None:
                pieces = [unk_token]
                break
            
            pieces.append(match)
            start = end
        tokens.extend(pieces)

    return tokens