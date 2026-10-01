def create_nsp_pairs(documents: list, pair_specs: list) -> list:
    """
    Returns sentence_a, sentence_b, and is_next dictionaries in a list.
    """
    pairs = []

    for spec in pair_specs:
        same_doc = spec["doc_a"] == spec["doc_b"]
        consec = spec["sent_b"] == spec["sent_a"] + 1
        pairs.append({
            "sentence_a": documents[spec["doc_a"]][spec["sent_a"]],
            "sentence_b": documents[spec["doc_b"]][spec["sent_b"]],
            "is_next": int(same_doc and consec)
        })
    
    return pairs