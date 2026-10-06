def canonical_preference_pairs(records: list) -> dict:
    """
    Returns a dict with pairs: a list of prompt_id, winner_id, loser_id dictionaries.
    """
    pairs = []
    for record in records:
        ordered = sorted(record["candidates"], key=lambda candidate: candidate["rank"])
        for winner_index in range(len(ordered)):
            for loser_index in range(winner_index + 1, len(ordered)):
                pairs.append({
                    "prompt_id": record["prompt_id"],
                    "winner_id": ordered[winner_index]["id"],
                    "loser_id": ordered[loser_index]["id"],
                })

    return {
        "pairs": pairs
    }
