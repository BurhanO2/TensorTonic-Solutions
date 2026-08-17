def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    sums = {}
    counts = {}

    n = len(categories)

    for i in range(n):
        if categories[i] not in counts:
            counts[categories[i]] = 0
        if categories[i] not in sums:
            sums[categories[i]] = 0

        counts[categories[i]] += 1
        sums[categories[i]] += targets[i]

    return [float(sums[c] / counts[c]) for c in categories]
