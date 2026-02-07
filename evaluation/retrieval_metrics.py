def context_precision(retrieved, relevant):
    return round(len(set(retrieved) & set(relevant)) / max(len(retrieved), 1), 2)
