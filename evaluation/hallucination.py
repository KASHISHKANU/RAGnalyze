def hallucination_stats(citations):
    """
    citations = output of sentence_citations()
    """

    total = len(citations)
    unsupported = sum(1 for c in citations if len(c["citations"]) == 0)

    if total == 0:
        return 0.0

    return round((unsupported / total) * 100, 2)
