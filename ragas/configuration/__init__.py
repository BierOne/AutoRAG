from ragas.base import BaseConfiguration


# Sampling and extra parsing
def parse_sample(sample):
    """Parses a sampled configuration, handling prefixes for each method."""
    parsed = {}
    for key, value in sample.items():
        if key.startswith("chroma_"):
            parsed[key[len("chroma_"):]] = value  # Strip the prefix for ModelB parameters
        else:
            parsed[key] = value
    return parsed