"""Hidden Markov model decoding boundary."""
def decode(trip: object, candidates: object, probabilities: object) -> list[int]:
    """Decode arc IDs without reading a reference route."""
    del trip, candidates, probabilities
    raise NotImplementedError
