from collections import Counter


class SourceVocabulary:
    """A vocabulary that may be fitted only on source-train text."""

    def __init__(self, minimum_frequency=1):
        self.minimum_frequency = minimum_frequency
        self.tokens = {"[PAD]": 0, "[UNK]": 1}

    def fit(self, source_train_rows):
        counts = Counter(token for row in source_train_rows for token in row["text_a"].split())
        for token, count in sorted(counts.items()):
            if count >= self.minimum_frequency and token not in self.tokens:
                self.tokens[token] = len(self.tokens)
        return self

    def encode(self, text):
        return [self.tokens.get(token, 1) for token in text.split()]
