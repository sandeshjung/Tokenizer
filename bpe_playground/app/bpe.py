class BPEProcessor:
    def __init__(self):
        """
        Initializes the BPEProcessor class.

        Args:
        num_merges (int): The number of merge operations to perform during BPE.
        """
        # self.num_merges = num_merges
        self.merges = {}
        self.vocab = {}

    def get_stats(self, ids):
        """Get token pair statistics."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """Merge a pair of tokens in the token list with a new token."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2  # Skip the pair
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def learn_bpe(self, text, num_merges):
        """Perform the BPE algorithm to merge the most frequent pairs."""
        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                # print(f"Warning: No pairs to merge after {i} merges.")
                break 

            pair = max(stats, key=stats.get)
            idx = 256 + i
            # print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

        return ids

    def encode(self, text):
        """Encode the text using learned BPE merges."""
        tokens = list(text.encode("utf-8"))
        
        while True:
            stats = self.get_stats(tokens)
            valid_pairs = [pair for pair in stats if pair in self.merges]

            if not valid_pairs:
                # print("Warning: No valid pairs to merge. Stopping encoding.")
                break

            pair = min(valid_pairs, key=lambda p: self.merges.get(p, float("inf")))
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)

        return tokens

    def decode(self, ids):
        """Decode a list of token IDs back into text."""
        if not self.vocab:
            self.initialize_vocab()

        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def initialize_vocab(self):
        """Initialize vocabulary from the BPE merges."""
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def process(self, text, num_merges):
        """Process the text: learn BPE, encode, and decode."""
        self.learn_bpe(text, num_merges)
        encoded_tokens = self.encode(text)
        decoded_text = self.decode(encoded_tokens)

        vocab_size = len(self.vocab)
        num_merges = len(self.merges)
        return encoded_tokens, decoded_text, vocab_size, num_merges
