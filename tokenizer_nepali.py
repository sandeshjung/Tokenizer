import argparse
import ast
import dill as pickle
from collections import Counter

class Tokenizer:
    """
    A BPE-based tokenizer for Nepali text.

    Attributes:
        input_file (str): Path to the training text file.
        req_tokens (int): Desired final vocabulary size (capped at max_vocab_size).
        verbose (bool): Whether to display merge steps during training.
        words (dict): Base mapping from byte-tuples and merge IDs to string tokens.
        merges (dict): Mapping from byte-pair tuples to new merge IDs.
    """
    def __init__(self, input_file: str, req_tokens: int = 300, max_vocab_size: int = 300, verbose: bool = False):
        """
        Initialize the Tokenizer and train on provided text.

        Args:
            input_file (str): Path to the input Nepali text data.
            req_tokens (int): Desired vocabulary size (will be capped at max_vocab_size).
            max_vocab_size (int): Maximum allowed vocabulary size (default: 300).
            verbose (bool): If True, display merge steps during training.
        """
        self.input_file = input_file
        self.max_vocab_size = max_vocab_size
        self.verbose = verbose
        # Cap requested tokens to maximum
        if req_tokens > self.max_vocab_size:
            print(f"Requested vocab size {req_tokens} exceeds max {self.max_vocab_size}; using {self.max_vocab_size} instead.")
            self.req_tokens = self.max_vocab_size
        else:
            self.req_tokens = req_tokens
        self._initial_token_id = 256
        self._load_data()

    def _load_data(self):
        """
        Load text, initialize base characters, and compute BPE merges.
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.sentence = f.read()

        english_chars = set('abcdefghijklmnopqrstuvwxyz0123456789')
        nepali_chars = {ch for ch in self.sentence if ch.lower() not in english_chars}
        self.words = {tuple(ch.encode('utf-8')): ch for ch in nepali_chars}
        base_vocab = len(self.words)
        self.req_merges = max(self.req_tokens - base_vocab, 0)
        self.next_token_id = self._initial_token_id
        self.tokens = [tuple(ch.encode('utf-8')) for ch in self.sentence]
        self.merges = self._get_merges()

    def _find_common_pair(self, tokens):
        """Return frequency counts of adjacent token pairs."""
        return Counter(zip(tokens, tokens[1:]))

    def _merge(self, tokens, pair, new_id):
        """Merge occurrences of 'pair' in tokens into new_id."""
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _get_merges(self):
        """Perform BPE merges; display steps if verbose."""
        merges = {}
        for _ in range(self.req_merges):
            pair_counts = self._find_common_pair(self.tokens)
            if not pair_counts:
                break
            # Identify most frequent pair
            pair, _ = pair_counts.most_common(1)[0]
            new_id = self.next_token_id
            if self.verbose:
                print(f"merging {pair} into a new token {new_id}")
            self.tokens = self._merge(self.tokens, pair, new_id)
            merges[pair] = new_id
            self.next_token_id += 1
        return merges

    def encode(self, text: str):
        """Encode text to a sequence of token IDs using learned merges."""
        tokens = [tuple(ch.encode('utf-8')) for ch in text]
        while True:
            pair_counts = self._find_common_pair(tokens)
            candidates = [(p, mid) for p, mid in self.merges.items() if p in pair_counts]
            if not candidates:
                break
            pair, _ = min(candidates, key=lambda x: x[1])
            tokens = self._merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids):
        """Decode a list of token IDs back into the original string."""
        for (a, b), idx in self.merges.items():
            if idx not in self.words:
                str_a = self.words.get(a, chr(a) if isinstance(a, int) and a < 256 else '')
                str_b = self.words.get(b, chr(b) if isinstance(b, int) and b < 256 else '')
                self.words[idx] = str_a + str_b
        # Build output string
        result = []
        for token in ids:
            if token in self.words:
                result.append(self.words[token])
            elif isinstance(token, int) and token < 256:
                result.append(chr(token))
            else:
                raise KeyError(f"No mapping found for token {token}")
        return ''.join(result)

    def save(self, filepath: str):
        """Save the tokenizer instance to a dill pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def show_vocab(self):
        """Return a sorted list of (token, representation) for vocabulary."""
        items = []
        for key, val in self.words.items():
            items.append((key, val))
        items.sort(key=lambda x: (isinstance(x[0], tuple), x[0]))
        return items


def main():
    """CLI: train tokenizer, then encode, decode, save, or show vocab."""
    parser = argparse.ArgumentParser(description="Nepali BPE Tokenizer CLI")
    parser.add_argument('-i', '--input-file', required=True,
                        help="Path to input Nepali text file")
    parser.add_argument('-r', '--req-tokens', type=int, default=300,
                        help="Desired vocabulary size (max 300; default: 300)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Display merge steps during training")
    parser.add_argument('--encode', help="Text to encode into token IDs")
    parser.add_argument('--decode', help="Python-style list or tuple of token IDs (e.g. '(256,258),32')")
    parser.add_argument('--show-vocab', action='store_true',
                        help="Display the learned vocabulary mapping")
    parser.add_argument('-s', '--save-file', help="Path to save trained tokenizer pickle")
    args = parser.parse_args()

    max_vocab = 300
    if args.req_tokens > max_vocab:
        print(f"Requested vocab size {args.req_tokens} exceeds max {max_vocab}; using {max_vocab} instead.")
        args.req_tokens = max_vocab

    tokenizer = Tokenizer(
        input_file=args.input_file,
        req_tokens=args.req_tokens,
        max_vocab_size=max_vocab,
        verbose=args.verbose
    )

    print(f"Training complete: {len(tokenizer.merges)} merges applied, final vocabulary size is {len(tokenizer.words)} tokens.")

    # Encode text
    if args.encode:
        encoded = tokenizer.encode(args.encode)
        print("Encoded IDs:", encoded)

    # Decode token list
    if args.decode:
        raw = args.decode.strip()
        if not raw.startswith('['):
            raw = '[' + raw + ']'
        try:
            id_list = ast.literal_eval(raw)
        except Exception as e:
            print(f"Error parsing decode input: {e}")
            return
        decoded = tokenizer.decode(id_list)
        print("Decoded text:", decoded)

    # Show vocabulary
    if args.show_vocab:
        vocab = tokenizer.show_vocab()
        for token, rep in vocab:
            print(f"{token} -> {rep}")

    # Save tokenizer
    if args.save_file:
        tokenizer.save(args.save_file)
        print(f"Tokenizer saved to {args.save_file}")

if __name__ == '__main__':
    main()
