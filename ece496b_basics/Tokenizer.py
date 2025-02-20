import regex as re
import pickle
from typing import List, Dict, Tuple, Iterable


class Tokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer that encodes and decodes text.
    """

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        """
        Initialize the tokenizer with a vocabulary and merges.

        Args:
            vocab (Dict[int, bytes]): A mapping from token ID to byte sequences.
            merges (List[Tuple[bytes, bytes]]): A list of byte pair merges.
            special_tokens (List[str], optional): Special tokens to be handled separately.
        """
        self.vocab = vocab
        self.inverse_vocab = {token: idx for idx, token in vocab.items()}
        self.merges = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens else []
        
        # GPT-2 style tokenization regex
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: List[str] = None) -> "Tokenizer":
        """
        Load a pre-trained tokenizer from saved vocabulary and merge files.

        Args:
            vocab_path (str): Path to the saved vocabulary file.
            merges_path (str): Path to the saved merges file.
            special_tokens (List[str], optional): Special tokens for encoding.

        Returns:
            Tokenizer: A tokenizer instance.
        """
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            text (str): The input text.

        Returns:
            List[int]: A list of token IDs.
        """
        # Handle special tokens
        if self.special_tokens:
            special_pattern = re.compile("(" + "|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True))) + ")")
            special_chunks = re.split(special_pattern, text)
        else:
            special_chunks = [text]

        token_ids = []

        for chunk in special_chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.inverse_vocab[chunk.encode("utf-8")])
            else:
                token_ids.extend(self._bpe_encode(chunk))

        return token_ids

    def _bpe_encode(self, text: str) -> List[int]:
        """
        Apply Byte Pair Encoding (BPE) on the input text.

        Args:
            text (str): The input text.

        Returns:
            List[int]: A list of token IDs after applying BPE.
        """
        tokens = re.findall(self.pattern, text)
        token_ids = []

        for token in tokens:
            token_bytes = tuple(bytes([b]) for b in token.encode("utf-8"))

            while len(token_bytes) > 1:
                pairs = list(zip(token_bytes, token_bytes[1:]))
                best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))

                if best_pair not in self.merges:
                    break

                idx = pairs.index(best_pair)
                token_bytes = token_bytes[:idx] + (best_pair[0] + best_pair[1],) + token_bytes[idx + 2:]

            token_ids.extend([self.inverse_vocab[tok] for tok in token_bytes])

        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Encodes an iterable of strings, yielding token IDs one at a time.

        Args:
            iterable (Iterable[str]): An iterable of text strings.

        Yields:
            int: A single token ID at a time.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            token_ids (List[int]): The list of token IDs.

        Returns:
            str: The reconstructed text.
        """
        return b"".join(self.vocab[i] for i in token_ids).decode("utf-8", errors="replace")

    def save(self, path: str, prefix: str = "", overwrite: bool = False):
        """
        Save the tokenizer state to disk.

        Args:
            path (str): Directory where the tokenizer should be saved.
            prefix (str, optional): Prefix for the saved files.
            overwrite (bool, optional): If True, overwrite existing files.
        """
        import os
        os.makedirs(path, exist_ok=True)

        vocab_path = os.path.join(path, prefix + "vocab.pkl")
        merges_path = os.path.join(path, prefix + "merges.pkl")

        if not overwrite and (os.path.exists(vocab_path) or os.path.exists(merges_path)):
            raise ValueError("Files already exist. Use `overwrite=True` to overwrite.")

        with open(vocab_path, "wb") as f:
            pickle.dump(self.vocab, f)
        with open(merges_path, "wb") as f:
            pickle.dump(self.merges, f)

        print(f"Tokenizer saved to {path}")