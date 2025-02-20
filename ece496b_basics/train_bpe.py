import regex as re
import heapq
import logging
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

# Set up logging
logger = logging.getLogger(__name__)

# GPT-2 pre-tokenization regex pattern
GPT2_TOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Helper func for train_bpe
def compute_pretoken_frequencies(file_path: str, special_tokens: List[str]) -> Dict[Tuple[bytes], int]:
    """
    Reads a text file, removes special tokens, and computes pre-token frequencies.
    Returns the text file represented as a pretoken frequency table
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    for token in special_tokens:
        text = text.replace(token, "")

    pretoken_counts = Counter(re.findall(GPT2_TOKENIZER_PATTERN, text, concurrent=True))
    del text  # Free memory

    return {
        tuple(bytes([b]) for b in token.encode("utf-8")): freq
        for token, freq in pretoken_counts.items()
    }

# Helper func for train_bpe
def compute_pair_frequencies(pretoken_frequencies: Dict[Tuple[bytes], int]) -> Tuple[Dict[Tuple[bytes, bytes], int], List[Tuple[int, Tuple[bytes, bytes]]]]:
    """Computes the frequency of adjacent byte pairs."""
    pair_frequencies = Counter()
    for token_seq, count in pretoken_frequencies.items():
        for pair in zip(token_seq, token_seq[1:]):
            pair_frequencies[pair] += count

    heap = [(-freq, pair) for pair, freq in pair_frequencies.items()]
    heapq.heapify(heap)

    return pair_frequencies, heap

# Helper func for train_bpe
def map_bytes_to_pretokens(pretoken_frequencies: Dict[Tuple[bytes], int]) -> Dict[bytes, set]:
    """Creates a mapping from bytes to pre-token sequences."""
    byte_to_token_map = defaultdict(set)
    for token_seq in pretoken_frequencies:
        for byte in token_seq:
            byte_to_token_map[byte].add(token_seq)

    return byte_to_token_map

# Helper func for train_bpe
def merge_token_sequence(token_seq: Tuple[bytes], merge_pair: Tuple[bytes, bytes], index: int) -> Tuple[Tuple[bytes], Tuple[bytes], Tuple[bytes]]:
    """Merges a pair of bytes within a token sequence."""
    prefix = token_seq[:index]
    suffix = token_seq[index + 2:]
    merged_seq = prefix + (b"".join(merge_pair),) + suffix
    return merged_seq, prefix, suffix

# Main function to train BPE
def train_bpe(file_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a Byte Pair Encoding (BPE) tokenizer.

    Args:
        file_path (str): Path to the input text file.
        vocab_size (int): The desired vocabulary size.
        special_tokens (List[str]): List of special tokens to include.

    Returns:
        Tuple:
            - Dict[int, bytes]: The final vocabulary mapping token IDs to byte sequences.
            - List[Tuple[bytes, bytes]]: A list of merged byte pairs.
    """
    # Initialize vocabulary with basic bytes and special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # Compute initial pre-token frequencies
    pretoken_frequencies = compute_pretoken_frequencies(file_path, special_tokens)

    # Build auxiliary data structures
    byte_to_token_map = map_bytes_to_pretokens(pretoken_frequencies)
    pair_frequencies, merge_heap = compute_pair_frequencies(pretoken_frequencies)

    merges = []

    for _ in tqdm(range(vocab_size - len(vocab))):
        if not merge_heap:
            break

        max_pair = None
        max_freq = None
        repush_pairs = []

        while merge_heap:
            freq, pair = heapq.heappop(merge_heap)

            # Ignore outdated frequencies
            if pair_frequencies[pair] != -freq:
                continue

            if max_pair is None:
                max_pair, max_freq = pair, freq
                continue

            if freq != max_freq:
                heapq.heappush(merge_heap, (freq, pair))
                break

            # Restore previous behavior
            if pair > max_pair:
                repush_pairs.append(max_pair)
                max_pair = pair
            else:
                repush_pairs.append(pair)

        for pair in repush_pairs:
            heapq.heappush(merge_heap, (max_freq, pair))

        if max_pair is None:
            break

        # Update vocabulary
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)

        changed_keys = set()

        # Identify affected pre-token sequences
        affected_tokens = byte_to_token_map[max_pair[0]] & byte_to_token_map[max_pair[1]]

        for token_seq in affected_tokens:
            if token_seq not in pretoken_frequencies:
                continue

            token_count = pretoken_frequencies[token_seq]
            del pretoken_frequencies[token_seq]

            i = 0
            while i < len(token_seq) - 1:
                pair = token_seq[i:i + 2]
                if pair == max_pair:
                    token_seq, prefix, suffix = merge_token_sequence(token_seq, max_pair, i)

                    # Restore original heap update logic
                    if prefix:
                        add_pair = (prefix[-1], vocab[len(vocab) - 1])
                        del_pair = (prefix[-1], max_pair[0])
                        pair_frequencies[add_pair] += token_count
                        pair_frequencies[del_pair] -= token_count
                        changed_keys.update([add_pair, del_pair])

                    if suffix:
                        add_pair = (vocab[len(vocab) - 1], suffix[0])
                        del_pair = (max_pair[1], suffix[0])
                        pair_frequencies[add_pair] += token_count
                        pair_frequencies[del_pair] -= token_count
                        changed_keys.update([add_pair, del_pair])

                i += 1

            pretoken_frequencies[token_seq] = token_count

            for byte in token_seq:
                byte_to_token_map[byte].add(token_seq)

        del pair_frequencies[max_pair]

        # Restore original push-back behavior
        for key in changed_keys:
            heapq.heappush(merge_heap, (-pair_frequencies[key], key))

    return vocab, merges
