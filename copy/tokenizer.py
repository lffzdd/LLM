from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.special_tokens: dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        self.token_to_id: dict[str, int] = dict(self.special_tokens)
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}

    def _get_word_freqs(self, texts: list[str]):
        word_freqs = Counter()
        for text in texts:
            words = text.strip().split()
            word_freqs.update(words)

        return word_freqs

    def _tokenizer_word(self, word: str):
        word_list = list(word)
        return word_list + ["</w>"]

    def _get_pair_freqs(self, word_freqs, tokenized_words):
        word_pairs=
