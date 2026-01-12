from collections import Counter


class Vocab:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self) -> None:
        self.token_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        self.id_to_token = {k: v for v, k in self.token_to_id.items()}

    def add_token(self, token: str):
        if token not in self.token_to_id:
            id = len(self.token_to_id)
            self.token_to_id[token] = id
            self.id_to_token[id] = token


class BPETokenizer:
    def __init__(self, vocab: Vocab | None) -> None:
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocab()

    def _sperate_texts(texts: list[str]):
        """把文本拆分成单词

        Args:
            texts: 文本列表,每个文本是一个字符串
        """
