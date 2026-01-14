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

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def add_token(self, token: str):
        if token not in self.token_to_id:
            id = len(self.token_to_id)
            self.token_to_id[token] = id
            self.id_to_token[id] = token

    def __len__(self):
        return len(self.token_to_id)


class BPETokenizer:
    def __init__(self, vocab: Vocab | None = None) -> None:
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocab()

        self.pair_freq = Counter()

    def _tokenize_text(
        self,
        text: str,
        eow: str = "</w>",
        bos: str = Vocab.BOS_TOKEN,
        eos: str = Vocab.EOS_TOKEN,
    ):
        text_tokens = [bos]
        for word in text.strip().split():
            text_tokens.extend(
                word
            )  # 这里不是word作为整体加入列表而是分成一个个字母加入，是[bos,'h','e','l','l','o','</w>','w','o'...]而不是[bos,'hello</w>','world</w>']
            text_tokens.append(eow)
        text_tokens.append(eos)

        return text_tokens

    def _update_pair_freq(self, text_tokens: list[str]):
        for i in range(len(text_tokens) - 1):
            self.pair_freq.update([text_tokens[i] + text_tokens[i + 1]])

    def _merge_text(self, text_tokens: list[str], pair):
        merged_text = []
        i = 0
        length = len(text_tokens)

        while i < length - 1:
            if text_tokens[i] + text_tokens[i + 1] == pair:
                merged_text.append(text_tokens[i] + text_tokens[i + 1])
                i += 2
            else:
                merged_text.append(text_tokens[i])
                i += 1

        if i != length:  # 如果最后一步没被merge,那上述循环只循环到了倒数第二个token
            merged_text.append(text_tokens[length - 1])

        return merged_text

    def train(self, texts: list[str], max_vocab_size: int, num_epoch: int):
        texts_tokens: list[list[str]] = []
        for text in texts:
            tokenized_text = self._tokenize_text(text)
            texts_tokens.append(tokenized_text)

            for token in tokenized_text:
                self.vocab.add_token(token)

        epoch = 0
        while len(self.vocab) <= max_vocab_size and epoch <= num_epoch:
            for text_tokens in texts_tokens:
                self._update_pair_freq(text_tokens)

            if (
                len(self.pair_freq) == 0
            ):  # 词对数和句子数一样，说明每个句子都被merge完毕了
                print("所有句子都被merge成了一个token,训练完毕")
                break

            most_pair, freq = self.pair_freq.most_common(1)[0]
            self.vocab.add_token(most_pair)

            texts_tokens = list(
                map(lambda t: self._merge_text(t, most_pair), texts_tokens)
            )

            epoch += 1
            print(
                f"第{epoch}轮:\n \
                检测出pair：{most_pair}\n \
                得到texts_tokens:\n{texts_tokens}\n \
                词表为{self.vocab.token_to_id}\n",
            )

            self.pair_freq.clear()


if __name__ == "__main__":
    tokenizer = BPETokenizer()

    texts = ["hello world", "how are you"]

    tokenizer.train(texts, 100, num_epoch=100)
