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
        self.id_to_token: dict[int, str] = {
            v: k for k, v in self.token_to_id.items()}

    def _get_word_freqs(self, text_list: list[str]):
        word_freqs = Counter()
        for text in text_list:
            words = text.strip().split()
            word_freqs.update(words)

        return word_freqs

    def _tokenizer_word(self, word: str):
        word_list = list(word)
        return word_list + ["</w>"]

    def _get_pair_freqs(
        self, word_freqs: Counter[str], tokenized_words: dict[str, list[str]]
    ):
        pair_freqs = Counter()

        # 遍历tokenized_words中的每个word的token
        for word, token in tokenized_words.items():
            freq = word_freqs[word]
            # 遍历单个word的token,统计token对的频率
            for i in range(len(token) - 1):
                token_pair = (token[i], token[i + 1])
                pair_freqs[token_pair] += freq

        return pair_freqs

    def _merge_by_pair(
        self, tokenized_words: dict[str, list[str]], pair: tuple[str, str]
    ):
        updated_tokenized_words = {}

        for word, token in tokenized_words.items():
            t = []
            is_merged_last_time = False

            for i in range(len(token) - 1):
                if is_merged_last_time:
                    is_merged_last_time = False
                    continue

                if (token[i], token[i + 1]) == pair:
                    t.append(token[i] + token[i + 1])
                    is_merged_last_time = True
                else:
                    t.append(token[i])

            if not is_merged_last_time:
                t.append(token[-1])

            updated_tokenized_words[word] = t

        return updated_tokenized_words

    def train(self, text_list, target_size):
        # 统计各个单词的数量
        word_freqs = self._get_word_freqs(text_list)

        # 把各单词化为token序列
        tokenized_words = {}
        for word in word_freqs.keys():
            token = self._tokenizer_word(word)
            tokenized_words[word] = token

        while (len(self.token_to_id) < target_size):
            # 根据每个单词的数量和token序列统计token_pair
            pair_freqs = self._get_pair_freqs(word_freqs, tokenized_words)

            # 根据频率最高的token_pair,融合token序列
            most_pair = max(pair_freqs, key=pair_freqs.get)
            tokenized_words = self._merge_by_pair(tokenized_words, most_pair)

            # 更新词表
            self.token_to_id[most_pair[0]+most_pair[1]] = len(self.token_to_id)
