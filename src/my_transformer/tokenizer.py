from collections import Counter
from utils.logger import get_logger

logger = get_logger(__name__)


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

        self.merge_pairs: list[tuple[str, str]] = []

    def _get_word_freqs(self, text_list: list[str]) -> Counter[str]:
        word_freqs = Counter()
        for text in text_list:
            words = text.strip().split()
            word_freqs.update(words)

        return word_freqs

    def _tokenizer_word(self, word: str) -> list[str]:
        word_list = list(word)
        return word_list + ["</w>"]

    def _get_pair_freqs(
        self, word_freqs: Counter[str], tokenized_words: dict[str, list[str]]
    ) -> Counter[tuple[str, str]]:
        pair_freqs = Counter()

        # 遍历tokenized_words中的每个word的token
        for word, word_tokens in tokenized_words.items():
            freq = word_freqs[word]
            # 遍历单个word的token,统计token对的频率
            for i in range(len(word_tokens) - 1):
                token_pair = (word_tokens[i], word_tokens[i + 1])
                pair_freqs[token_pair] += freq

        return pair_freqs

    def _merge_by_pair(
        self, tokenized_words: dict[str, list[str]], pair: tuple[str, str]
    ) -> dict[str, list[str]]:
        updated_tokenized_words: dict[str, list[str]] = {}

        for word, word_token in tokenized_words.items():
            t = []
            is_merged_last_time = False

            for i in range(len(word_token) - 1):
                if is_merged_last_time:
                    is_merged_last_time = False
                    continue

                if (word_token[i], word_token[i + 1]) == pair:
                    t.append(word_token[i] + word_token[i + 1])
                    is_merged_last_time = True
                else:
                    t.append(word_token[i])

            if not is_merged_last_time:
                t.append(word_token[-1])

            updated_tokenized_words[word] = t

        return updated_tokenized_words

    def train(self, text_list: list[str], target_size: int, min_frequency: int) -> None:
        # 统计各个单词的数量
        word_freqs = self._get_word_freqs(text_list)

        # 把各单词化为token序列
        tokenized_words: dict[str, list[str]] = {}
        for word in word_freqs.keys():
            token = self._tokenizer_word(word)
            tokenized_words[word] = token

        # 统计单字符频率
        char_freqs = Counter()
        for word, work_tokens in tokenized_words.items():
            freq = word_freqs[word]
            for t in work_tokens:
                char_freqs[t] += freq

        # 只保留频率达标的单字符
        for char, freq in char_freqs.items():
            if freq < min_frequency:
                continue
            if char not in self.token_to_id.keys():
                self.token_to_id[char] = len(self.token_to_id)

        while len(self.token_to_id) < target_size:
            # 根据每个单词的数量和token序列统计token_pair
            pair_freqs = self._get_pair_freqs(word_freqs, tokenized_words)
            # 如果没有token_pair,则结束训练
            if not pair_freqs:
                logger.info("No more token pairs to merge.")
                break

            # 根据频率最高的token_pair,融合token序列
            most_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[most_pair] < min_frequency:
                break
            tokenized_words = self._merge_by_pair(tokenized_words, most_pair)

            # 更新规则
            self.merge_pairs.append(most_pair)

            # 更新词表
            self.token_to_id[most_pair[0] + most_pair[1]] = len(self.token_to_id)

        # 更新id_to_token
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text: str) -> list[int]:
        words = text.strip().split()
        text_tokens_id: list[int] = []

        # 对每一个单词进行pair合并
        for word in words:
            word_tokens = self._tokenizer_word(word)

            # 用到每个pair规则
            for i in range(len(self.merge_pairs)):
                # 从前到后遍历token序列
                word_tokens = self._merge_by_pair(
                    {word: word_tokens}, self.merge_pairs[i]
                )
                word_tokens = word_tokens[word]  # _merge_by_pair返回的是dict

            # 转换为id序列并直接添加到结果中
            for t in word_tokens:
                text_tokens_id.append(
                    self.token_to_id.get(t, self.special_tokens["<UNK>"])
                )

        return text_tokens_id

    def decode(self, text_tokens_id: list[int]) -> str:
        # 转换成整个句子的token序列
        text_tokens = []
        for id in text_tokens_id:
            token = self.id_to_token[id]
            text_tokens.append(token)

        # 将所有 token 拼接，然后用 </w> 分割成单词
        text = "".join(text_tokens)
        # </w> 标记单词边界，替换为空格
        text = text.replace("</w>", " ")
        return text.strip()

    def save(self, path):
        import json
        from pathlib import Path

        data = {"token_to_id": self.token_to_id, "merge_pairs": self.merge_pairs}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        abs_path = Path(path).resolve()
        logger.info(f"词表已保存到: {abs_path}")

    def load(self, path):
        import json

        data = {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        merge_pairs = data["merge_pairs"]
        self.merge_pairs = [tuple(k) for k in merge_pairs]
        logger.info(f"词表已从 {path} 加载")


if __name__ == "__main__":
    # 准备一些训练数据
    training_texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat and the dog played together",
        "I love machine learning",
        "machine learning is amazing",
        "deep learning is a subset of machine learning",
        "natural language processing uses deep learning",
        "the quick brown fox jumps over the lazy dog",
        "hello world",
        "hello there",
        "world of programming",
    ]

    # 创建并训练分词器
    tokenizer = BPETokenizer()
    tokenizer.train(training_texts, target_size=50, min_frequency=1)

    # 输出词表
    print("Token to ID mapping:")
    for token, id in tokenizer.token_to_id.items():
        print(f"{token}: {id}")

    # 测试编码和解码
    sample_text = "the cat loves machine learning"
    encoded = tokenizer.encode(sample_text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # 保存和加载词表
    tokenizer.save("bpe_tokenizer.json")
    new_tokenizer = BPETokenizer()

    new_tokenizer.load("bpe_tokenizer.json")

    # 加载保存的词表
    print("Loaded Token to ID mapping:")
    for token, id in new_tokenizer.token_to_id.items():
        print(f"{token}: {id}")

    # 测试加载后的编码是否一致
    new_encoded = new_tokenizer.encode(sample_text)
    print(f"New Encoded: {new_encoded}")
    assert encoded == new_encoded

