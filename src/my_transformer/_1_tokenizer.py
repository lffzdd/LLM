from __future__ import annotations  # Python 3.7+: 延迟类型注解求值

from collections import Counter
import re
from utils.logger import get_logger

logger = get_logger(__name__)


class BPETokenizer:
    def __init__(self, vocab: Vocab = None):
        """
        初始化 BPE Tokenizer

        Args:
            vocab: 可选的 Vocab 实例，如果不提供则自动创建新的 Vocab

        Example:
            # 方式 1: 自动创建 vocab
            tokenizer = BPETokenizer()

            # 方式 2: 使用自定义 vocab（高级用法）
            custom_vocab = Vocab()
            tokenizer = BPETokenizer(custom_vocab)
        """
        self.vocab = vocab if vocab is not None else Vocab()
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
            if char not in self.vocab:
                self.vocab.add_tokens(char)

        while len(self.vocab) < target_size:
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
            self.vocab.add_tokens(most_pair[0] + most_pair[1])

    def encode(self, text: str) -> list[int]:
        words = text.strip().split()
        text_tokens_id: list[int] = []

        # 对每一个单词进行pair合并
        for word in words:
            word_token = self._tokenizer_word(word)

            # 用到每个pair规则
            for i in range(len(self.merge_pairs)):
                # 从前到后遍历token序列
                word_token = self._merge_by_pair(
                    {word: word_token}, self.merge_pairs[i]
                )
                word_token = word_token[word]  # _merge_by_pair返回的是dict

            # 转换为id序列并直接添加到结果中
            for t in word_token:
                text_tokens_id.append(self.vocab.get_id(t))

        return text_tokens_id

    def decode(self, text_tokens_id: list[int]) -> str:
        # 转换成整个句子的token序列
        text_tokens = []
        for id in text_tokens_id:
            token = self.vocab.get_token(id)
            text_tokens.append(token)

        # 将所有 token 拼接，然后用 </w> 分割成单词
        text = "".join(text_tokens)
        # </w> 标记单词边界，替换为空格
        text = text.replace("</w>", " ")
        return text.strip()

    def save(self, path):
        import json
        from pathlib import Path

        data = {"token_to_id": self.vocab.token_to_id, "merge_pairs": self.merge_pairs}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        abs_path = Path(path).resolve()
        logger.info(f"词表已保存到: {abs_path}")

    def load(self, path):
        import json

        data = {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab.token_to_id = data["token_to_id"]
        self.vocab.id_to_token = {int(v): k for k, v in self.vocab.token_to_id.items()}
        merge_pairs = data["merge_pairs"]
        self.merge_pairs = [tuple(k) for k in merge_pairs]
        logger.info(f"词表已从 {path} 加载")

    # ========== 工厂方法 ==========

    @classmethod
    def from_file(cls, path: str):
        """
        工厂方法：从文件加载已训练的 tokenizer

        Args:
            path: tokenizer 文件路径

        Returns:
            加载好的 BPETokenizer 实例

        Example:
            tokenizer = BPETokenizer.from_file("tokenizer.json")
        """
        tokenizer = cls()  # 自动创建 Vocab
        tokenizer.load(path)
        return tokenizer

    @classmethod
    def from_pretrained(
        cls, texts: list[str], target_size: int = 1000, min_frequency: int = 2
    ):
        """
        工厂方法：直接训练并返回 tokenizer

        Args:
            texts: 训练文本列表
            target_size: 目标词表大小
            min_frequency: 最小频率阈值

        Returns:
            训练好的 BPETokenizer 实例

        Example:
            tokenizer = BPETokenizer.from_pretrained(
                texts=["hello world", "machine learning"],
                target_size=100,
                min_frequency=2
            )
        """
        tokenizer = cls()  # 自动创建 Vocab
        tokenizer.train(texts, target_size, min_frequency)
        return tokenizer


class Vocab:
    """
    词表类：负责 token <-> id 的映射管理

    职责：
    - 管理 token 到 id 的双向映射
    - 提供 token 的增删查功能
    - 处理特殊 token (PAD, UNK, BOS, EOS)

    注意：
    - 此类只负责内存中的词表管理
    - 持久化（save/load）由 BPETokenizer 负责
    - 这样设计遵循单一职责原则

    Example:
        vocab = Vocab()
        vocab.add_tokens(["hello", "world"])
        id = vocab.get_id("hello")
        token = vocab.get_token(id)
    """

    def __init__(self):
        self.token_to_id = {
            "<PAD>": 0,  # 填充
            "<UNK>": 1,  # 未知
            "<BOS>": 2,  # 句子开始
            "<EOS>": 3,  # 句子结束
        }

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def add_tokens(self, tokens: list[str] | str):
        if isinstance(tokens, str):
            tokens = [tokens]

        next_id = len(self.token_to_id)
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

    def get_id(self, token: str):
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])

    def get_token(self, id: int):
        return self.id_to_token.get(id, "<UNK>")

    def __len__(self):
        return len(self.token_to_id)

    def __contains__(self, token: str):
        return token in self.token_to_id


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

    print("=" * 60)
    print("方式 0: 最简单的用法 - 直接创建")
    print("=" * 60)

    # ✨ 最简单：不需要任何参数！
    simple_tokenizer = BPETokenizer()
    print(f"创建成功！词表大小: {len(simple_tokenizer.vocab)}")
    print("可以直接调用 train() 方法进行训练\n")

    print("=" * 60)
    print("方式 1: 使用 from_pretrained 直接训练")
    print("=" * 60)

    # ✨ 使用工厂方法：一步完成训练
    tokenizer = BPETokenizer.from_pretrained(
        texts=training_texts, target_size=50, min_frequency=1
    )

    # 输出词表
    print("\nToken to ID mapping:")
    for token, id in tokenizer.vocab.token_to_id.items():
        print(f"{token}: {id}")

    # 测试编码和解码
    sample_text = "the cat loves machine learning"
    encoded = tokenizer.encode(sample_text)
    print(f"\nEncoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # 保存
    tokenizer.save("bpe_tokenizer.json")

    print("\n" + "=" * 60)
    print("方式 2: 使用 from_file 加载已保存的 tokenizer")
    print("=" * 60)

    # ✨ 使用工厂方法：一步完成加载
    new_tokenizer = BPETokenizer.from_file("bpe_tokenizer.json")

    # 加载保存的词表
    print("\nLoaded Token to ID mapping:")
    for token, id in new_tokenizer.vocab.token_to_id.items():
        print(f"{token}: {id}")

    # 测试加载后的编码是否一致
    new_encoded = new_tokenizer.encode(sample_text)
    print(f"\nNew Encoded: {new_encoded}")
    assert encoded == new_encoded, "编码结果不一致！"
    print("✅ 编码结果一致！")

    print("\n" + "=" * 60)
    print("方式 3: 手动训练（传统方式）")
    print("=" * 60)

    # ✨ 直接创建，然后手动训练
    manual_tokenizer = BPETokenizer()
    manual_tokenizer.train(training_texts, target_size=50, min_frequency=1)
    print(f"训练完成！词表大小: {len(manual_tokenizer.vocab)}")
