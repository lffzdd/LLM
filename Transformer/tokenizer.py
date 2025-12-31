"""
BPE (Byte Pair Encoding) 分词器

BPE 是目前主流的子词分词算法，被 GPT 系列模型使用。

核心思想：
1. 初始时，把每个字符作为一个 token
2. 统计相邻 token 对的出现频率
3. 把出现最频繁的 token 对合并成新 token
4. 重复步骤 2-3，直到达到目标词表大小

优点：
- 平衡了字符级和词级的优缺点
- 常见词保持完整，罕见词被拆成子词
- 可以处理任何语言，包括未见过的词
"""

import re
from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self):
        # 特殊 token
        self.special_tokens = {
            "<PAD>": 0,  # 填充 token
            "<UNK>": 1,  # 未知 token
            "<BOS>": 2,  # Begin of Sentence
            "<EOS>": 3,  # End of Sentence
        }

        # 词表：token -> id
        self.token_to_id = dict(self.special_tokens)
        # 反向映射：id -> token
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # BPE 合并规则：(token1, token2) -> merged_token
        # 按学习顺序存储，编码时按此顺序应用
        self.merges = {}

        # 词表大小
        self.vocab_size = len(self.special_tokens)

    def _get_word_freqs(self, texts):
        """
        统计语料中每个词的频率

        Args:
            texts: 文本列表
        Returns:
            word_freqs: {word: frequency}

        例如: ["hello world", "hello"] -> {"hello": 2, "world": 1}
        """
        word_freqs = Counter()
        for text in texts:
            # 简单按空格分词，你也可以用更复杂的预分词
            words = text.strip().split()
            word_freqs.update(words)
        return word_freqs

    def _tokenize_word(self, word):
        """
        把一个词拆成字符序列，末尾加 </w> 标记词边界

        例如: "hello" -> ["h", "e", "l", "l", "o", "</w>"]

        </w> 的作用是区分词内和词尾：
        - "low" 作为独立词: ["l", "o", "w", "</w>"]
        - "lower" 中的 "low": ["l", "o", "w", "e", "r", "</w>"]
        """
        return list(word) + ["</w>"]

    def _get_pair_freqs(self, word_splits, word_freqs):
        """
        统计所有相邻 token 对的频率

        Args:
            word_splits: {word: [tokens]}  每个词当前的分词结果
            word_freqs: {word: freq}  每个词的出现频率
        Returns:
            pair_freqs: {(token1, token2): frequency}
        """
        pair_freqs = Counter()
        for word, tokens in word_splits.items():
            freq = word_freqs[word]
            # 遍历相邻的 token 对
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, word_splits, pair):
        """
        在所有词中，把指定的 token 对合并成新 token

        Args:
            word_splits: {word: [tokens]}
            pair: (token1, token2) 要合并的 token 对
        Returns:
            更新后的 word_splits
        """
        new_token = pair[0] + pair[1]
        new_splits = {}

        for word, tokens in word_splits.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 如果当前位置匹配到要合并的 pair
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(new_token)
                    i += 2  # 跳过两个 token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_splits[word] = new_tokens

        return new_splits

    def train(self, texts, target_vocab_size=1000, min_frequency=2, verbose=True):
        """
        在语料上训练 BPE 词表

        Args:
            texts: 训练文本列表
            target_vocab_size: 目标词表大小
            min_frequency: 最小合并频率阈值
            verbose: 是否打印训练过程

        训练过程：
        1. 统计词频
        2. 初始化：每个词拆成字符
        3. 循环：找最频繁的 pair -> 合并 -> 加入词表
        4. 直到达到目标词表大小
        """
        if verbose:
            print(f"开始训练 BPE，目标词表大小: {target_vocab_size}")

        # Step 1: 统计词频
        word_freqs = self._get_word_freqs(texts)
        if verbose:
            print(f"语料中不同的词数: {len(word_freqs)}")

        # Step 2: 初始化，把每个词拆成字符
        word_splits = {word: self._tokenize_word(word) for word in word_freqs}

        # 收集所有初始字符作为基础词表
        all_chars = set()
        for tokens in word_splits.values():
            all_chars.update(tokens)

        # 把基础字符加入词表
        for char in sorted(all_chars):
            if char not in self.token_to_id:
                self.token_to_id[char] = len(self.token_to_id)
                self.id_to_token[self.token_to_id[char]] = char

        self.vocab_size = len(self.token_to_id)
        if verbose:
            print(f"基础字符数: {len(all_chars)}，当前词表大小: {self.vocab_size}")

        # Step 3: 迭代合并
        merge_count = 0
        while self.vocab_size < target_vocab_size:
            # 统计 pair 频率
            pair_freqs = self._get_pair_freqs(word_splits, word_freqs)

            if not pair_freqs:
                if verbose:
                    print("没有更多可合并的 pair，停止训练")
                break

            # 找出频率最高的 pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            # 如果最高频率低于阈值，停止
            if best_freq < min_frequency:
                if verbose:
                    print(f"最高频率 {best_freq} 低于阈值 {min_frequency}，停止训练")
                break

            # 合并这个 pair
            word_splits = self._merge_pair(word_splits, best_pair)

            # 把合并后的新 token 加入词表
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = len(self.token_to_id)
                self.id_to_token[self.token_to_id[new_token]] = new_token
                self.vocab_size = len(self.token_to_id)

            # 记录合并规则
            self.merges[best_pair] = new_token
            merge_count += 1

            if verbose and merge_count % 100 == 0:
                print(
                    f"  合并 {merge_count}: {best_pair} -> {new_token} (频率: {best_freq})"
                )

        if verbose:
            print(
                f"训练完成！总合并次数: {merge_count}，最终词表大小: {self.vocab_size}"
            )

    def _tokenize_word_with_merges(self, word):
        """
        使用学到的 merge 规则对单个词进行分词

        Args:
            word: 要分词的词
        Returns:
            tokens: 分词结果列表
        """
        # 先拆成字符
        tokens = self._tokenize_word(word)

        # 按 merge 规则的学习顺序，依次应用合并
        for pair, merged in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text, add_special_tokens=True):
        """
        把文本编码成 token id 序列

        Args:
            text: 输入文本
            add_special_tokens: 是否添加 <BOS> 和 <EOS>
        Returns:
            token_ids: id 列表

        例如: "hello world" -> [2, 156, 892, 234, 3]  (带 BOS/EOS)
        """
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.token_to_id["<BOS>"])

        # 按空格分词，然后对每个词应用 BPE
        words = text.strip().split()
        for word in words:
            tokens = self._tokenize_word_with_merges(word)
            for token in tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    token_ids.append(self.token_to_id["<UNK>"])

        if add_special_tokens:
            token_ids.append(self.token_to_id["<EOS>"])

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """
        把 token id 序列解码回文本

        Args:
            token_ids: id 列表
            skip_special_tokens: 是否跳过特殊 token
        Returns:
            text: 解码后的文本
        """
        tokens = []
        for id in token_ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append("<UNK>")

        # 把 tokens 拼接成文本
        # </w> 标记词的结束，需要替换成空格
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

    def get_vocab(self):
        """返回词表"""
        return dict(self.token_to_id)

    def save(self, path):
        """保存词表和 merge 规则到文件"""
        import json

        data = {
            "token_to_id": self.token_to_id,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"词表已保存到: {path}")

    def load(self, path):
        """从文件加载词表和 merge 规则"""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.merges = {tuple(k.split("|||")): v for k, v in data["merges"].items()}
        self.vocab_size = len(self.token_to_id)
        print(f"词表已加载，大小: {self.vocab_size}")


# ========== 测试代码 ==========
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
    print("BPE 分词器训练示例")
    print("=" * 60)

    # 创建并训练分词器
    tokenizer = BPETokenizer()
    tokenizer.train(
        training_texts, target_vocab_size=100, min_frequency=2, verbose=True
    )

    print("\n" + "=" * 60)
    print("编码/解码测试")
    print("=" * 60)

    # 测试编码和解码
    test_texts = [
        "the cat",
        "machine learning",
        "hello world",
        "unknown words here",  # 测试未见过的词
    ]

    for text in test_texts:
        print(f"\n原文: '{text}'")

        # 编码
        token_ids = tokenizer.encode(text)
        print(f"Token IDs: {token_ids}")

        # 显示每个 token
        tokens = [tokenizer.id_to_token.get(id, "<?>") for id in token_ids]
        print(f"Tokens: {tokens}")

        # 解码
        decoded = tokenizer.decode(token_ids)
        print(f"解码: '{decoded}'")

    print("\n" + "=" * 60)
    print("词表前 30 个 token")
    print("=" * 60)
    vocab = tokenizer.get_vocab()
    for i, (token, id) in enumerate(list(vocab.items())[:30]):
        print(f"  {id:3d}: '{token}'")
