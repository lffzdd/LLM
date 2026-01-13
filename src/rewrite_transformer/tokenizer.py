"""BPE (Byte Pair Encoding) 分词器实现

BPE算法核心思想:
1. 将文本拆分为最小单元(字符)
2. 统计相邻token对出现频率
3. 将最高频的pair合并成新token
4. 重复步骤2-3直到达到目标词表大小

示例:
    输入: "hello world"
    初始: ['h', 'e', 'l', 'l', 'o', '</w>', 'w', 'o', 'r', 'l', 'd', '</w>']
    合并 'l'+'l' -> 'll': ['h', 'e', 'll', 'o', '</w>', 'w', 'o', 'r', 'l', 'd', '</w>']
    合并 'h'+'e' -> 'he': ['he', 'll', 'o', '</w>', 'w', 'o', 'r', 'l', 'd', '</w>']
    ... 继续合并直到达到词表大小
"""

from collections import Counter
import re


class Vocab:
    """词表类，管理token与id的双向映射

    Attributes:
        token_to_id: token -> id 的映射字典
        id_to_token: id -> token 的映射字典

    特殊token:
        <PAD>: 填充token，用于batch对齐
        <UNK>: 未知token，用于处理词表外的词
        <BOS>: 句子开始标记 (Beginning of Sentence)
        <EOS>: 句子结束标记 (End of Sentence)
    """

    # 特殊token定义为类变量，方便全局访问
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self) -> None:
        """初始化词表，预置4个特殊token"""
        # 特殊token的id固定为0-3，保证不同训练结果的一致性
        self.token_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        # 反向映射：通过字典推导式交换key-value
        self.id_to_token = {k: v for v, k in self.token_to_id.items()}

    def add_token(self, token: str):
        """向词表添加新token（如果不存在）

        新token的id = 当前词表大小，保证id连续递增
        """
        if token not in self.token_to_id:
            id = len(self.token_to_id)  # 新id = 当前词表大小
            self.token_to_id[token] = id
            self.id_to_token[id] = token

    def __len__(self):
        """返回词表大小"""
        return len(self.token_to_id)


class BPETokenizer:
    """BPE分词器

    Attributes:
        vocab: 词表对象
        pair_freq: 统计相邻token对的出现频率
        merge_pair: 记录合并历史，用于保存/加载词表

    使用流程:
        1. tokenizer = BPETokenizer()
        2. tokenizer.train(texts, max_vocab_size, max_epoch)
        3. tokenizer.save(path)  # 保存词表
    """

    def __init__(self, vocab: Vocab | None = None) -> None:
        """初始化分词器

        Args:
            vocab: 可选，传入已有词表；否则创建新词表
        """
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocab()

        self.pair_freq = Counter()  # 用于统计相邻pair频率
        self.merge_pair: list[str] = []  # 记录合并顺序，用于后续编码

    def _tokenize_text(
        self,
        text: str,
        eow: str = "</w>",
        bos: str = Vocab.BOS_TOKEN,
        eos: str = Vocab.EOS_TOKEN,
    ) -> list[str]:
        """把文本拆分成单词

        Args:
            texts: 文本列表,每个文本是一个字符串
        """
        # 句首加标识
        tokens: list[str] = [bos]

        # 遍历['hello','world']
        for word in text.strip().split():  # 'hello world'->['hello','world']
            # extend会把'hello'迭代加到s，即[...]->[...,'h','e','l','l','o']
            tokens.extend(word)
            tokens.append(eow)

        # [...]->[...,['h','e','l','l','o','</w>','w','o','r','l','d','</w>']]
        tokens.append(eos)
        return tokens

    def _update_pair_freq(self, text: list[str]):
        """统计文本中相邻token对的出现频率

        Args:
            text: token列表，如 ['h', 'e', 'l', 'l', 'o', '</w>']

        工作原理:
            遍历相邻pair: (h,e), (e,l), (l,l), (l,o), (o,</w>)
            每个pair计数+1
        """
        length = len(text)
        if length == 1:  # 只有一个token时无法形成pair
            return

        # 遍历所有相邻位置，统计pair
        for i in range(length - 1):
            pair = text[i] + text[i + 1]  # 拼接相邻两个token
            self.pair_freq.update([pair])  # Counter.update需要传入可迭代对象

    def _update_merge_pair(self, pair_freq: Counter | None = None):
        """记录本轮合并的pair到merge_pair列表

        merge_pair列表记录了合并的顺序，用于:
        1. 保存词表时记录BPE合并规则
        2. 加载词表后按相同顺序编码新文本
        """
        if pair_freq is None:
            pair_freq = self.pair_freq
        # most_common(1) 返回 [(pair, count)]，取第一个元素
        most_pair, freq = pair_freq.most_common(1)[0]
        self.merge_pair.append(most_pair)

    def _merge_text(self, text: list[str], pair: str):
        """根据pair对文本进行merge

        Args:
            text: token列表，如 ['h', 'e', 'l', 'l', 'o']
            pair: 要合并的pair，如 'he'

        Returns:
            合并后的列表，如 ['he', 'l', 'l', 'o']
        """
        merged_text = []

        i = 0
        length = len(text)
        while i < length - 1:
            if text[i] + text[i + 1] == pair:
                merged_text.append(text[i] + text[i + 1])
                i += 2  # 跳过已合并得两个token，避免重复处理
            else:
                merged_text.append(text[i])
                i += 1

        # 如果最后一个没添加，添加最后一个
        if i != length:
            merged_text.append(text[i])
        return merged_text

    def train(self, texts: list[str], max_vocab_size: int, max_epoch: int):
        """训练BPE分词器

        Args:
            texts: 训练文本列表，如 ["hello world", "how are you"]
            max_vocab_size: 目标词表大小上限
            max_epoch: 最大迭代轮数（每轮合并一个pair）

        训练流程:
            1. 文本 -> 字符级token列表
            2. 初始化词表（所有字符）
            3. 循环: 统计pair频率 -> 合并最高频pair -> 更新词表
        """
        # ===== 阶段1: 文本预处理 =====
        # 将每个文本拆分成字符级token列表
        texts_tokens: list[list[str]] = []
        for text in texts:
            texts_tokens.append(self._tokenize_text(text))

        print("初始token化文本：\n", texts_tokens, "\n")

        # ===== 阶段2: 初始化词表 =====
        # 将所有出现的字符加入词表
        for text_tokens in texts_tokens:
            for token in text_tokens:
                self.vocab.add_token(token)

        # ===== 阶段3: 迭代合并 =====
        # 核心循环: 每轮找到最高频pair并合并
        epoch = 0
        while len(self.vocab) <= max_vocab_size and epoch <= max_epoch:
            # 3. 计算pair出现频率
            for text_tokens in texts_tokens:
                self._update_pair_freq(text_tokens)

            print(f"第{epoch}轮：\n", self.pair_freq, "\n")

            if len(self.pair_freq) == len(texts_tokens):
                print("所有text已merge完毕，结束训练")
                break

            # 4. 根据频率更新merge_pair,这里merger_pair没有参与更新，仅仅是用于后续能保存词表
            self._update_merge_pair()

            # 4. 根据当前频率最多的pair更新text
            most_pair, _ = self.pair_freq.most_common(1)[0]
            for i in range(len(texts_tokens)):
                texts_tokens[i] = self._merge_text(texts_tokens[i], most_pair)

            # 4. 新pair纳入词表
            self.vocab.add_token(most_pair)

            # 5. 清空pair为下一轮做准备
            self.pair_freq.clear()

            epoch += 1

        print(self.vocab.token_to_id)

    def save(self, path: str):
        """保存词表到文件

        TODO: 实现保存逻辑，需要保存:
            1. vocab.token_to_id - token与id的映射
            2. merge_pair - 合并顺序（用于编码新文本）
        """
        pass


if __name__ == "__main__":
    texts = ["hello world", "how are you"]

    tokenizer = BPETokenizer()
    tokenizer.train(texts, 100, 100)
