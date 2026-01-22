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
import json
import time
from rewrite_transformer.util import get_logger

logger = get_logger(__name__)


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

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

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
            idx = len(self.token_to_id)  # 新id = 当前词表大小
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def get_token(self, token_id: int) -> str:
        """根据ID获取token，未知ID返回UNK_TOKEN"""
        if token_id not in self.id_to_token:
            return self.UNK_TOKEN
        return self.id_to_token[token_id]

    def get_id(self, token: str) -> int:
        """根据token获取ID，未知token返回UNK的ID"""
        if token not in self.token_to_id:
            return self.token_to_id[self.UNK_TOKEN]
        return self.token_to_id[token]

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

    # BPE特有的词边界标记，用于区分 "hello" 和 "hell"+"o"
    EOW_TOKEN = "</w>"

    def __init__(self, vocab: Vocab | None = None) -> None:
        """初始化分词器

        Args:
            vocab: 可选，传入已有词表；否则创建新词表
        """
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocab()

        # 训练时临时使用，统计相邻pair频率
        self.pair_freq = Counter()

        # 记录合并顺序，用于后续编码和保存
        self.merge_pair: list[str] = []

        # merge_pair的字典版本，用于encode时快速查找合并优先级
        # key: pair, value: 合并顺序（越小优先级越高）
        self._merge_pair_rank: dict[str, int] = {}

    def _tokenize_text(
        self,
        text: str,
        bos: str = Vocab.BOS_TOKEN,
        eos: str = Vocab.EOS_TOKEN,
        add_bos_eos: bool = False,
    ) -> list[str]:
        """将文本拆分成字符级token列表

        Args:
            text: 输入文本，如 "hello world"
            bos: 句子开始标记
            eos: 句子结束标记
            add_bos_eos: 是否添加BOS/EOS标记

        Returns:
            字符级token列表，如 ['h', 'e', 'l', 'l', 'o', '</w>', 'w', 'o', 'r', 'l', 'd', '</w>']
        """
        tokens: list[str] = []

        # 可选：添加句首标记
        if add_bos_eos:
            tokens.append(bos)

        # 按空格分词，每个词拆成字符 + 词尾标记
        for word in text.strip().split():
            tokens.extend(word)  # 'hello' -> ['h', 'e', 'l', 'l', 'o']
            tokens.append(self.EOW_TOKEN)  # 添加词边界标记

        # 可选：添加句尾标记
        if add_bos_eos:
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

    def _update_merge_pair(self, pair: str):
        """记录本轮合并的pair到merge_pair列表

        merge_pair列表记录了合并的顺序，用于:
        1. 保存词表时记录BPE合并规则
        2. 加载词表后按相同顺序编码新文本
        """
        self.merge_pair.append(pair)

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
        if i == length - 1:
            merged_text.append(text[-1])
        return merged_text

    def _update_merge_pair_rank(self):
        """根据merge_pair构建rank字典，用于encode时快速查找合并优先级"""
        self._merge_pair_rank = {
            pair: rank for rank, pair in enumerate(self.merge_pair)
        }

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

        logger.info("初始token化文本完毕")

        # ===== 阶段2: 初始化词表 =====
        # 将所有出现的字符加入词表
        for text_tokens in texts_tokens:
            for token in text_tokens:
                self.vocab.add_token(token)

        # ===== 阶段3: 迭代合并 =====
        # 核心循环: 每轮找到最高频pair并合并
        logger.info("开始训练")

        epoch = 0
        while len(self.vocab) < max_vocab_size and epoch <= max_epoch:
            # 3. 计算pair出现频率
            for text_tokens in texts_tokens:
                self._update_pair_freq(text_tokens)

            start_time = time.perf_counter()

            logger.info(f"进入第{epoch}轮,开始时间：{start_time}")

            if len(self.pair_freq) == 0:
                logger.info("所有text已merge完毕")
                break

            # 4. 获取最高频pair
            most_pair, _ = self.pair_freq.most_common(1)[0]

            # 5. 记录合并历史（用于后续encode和保存）
            self._update_merge_pair(most_pair)

            # 6. 执行合并：将所有文本中的该pair合并
            texts_tokens = list(
                map(lambda t: self._merge_text(t, most_pair), texts_tokens)
            )

            # 7. 新pair加入词表
            self.vocab.add_token(most_pair)

            # 8. 清空pair频率统计，为下一轮做准备
            self.pair_freq.clear()

            epoch += 1

            end_time = time.perf_counter()
            logger.info(f"第{epoch}轮训练完毕，耗时{end_time - start_time}秒")

        self._update_merge_pair_rank()

        logger.info("训练完毕")

    def save(self, path: str):
        """保存tokenizer到JSON文件

        保存内容:
            - token_to_id: token与id的映射
            - merge_pair: 合并顺序（encode时按此顺序合并）
        """
        data = {"token_to_id": self.vocab.token_to_id, "merge_pair": self.merge_pair}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"训练结果保存到{path}")

    def load(self, path: str):
        """从JSON文件加载tokenizer

        Args:
            path: tokenizer文件路径
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 恢复词表的双向映射
        self.vocab.token_to_id = data["token_to_id"]
        self.vocab.id_to_token = {
            token_id: token for token, token_id in self.vocab.token_to_id.items()
        }

        # 恢复合并历史和rank缓存
        self.merge_pair = data["merge_pair"]
        self._update_merge_pair_rank()

        logger.info(f"已从{path}加载词表")

    def encode(
        self,
        texts: list[str] | str,
        add_special_tokens: bool = False,
        padding: bool = False,
        padded_len: int | None = None,
        truncation: bool = False,
    ) -> list[int] | list[list[int]]:
        """将文本编码为token ID列表

        Args:
            text: 输入文本
            add_special_tokens: 是否添加BOS/EOS标记
            padding: 是否进行padding
            padded_len: padding后的长度，如果不指定，则为所有文本中最长序列的长度
            truncation: 若padding后的长度超过max_len，则截断到max_len

        Returns:
            token ID列表
        """
        if isinstance(texts, str):
            texts = [texts]

        texts_ids: list[list[int]] = []

        max_len = 0  # 记录最长序列长度
        for text in texts:
            # 1. 文本 -> 字符级token列表
            text_tokens = self._tokenize_text(text, add_bos_eos=add_special_tokens)

            # 2. 按训练时的合并顺序，依次合并存在的pair
            while True:
                # 找到当前文本中存在的、rank最小的pair
                min_rank = float("inf")
                best_pair = None

                for i in range(len(text_tokens) - 1):
                    pair = text_tokens[i] + text_tokens[i + 1]
                    if (
                        pair in self._merge_pair_rank
                        and self._merge_pair_rank[pair] < min_rank
                    ):
                        best_pair = pair
                        min_rank = self._merge_pair_rank[pair]

                # 没有可合并的pair，结束
                if not best_pair:
                    break

                text_tokens = self._merge_text(text_tokens, best_pair)

            # 3. token -> ID
            text_ids = list(map(self.vocab.get_id, text_tokens))

            texts_ids.append(text_ids)
            if not padded_len and len(text_ids) > max_len:
                max_len = len(text_ids)

        # 统一 padding（在所有文本编码完成后）
        if padded_len:
            max_len = padded_len

        if (padding or padded_len) and len(texts_ids) > 1:
            pad_id = self.vocab.get_id(Vocab.PAD_TOKEN)
            for i, ids in enumerate(texts_ids):
                pad_cnt = max_len - len(ids)
                if pad_cnt < 0:
                    if truncation:
                        del ids[max_len:]
                    else:
                        raise ValueError(
                            f"序列 {i} 的长度 ({len(ids)}) 超过了 padded_len ({max_len})"
                        )
                if pad_cnt > 0:
                    ids.extend([pad_id] * pad_cnt)

        if isinstance(texts, str):
            return texts_ids[0]

        return texts_ids

    def decode(self, text_ids: list[int] | list[list[int]]) -> str | list[str]:
        """将token ID列表解码为文本

        Args:
            text_ids: token ID列表

        Returns:
            解码后的文本
        """
        # 需要过滤掉的特殊token ID
        special_ids = {
            self.vocab.get_id(Vocab.BOS_TOKEN),
            self.vocab.get_id(Vocab.EOS_TOKEN),
        }

        if not isinstance(text_ids[0], list):
            text_ids = [text_ids]

        texts: list[str] = []
        for text_id in text_ids:
            # ID -> token，同时过滤特殊token
            text_tokens = [
                self.vocab.get_token(token_id)
                for token_id in text_id
                if token_id not in special_ids
            ]

            texts.append("".join(text_tokens))

        # 拼接并将词边界标记还原为空格,strip去除首尾空格
        texts = ["".join(text).replace(self.EOW_TOKEN, " ").strip() for text in texts]

        return texts

    @property
    def vocab_size(self):
        return len(self.vocab)
