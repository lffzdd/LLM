"""
词表管理模块

功能：
- 管理 token 和 id 的双向映射
- 处理特殊 token (PAD, UNK, BOS, EOS)
- 支持词表的保存和加载
"""

import json
from typing import Dict, List, Optional


class Vocabulary:
    """词表类：管理 token <-> id 的映射"""

    # 特殊 token 定义
    PAD_TOKEN = "<PAD>"  # 填充 token
    UNK_TOKEN = "<UNK>"  # 未知 token
    BOS_TOKEN = "<BOS>"  # 句子开始
    EOS_TOKEN = "<EOS>"  # 句子结束

    def __init__(self):
        """初始化词表，预先添加特殊 token"""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # 添加特殊 token，它们占据固定的 id
        self._add_special_tokens()

    def _add_special_tokens(self):
        """添加特殊 token 到词表"""
        special_tokens = [
            self.PAD_TOKEN,  # id = 0
            self.UNK_TOKEN,  # id = 1
            self.BOS_TOKEN,  # id = 2
            self.EOS_TOKEN,  # id = 3
        ]
        for token in special_tokens:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """
        添加一个 token 到词表

        Args:
            token: 要添加的 token
        Returns:
            token 的 id
        """
        if token not in self.token_to_id:
            id = len(self.token_to_id)
            self.token_to_id[token] = id
            self.id_to_token[id] = token
        return self.token_to_id[token]

    def add_tokens(self, tokens: List[str]) -> List[int]:
        """批量添加 token"""
        return [self.add_token(token) for token in tokens]

    def get_id(self, token: str) -> int:
        """
        获取 token 对应的 id

        Args:
            token: 输入 token
        Returns:
            对应的 id，如果不存在则返回 UNK 的 id
        """
        return self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])

    def get_token(self, id: int) -> str:
        """
        获取 id 对应的 token

        Args:
            id: 输入 id
        Returns:
            对应的 token，如果不存在则返回 UNK
        """
        return self.id_to_token.get(id, self.UNK_TOKEN)

    def __len__(self) -> int:
        """返回词表大小"""
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        """检查 token 是否在词表中"""
        return token in self.token_to_id

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    def save(self, path: str):
        """保存词表到文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        print(f"词表已保存到: {path}")

    def load(self, path: str):
        """从文件加载词表"""
        with open(path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        print(f"词表已加载，大小: {len(self)}")

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """从文件创建词表实例"""
        vocab = cls()
        vocab.load(path)
        return vocab


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("词表管理测试")
    print("=" * 60)

    # 创建词表
    vocab = Vocabulary()
    print(f"初始词表大小: {len(vocab)}")
    print(
        f"特殊 token: PAD={vocab.pad_id}, UNK={vocab.unk_id}, BOS={vocab.bos_id}, EOS={vocab.eos_id}"
    )

    # 添加词
    vocab.add_tokens(["hello", "world", "machine", "learning"])
    print(f"\n添加词后词表大小: {len(vocab)}")

    # 查询
    print(f"\nhello 的 id: {vocab.get_id('hello')}")
    print(f"id=5 的 token: {vocab.get_token(5)}")
    print(f"未知词的 id: {vocab.get_id('unknown_word')}")

    # 检查是否存在
    print(f"\n'hello' 在词表中: {'hello' in vocab}")
    print(f"'xyz' 在词表中: {'xyz' in vocab}")

    print("\n词表内容:")
    for token, id in vocab.token_to_id.items():
        print(f"  {id}: '{token}'")
