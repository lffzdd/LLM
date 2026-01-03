import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        """
        Args:
            vocab_size: 词典大小
            embed_size: 词嵌入维度
        """
        super().__init__()
        # 手写实现词嵌入层
        # 权重矩阵 shape: [vocab_size, embed_size]
        # 每一行代表一个词的 embedding 向量
        # 例如: vocab_size=10000, embed_size=512
        # 则 w 的形状为 [10000, 512]
        self.w = nn.Parameter(torch.randn(vocab_size, embed_size))

    def forward(self, x):
        """
        Args:
            x: token ids, shape=[batch_size, seq_len] 或 [seq_len]
        Returns:
            embeddings: shape=[batch_size, seq_len, embed_size] 或 [seq_len, embed_size]

        原理:
        - 不使用 one-hot + 矩阵乘法(效率低)
        - 直接用索引查找(等价但高效)
        - self.w[x] 会自动广播,取出对应行的 embedding
        """
        return self.w[x]


if __name__ == "__main__":
    # 测试代码
    vocab_size = 10
    embed_size = 4

    # 创建 embedding 层
    embedding = TokenEmbedding(vocab_size, embed_size)
    print(f"权重矩阵形状: {embedding.w.shape}")  # [10, 4]
    print(f"权重矩阵:\n{embedding.w.data}\n")

    # 测试1: 单个序列
    tokens = torch.tensor([1, 2, 3])
    output = embedding(tokens)
    print(f"输入 token ids: {tokens}")
    print(f"输出形状: {output.shape}")  # [3, 4]
    print(f"输出:\n{output}\n")

    # 测试2: batch 输入
    batch_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    batch_output = embedding(batch_tokens)
    print(f"Batch 输入形状: {batch_tokens.shape}")  # [2, 3]
    print(f"Batch 输出形状: {batch_output.shape}")  # [2, 3, 4]
    print(f"Batch 输出:\n{batch_output}")

