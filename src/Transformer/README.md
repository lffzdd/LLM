# Transformer 从零实现

从零手写实现 Transformer 模型，用于学习和理解其核心原理。

## 项目结构

```
src/Transformer/
├── tokenizer.py           # BPE 分词器
├── vocab.py               # 词表管理
├── embedding.py           # Token Embedding
├── positional_encoding.py # 位置编码 (Sin/Cos)
├── attention.py           # 多头注意力机制
├── encoder.py             # Transformer 编码器
├── decoder.py             # Transformer 解码器
├── transformer.py         # 完整 Transformer 模型
└── train.py               # 训练脚本
```

## 数据流

```
"我爱机器学习"
    ↓ (1) Tokenizer 分词
["我", "爱", "机器", "学习"]
    ↓ (2) Vocab 词表映射
[23, 156, 892, 1024]
    ↓ (3) Token Embedding
[[0.1, 0.2, ...], ...]  (d_model 维向量)
    ↓ (4) + Positional Encoding
[[...], [...], ...]  (加上位置信息)
    ↓ (5) Encoder / Decoder
    ↓
输出
```

## 模块说明

### 1. 分词器 (tokenizer.py)

使用 **BPE (Byte Pair Encoding)** 算法，GPT 系列模型的标准分词方式。

```python
from tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.train(texts, target_vocab_size=1000)

ids = tokenizer.encode("hello world")    # [2, 156, 892, 3]
text = tokenizer.decode(ids)              # "hello world"
```

### 2. 词表 (vocab.py)

管理 token 和 id 的双向映射，包含特殊 token：
- `<PAD>` (0): 填充
- `<UNK>` (1): 未知词
- `<BOS>` (2): 句子开始
- `<EOS>` (3): 句子结束

### 3. Embedding (embedding.py)

将 token id 映射为稠密向量：

```python
# 本质是查表操作
self.w[x]  # x=[1,2,3] -> 取出第1,2,3行的向量
```

### 4. 位置编码 (positional_encoding.py)

使用 sin/cos 函数生成位置信息：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 5. 注意力机制 (attention.py)

核心公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

包含：
- `ScaledDotProductAttention`: 缩放点积注意力
- `MultiHeadAttention`: 多头注意力
- `create_causal_mask`: 因果掩码（解码器用）
- `create_padding_mask`: 填充掩码

### 6. 编码器 (encoder.py)

```
输入 → Self-Attention → Add & Norm → FFN → Add & Norm → 输出
       ↑________________|            ↑_____|
           残差连接                   残差连接
```

### 7. 解码器 (decoder.py)

比编码器多一层交叉注意力：

```
输入 → Masked Self-Attn → Add & Norm → Cross-Attn → Add & Norm → FFN → Add & Norm
       ↑_________________|             ↑___________|             ↑_____|
```

### 8. 完整模型 (transformer.py)

```python
from transformer import Transformer

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    embed_size=512,
    num_heads=8,
    ff_hidden_size=2048,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# 训练
output = model(src, tgt)  # [batch, tgt_len, vocab_size]

# 推理
generated = model.generate(src, max_len=50)
```

## 快速开始

```bash
# 测试各模块
uv run python tokenizer.py
uv run python attention.py
uv run python encoder.py
uv run python decoder.py
uv run python transformer.py
uv run python train.py
```

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化讲解
