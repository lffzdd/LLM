from my_transformer._7_transformer import Transformer
from my_transformer._8_train import Trainer, TranslationDataset, collate_fn
from my_transformer._1_tokenizer import BPETokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 创建简单的测试数据
src_texts = ["hello world", "how are you"]
tgt_texts = ["你好世界", "你好吗"]

# 创建tokenizer
src_tokenizer = BPETokenizer()
tgt_tokenizer = BPETokenizer()
src_tokenizer.train(src_texts, target_size=100, min_frequency=1)
tgt_tokenizer.train(tgt_texts, target_size=100, min_frequency=1)

vocab_size = max(len(src_tokenizer.vocab), len(tgt_tokenizer.vocab)) + 10
print(f"Tokenizer vocab size: {len(src_tokenizer.vocab)}, {len(tgt_tokenizer.vocab)}")

# 创建模型
model = Transformer(
    vocab_size=vocab_size,
    embed_dim=64,
    max_seq_len=128,
    encoder_layer_num=2,
    decoder_layer_num=2,
    attn_head_num=4,
)

# 创建训练器
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)
trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

# 创建dataset和dataloader
dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# 训练一个epoch
loss = trainer.train_epoch(dataloader)
print(f"Train OK: loss = {loss:.4f}")
