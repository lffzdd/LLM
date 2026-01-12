import torch
from pathlib import Path

from my_transformer._1_tokenizer import BPETokenizer
from my_transformer._7_transformer import Transformer

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("checkpoints")

# ==================== 加载 Tokenizer ====================
src_tokenizer = BPETokenizer()
tgt_tokenizer = BPETokenizer()

src_tokenizer.load(SAVE_DIR / "src_tokenizer.json")
tgt_tokenizer.load(SAVE_DIR / "tgt_tokenizer.json")

print(
    f"Loaded tokenizers: src_vocab={len(src_tokenizer.vocab)}, tgt_vocab={len(tgt_tokenizer.vocab)}"
)

# ==================== 加载模型 ====================
# 注意：torch.load 加载的是 checkpoint dict，不是模型本身
checkpoint = torch.load(SAVE_DIR / "best_model.pt", map_location=DEVICE)

# 需要先创建模型，然后加载权重
vocab_size = max(len(src_tokenizer.vocab), len(tgt_tokenizer.vocab))
model = Transformer(
    vocab_size=vocab_size,
    embed_dim=256,  # 需要和训练时一致！
    max_seq_len=128,
    encoder_layer_num=4,
    decoder_layer_num=4,
    attn_head_num=8,
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print(
    f"Loaded model from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}"
)

# ==================== 推理：使用 generate 方法 ====================
# 推理时不需要 tgt！只需要 src，模型会自动生成


def translate(text: str) -> str:
    """翻译单个句子"""
    # 1. 编码源文本
    src_ids = src_tokenizer.encode(text)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

    # 2. 使用 generate 方法自回归生成
    # 不需要传入 tgt！模型会从 BOS 开始自动生成
    generated = model.generate(
        src_seq=src_tensor,
        bos_id=2,  # <BOS> token id
        eos_id=3,  # <EOS> token id
        max_len=50,
    )

    # 3. 解码生成的 token ids
    generated_ids = generated[0].tolist()  # 取 batch 中的第一个

    # 移除 BOS 和 EOS
    if generated_ids[0] == 2:  # 移除开头的 BOS
        generated_ids = generated_ids[1:]
    if 3 in generated_ids:  # 移除 EOS 及之后的内容
        eos_idx = generated_ids.index(3)
        generated_ids = generated_ids[:eos_idx]

    result = tgt_tokenizer.decode(generated_ids)
    return result


# ==================== 测试翻译 ====================
test_sentences = [
    "hello world",
    "how are you",
    "good morning",
    "thank you",
]

print("\n" + "=" * 60)
print("Translation Results:")
print("=" * 60)

for sentence in test_sentences:
    translation = translate(sentence)
    print(f"  EN: {sentence}")
    print(f"  ZH: {translation}")
    print()
