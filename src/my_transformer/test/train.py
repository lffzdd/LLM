import time
from pathlib import Path

from datasets import load_dataset
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, random_split
from my_transformer._1_tokenizer import BPETokenizer
from my_transformer._7_transformer import Transformer
from my_transformer._8_train import TranslationDataset, collate_fn

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

# 训练超参数（针对 RTX 3090 24GB 优化）
NUM_EPOCHS = 100
BATCH_SIZE = 64  # 3090 可以跑更大的 batch
LEARNING_RATE = 0.0001  # 大 batch 通常用小一点的学习率
MAX_SEQ_LEN = 128
DATA_SIZE = 20000  # 减小数据量以加快 BPE 训练（之后可以增加）

# 模型参数
EMBED_DIM = 256  # 增大嵌入维度
NUM_HEADS = 8
NUM_LAYERS = 4  # 增加层数

# Tokenizer 参数
VOCAB_SIZE = 2000  # 减小词表以加快 BPE 训练
MIN_FREQ = 1  # 提高最小频率过滤更多低频词

# 早停配置
PATIENCE = 10
MIN_DELTA = 0.001

# 混合精度训练（加速 + 省显存）
USE_AMP = True

print(f"Using device: {DEVICE}")

# ==================== 数据准备 ====================
src_tokenizer = BPETokenizer()
tgt_tokenizer = BPETokenizer()

if __name__ == "__main__":
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")

    # 获取训练数据子集（使用 select 方法）
    train_subset = dataset["train"].select(range(min(DATA_SIZE, len(dataset["train"]))))

    # 正确的访问方式
    src_texts = [item["en"] for item in train_subset["translation"]]
    tgt_texts = [item["zh"] for item in train_subset["translation"]]

    print(f"Loading {len(src_texts)} samples...")
    print(
        f"Training tokenizers (vocab_size={VOCAB_SIZE}, this may take a few minutes   )..."
    )

    # src_tokenizer.train(src_texts, VOCAB_SIZE, MIN_FREQ)
    # tgt_tokenizer.train(tgt_texts, VOCAB_SIZE, MIN_FREQ)

    # src_tokenizer.save(SAVE_DIR / "src_tokenizer.json")
    # tgt_tokenizer.save(SAVE_DIR / "tgt_tokenizer.json")

    src_tokenizer.load(SAVE_DIR / "src_tokenizer.json")
    tgt_tokenizer.load(SAVE_DIR / "tgt_tokenizer.json")

    # ==================== 创建数据集 ====================
    full_dataset = TranslationDataset(
        src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len=MAX_SEQ_LEN
    )

    # 划分训练集和验证集 (90% 训练, 10% 验证)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows 上多进程有问题，设为 0
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ==================== 创建模型 ====================
    vocab_size = max(len(src_tokenizer.vocab), len(tgt_tokenizer.vocab))
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        max_seq_len=MAX_SEQ_LEN,
        encoder_layer_num=NUM_LAYERS,
        decoder_layer_num=NUM_LAYERS,
        attn_head_num=NUM_HEADS,
    )
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 混合精度训练
    scaler = torch.amp.GradScaler() if USE_AMP else None

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"AMP enabled: {USE_AMP}")

    # ==================== 验证函数 ====================
    def validate(model, val_loader, criterion, device):
        """在验证集上评估模型"""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_label = tgt[:, 1:]

                output = model(src, tgt_input)
                loss = criterion(
                    output.reshape(-1, output.size(-1)), tgt_label.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(val_loader)

    # ==================== 早停类 ====================
    class EarlyStopping:
        """早停机制：当验证 loss 不再改善时停止训练"""

        def __init__(self, patience: int = 10, min_delta: float = 0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float("inf")
            self.counter = 0
            self.should_stop = False

        def __call__(self, val_loss: float) -> bool:
            if val_loss < self.best_loss - self.min_delta:
                # loss 有改善
                self.best_loss = val_loss
                self.counter = 0
                return False  # 保存模型的信号
            else:
                # loss 没有改善
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
                return True  # 不需要保存

    # ==================== 训练循环 ====================
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        start_time = time.time()

        for src, tgt in train_loader:
            src = src.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            optimizer.zero_grad()

            # 混合精度训练
            if USE_AMP:
                with torch.amp.autocast(device_type="cuda"):
                    output = model(src, tgt_input)
                    loss = criterion(
                        output.reshape(-1, output.size(-1)), tgt_label.reshape(-1)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(src, tgt_input)
                loss = criterion(
                    output.reshape(-1, output.size(-1)), tgt_label.reshape(-1)
                )
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        val_loss = validate(model, val_loader, criterion, DEVICE)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Time:    {elapsed:.2f}s"
        )

        # 早停检查
        should_skip_save = early_stopping(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, SAVE_DIR / "best_model.pt")
            print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = SAVE_DIR / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"  ✓ Checkpoint saved to {checkpoint_path}")

        # 检查是否需要早停
        if early_stopping.should_stop:
            print(
                f"\n⚠️ Early stopping triggered! No improvement for {PATIENCE} epochs."
            )
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

    # 保存最终模型
    torch.save(model.state_dict(), SAVE_DIR / "model_final.pt")
    print(f"\nTraining complete!")
    print(
        f"  - Best model: {SAVE_DIR / 'best_model.pt'} (val_loss: {best_val_loss:.4f})"
    )
    print(f"  - Final model: {SAVE_DIR / 'model_final.pt'}")
