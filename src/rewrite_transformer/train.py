import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rewrite_transformer.dataset import TransformerDataset, collate_fn
from rewrite_transformer.tokenizer import BPETokenizer, Vocab
from rewrite_transformer.transformer import Transformer
from rewrite_transformer.util import load_dataset

from rewrite_transformer.util import get_logger

logger = get_logger(__name__)


def trainMyTokenizer(texts_path, save_path):
    tokenizer = BPETokenizer()
    tokenizer.train(texts_path, 32000, 32000)
    tokenizer.save(save_path)
    logger.info(f"My BPE tokenizer saved to {save_path}")


def loadMyTokenizer(tokenizer_path):
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer


# 使用官方tokenizer
def trainOfficialTokenizer(texts_path, save_path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=32000, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )
    tokenizer.train(files=[texts_path], trainer=trainer)
    tokenizer.save(save_path)
    logger.info(f"Official BPE tokenizer saved to {save_path}")


def loadOfficialTokenizer(tokenizer_path):
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing

    tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> <BOS> $B <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )
    return tokenizer


def trainTransformer(
    bpe_src_path,
    bpe_tgt_path,
    data_path,
    labels_path,
    num_epochs: int = 50,
    use_official_tokenizer=False,
    resume_training=False,
    model_save_path="transformer_model.pth",
):
    """训练Transformer模型

    Args:
        bpe_src_path (str): 源语言的BPE词表文件路径
        bpe_tgt_path (str): 目标语言的BPE词表文件路径
        data_path (str): 源语言的语料文件路径
        labels_path (str): 目标语言的语料文件路径
        num_epochs (int, optional): 训练轮数. Defaults to 50.
        use_official_tokenizer (bool, optional): 是否使用官方tokenizer. Defaults to False.
        resume_training (bool, optional): 是否断点续训. Defaults to False.
        model_save_path (str, optional): 模型保存路径. Defaults to "transformer_model.pth".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_per_process_memory_fraction(0.95, device=0)

    # tokenizer需要的是词表文件，dataset需要的是语料文件
    if use_official_tokenizer:
        src_tokenizer = loadOfficialTokenizer(bpe_src_path)
        tgt_tokenizer = loadOfficialTokenizer(bpe_tgt_path)
        src_vocab_size = src_tokenizer.get_vocab_size()
        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    else:
        src_tokenizer = loadMyTokenizer(bpe_src_path)
        tgt_tokenizer = loadMyTokenizer(bpe_tgt_path)
        src_vocab_size = src_tokenizer.vocab_size
        tgt_vocab_size = tgt_tokenizer.vocab_size

    src_batch = load_dataset(data_path)
    tgt_batch = load_dataset(labels_path)

    dataset = TransformerDataset(src_batch, tgt_batch, src_tokenizer, tgt_tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        # persistent_workers=True,  # 保持worker进程存活，避免重复创建
        # prefetch_factor=2,  # 每个worker预取2个batch
    )

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=512,
        max_seq_len=2048,
        head_dim=512 // 8,
        head_num=8,
    )

    if resume_training:
        model.load_state_dict(torch.load(model_save_path))

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.PAD_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 混合精度训练
    scaler = torch.amp.GradScaler(device="cuda")

    # early stopping
    best_loss = float("inf")
    patience = 3
    no_improve = 0
    from tqdm import trange, tqdm

    for epoch in trange(num_epochs, desc="Epoch"):
        model.train()
        train_loss = 0.0

        pbar = tqdm(data_loader, desc="Batch")
        for src_batch, tgt_batch in pbar:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            tgt_input = tgt_batch[:, :-1]
            tgt_label = tgt_batch[:, 1:]

            with torch.amp.autocast(device_type="cuda"):
                output: torch.Tensor = model(src_batch, tgt_input)

                # tgt_label: [batch_size, seq_len]
                # output: [batch_size, seq_len, tgt_vocab_size]
                loss = criterion(
                    output.reshape(-1, output.shape[-1]), tgt_label.reshape(-1)
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            pbar.set_postfix(
                {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            )

            # 第一个 batch 后打印显存占用
            if epoch == 0 and pbar.n == 1:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                print(
                    f"\n[GPU 显存] 已分配: {allocated:.2f} GB / "
                    f"已预留: {reserved:.2f} GB / 峰值: {max_allocated:.2f} GB"
                )

                if max_allocated > 20:
                    print(
                        "⚠️ 警告：显存使用超过 20 GB，可能使用了共享内存（很慢）！减小 batch_size"
                    )

        avg_train_loss = (
            train_loss / len(data_loader)
        )  # len(data_loader)即为batch数量,会调用__len__方法，dataset的__len__方法返回样本数量/batch_size

        # early stopping
        if avg_train_loss < best_loss:
            best_loss = train_loss

            no_improve += 1
            if no_improve >= patience:
                torch.save(model.state_dict(), "best_model.pth")
                logger.info(f"Model saved to best_model.pth\tepoch: {epoch}")
                no_improve = 0

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), "transformer_model.pth")
    logger.info("Model saved to transformer_model.pth")
