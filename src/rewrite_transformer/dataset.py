from encodings import search_function
from httpx import get
from torch.utils.data import Dataset
import torch

from rewrite_transformer.tokenizer import BPETokenizer, Vocab
from tokenizers import Tokenizer

from rewrite_transformer.util import get_logger

logger = get_logger(__name__)


class TransformerDataset(Dataset):
    def __init__(
        self,
        data: list[str],
        labels: list[str],
        src_tokenizer: BPETokenizer | Tokenizer,
        tgt_tokenizer: BPETokenizer | Tokenizer,
    ) -> None:
        super().__init__()
        self.data = data
        self.labels = labels
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if isinstance(self.src_tokenizer, BPETokenizer) and isinstance(
            self.tgt_tokenizer, BPETokenizer
        ):
            src_ids = self.src_tokenizer.encode(x, add_special_tokens=True)
            tgt_ids = self.tgt_tokenizer.encode(y, add_special_tokens=True)
        elif isinstance(self.src_tokenizer, Tokenizer) and isinstance(
            self.tgt_tokenizer, Tokenizer
        ):
            src_ids = self.src_tokenizer.encode(x, add_special_tokens=True).ids
            tgt_ids = self.tgt_tokenizer.encode(y, add_special_tokens=True).ids
        else:
            raise ValueError("Tokenizers must be of the same type")

        return src_ids, tgt_ids


def collate_fn(
    batch: list[tuple[list[int], list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    # batch: [(src_ids, tgt_ids), (src_ids, tgt_ids), ...]
    src_batch, tgt_batch = zip(*batch)

    src_max_len = max(len(ids) for ids in src_batch)
    tgt_max_len = max(len(ids) for ids in tgt_batch)

    pad_id = Vocab.PAD_ID
    padded_src = [ids + [pad_id] * (src_max_len - len(ids)) for ids in src_batch]
    padded_tgt = [ids + [pad_id] * (tgt_max_len - len(ids)) for ids in tgt_batch]

    # logger.info(f"padded_src的类型:{type(padded_src)}", f"padded_src打印:{padded_src}")
    return torch.tensor(padded_src), torch.tensor(padded_tgt)
