from rewrite_transformer.tokenizer import BPETokenizer
from rewrite_transformer.util import load_dataset

import time


def test_tokenizer():
    tokenizer = BPETokenizer()

    texts = ["hello world", "how are you"]

    tokenizer.train(texts, 100, max_epoch=100)

    text_ids = tokenizer.encode("fuck you")
    print(text_ids)

    text = tokenizer.decode(text_ids)


def test_load_dataset():
    en_path = "../dataset/TED2020.en-zh_cn.en"
    cn_path = "../dataset/TED2020.en-zh_cn.zh_cn"

    en_data = load_dataset(en_path)
    cn_data = load_dataset(cn_path)

    print(en_data, print(cn_data))


def test_train_bpe():
    start = time.perf_counter()

    en_data = load_dataset("../dataset/TED2020.en-zh_cn.en")
    en_tokenizer = BPETokenizer()
    en_tokenizer.train(en_data, max_vocab_size=32000, max_epoch=32000)
    en_tokenizer.save("en_tokenizer.json")

    cn_data = load_dataset("../dataset/TED2020.en-zh_cn.zh_cn")
    cn_tokenizer = BPETokenizer()
    cn_tokenizer.train(cn_data, max_vocab_size=32000, max_epoch=32000)
    cn_tokenizer.save("cn_tokenizer.json")

    end = time.perf_counter()

    print("耗时：", end - start)


if __name__ == "__main__":
    # test_tokenizer()
    test_train_bpe()
