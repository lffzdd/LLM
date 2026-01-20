from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train():
    # 1. 创建 BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # 2. 设置预分词器（按空格分词）
    tokenizer.pre_tokenizer = Whitespace()

    # 3. 创建训练器
    trainer = BpeTrainer(
        vocab_size=32000, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )

    # 4. 训练（几分钟搞定你跑了十几分钟的量）
    tokenizer.train(files=["../dataset/TED2020.en-zh_cn.en"], trainer=trainer)

    # 5. 保存
    tokenizer.save("en_tokenizer.json")

    # 6. 使用
    output = tokenizer.encode("hello world")
    print(output.tokens)  # ['hello', '</w>', 'world', '</w>']
    print(output.ids)  # [123, 456, 789, 456]

    # 7. 解码
    text = tokenizer.decode(output.ids)
    print(text)  # "hello world"


def eval():
    tokenizer = Tokenizer.from_file("./en_tokenizer.json")

    output = tokenizer.encode("fuckyourgaylao@~￥")

    print(output.tokens)  # ['hello', '</w>', 'world', '</w>']
    print(output.ids)  # [123, 456, 789, 456]

    text = tokenizer.decode(output.ids)
    print(text)  # "hello world"

eval()