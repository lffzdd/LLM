from tokenizers import Tokenizer
from tokenizers.models import BPE

from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=32000, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

tokenizer.train(files=["../dataset/TED2020.en-zh_cn.zh_cn"], trainer=trainer)

tokenizer.save("cn_tokenizer.json")

# output = tokenizer.encode("hello world")
output = tokenizer.encode("你好世界")
print(output.tokens)
print(output.ids)

text = tokenizer.decode(output.ids)
print(text)

