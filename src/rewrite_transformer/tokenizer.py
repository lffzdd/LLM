from collections import Counter


class Vocab:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self) -> None:
        self.token_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        self.id_to_token = {k: v for v, k in self.token_to_id.items()}

    def add_token(self, token: str):
        if token not in self.token_to_id:
            id = len(self.token_to_id)
            self.token_to_id[token] = id
            self.id_to_token[id] = token


class BPETokenizer:
    def __init__(self, vocab: Vocab | None) -> None:
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocab()

        self.pair_freq=Counter()
        self.merge_pair:list[str]=[]

    def _sperate_texts(
        self, texts: list[str], eow: str = "/w", bos:str=Vocab.BOS_TOKEN,eos: str = Vocab.EOS_TOKEN
    ) -> list[list[str]]:
        """把文本拆分成单词

        Args:
            texts: 文本列表,每个文本是一个字符串
        """
        texts_to_tokens: list[list[str]] = []
        for text in texts:  # "hello world"
            s: list[str] = [bos]

            # 遍历['hello','world']
            for word in text.strip().split():
                # extend会把'hello'迭代加到s，即[...]->[...,'h','e','l','l','o']
                s.extend(word)
                s.append(eow)

            # [...]->[...,['h','e','l','l','o','</w>','w','o','r','l','d','</w>']]
            texts_to_tokens.append(s + [eos])
        return texts_to_tokens

    def _update_pair_freq(self,text:list[str]):
        """更新token对频率"""
        # text:['h','e','l','l','o','</w>','w','o','r','l','d','</w>']
        for i in range(len(text)-1):
            self.pair_freq.update(text[i]+text[i+1])
    
    def _update_merge_pair(self,pair_freq:Counter=self.pair_freq):
        """根据频率更新merge_pair"""
        most_pair,freq=pair_freq.most_common(1)[0]
        self.merge_pair.append(most_pair)

    def _merge_text(self,text:list[str],pair:str):
        new_text:list[str]=[]
        for i in range(len(text)-1):
            if text[i]+text[i+1]==pair:
                new_text.append(text[i])
        
    def train(self,texts:list[str]):
        texts_to_tokens=self._sperate_texts(texts)
        for text_tokens in texts_to_tokens:
