from my_transformer._1_tokenizer import BPETokenizer

def test_merge_by_pair():
    tokenizer = BPETokenizer()

    tokenized_words = {
        "hello": ["h", "e", "l", "l", "o"],
        "hell": ["h", "e", "l", "l"],
        "helicopter": ["h", "e", "l", "i", "c", "o", "p", "t", "e", "r"],
    }

    pair = ("l", "l")
    updated_tokenized_words = tokenizer._merge_by_pair(tokenized_words, pair)

    assert updated_tokenized_words == {
        "hello": ["h", "e", "ll", "o"],
        "hell": ["h", "e", "ll"],
        "helicopter": ["h", "e", "l", "i", "c", "o", "p", "t", "e", "r"],
    }

def test_train():
    tokenizer = BPETokenizer()
    text_list = ["hello hello", "hell helicopter", "hello hell"]
    target_size = 20  # 增加 target_size 以容纳更多 token
    min_frequency = 1

    tokenizer.train(text_list, target_size, min_frequency)

    # 检查是否至少包含基础字符
    basic_tokens = {"h", "e", "l", "o", "</w>"}
    actual_tokens = set(tokenizer.token_to_id.keys())
    
    print(f"Actual tokens: {actual_tokens}")
    print(f"Token count: {len(actual_tokens)}")

    assert basic_tokens.issubset(actual_tokens)
    assert len(actual_tokens) <= target_size  # 不超过目标大小

def test_encode():
    tokenizer = BPETokenizer()
    text_list = ["hello hello", "hell helicopter", "hello hell"]
    target_size = 10
    min_frequency = 1

    tokenizer.train(text_list, target_size, min_frequency)

    encoded = tokenizer.encode("hello hell")

    # 检查编码结果是否符合预期
    assert isinstance(encoded, list)
    assert all(isinstance(word_token, list) for word_token in encoded)
    assert all(all(isinstance(token_id, int) for token_id in word_token) for word_token in encoded)