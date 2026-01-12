//! BPE (Byte Pair Encoding) Tokenizer - Rust 实现
//! 
//! 这是一个用于学习的 BPE tokenizer 实现，展示了如何用 Rust 编写高效的 tokenizer。

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// 词表：管理 token 和 id 之间的映射
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    /// token -> id 映射
    token_to_id: HashMap<String, u32>,
    /// id -> token 映射
    id_to_token: HashMap<u32, String>,
}

impl Vocab {
    /// 创建新的词表，包含特殊 token
    pub fn new() -> Self {
        let mut vocab = Vocab {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
        };
        
        // 添加特殊 token
        vocab.add_token("<PAD>".to_string());  // id = 0
        vocab.add_token("<UNK>".to_string());  // id = 1
        vocab.add_token("<BOS>".to_string());  // id = 2
        vocab.add_token("<EOS>".to_string());  // id = 3
        
        vocab
    }
    
    /// 添加 token 到词表
    pub fn add_token(&mut self, token: String) -> u32 {
        if let Some(&id) = self.token_to_id.get(&token) {
            return id;
        }
        
        let id = self.token_to_id.len() as u32;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
        id
    }
    
    /// 获取 token 对应的 id
    pub fn get_id(&self, token: &str) -> u32 {
        *self.token_to_id.get(token).unwrap_or(&1)  // 默认返回 <UNK> 的 id
    }
    
    /// 获取 id 对应的 token
    pub fn get_token(&self, id: u32) -> &str {
        self.id_to_token.get(&id).map(|s| s.as_str()).unwrap_or("<UNK>")
    }
    
    /// 词表大小
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }
    
    /// 是否包含 token
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }
}

/// BPE Tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    /// 词表
    pub vocab: Vocab,
    /// 合并规则：(token1, token2) -> merged_token
    merge_rules: Vec<(String, String)>,
}

impl BPETokenizer {
    /// 创建新的 tokenizer
    pub fn new() -> Self {
        BPETokenizer {
            vocab: Vocab::new(),
            merge_rules: Vec::new(),
        }
    }
    
    /// 将单词拆分成字符，并添加 </w> 结尾标记
    fn tokenize_word(word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        tokens.push("</w>".to_string());
        tokens
    }
    
    /// 统计 token pair 频率
    fn count_pairs(
        word_freqs: &HashMap<String, u32>,
        tokenized_words: &HashMap<String, Vec<String>>,
    ) -> HashMap<(String, String), u32> {
        let mut pair_counts: HashMap<(String, String), u32> = HashMap::new();
        
        for (word, tokens) in tokenized_words {
            let freq = *word_freqs.get(word).unwrap_or(&0);
            
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                *pair_counts.entry(pair).or_insert(0) += freq;
            }
        }
        
        pair_counts
    }
    
    /// 合并指定的 pair
    fn merge_pair(
        tokenized_words: &mut HashMap<String, Vec<String>>,
        pair: &(String, String),
    ) {
        let merged = format!("{}{}", pair.0, pair.1);
        
        for tokens in tokenized_words.values_mut() {
            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    new_tokens.push(merged.clone());
                    i += 2;  // 跳过两个 token
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            
            *tokens = new_tokens;
        }
    }
    
    /// 训练 BPE tokenizer
    /// 
    /// # Arguments
    /// * `texts` - 训练文本列表
    /// * `vocab_size` - 目标词表大小
    /// * `min_frequency` - 最小频率阈值
    pub fn train(&mut self, texts: &[String], vocab_size: usize, min_frequency: u32) {
        println!("开始 BPE 训练...");
        println!("  目标词表大小: {}", vocab_size);
        println!("  最小频率: {}", min_frequency);
        
        // Step 1: 统计单词频率
        let mut word_freqs: HashMap<String, u32> = HashMap::new();
        for text in texts {
            for word in text.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }
        println!("  单词数量: {}", word_freqs.len());
        
        // Step 2: 将单词拆分成字符
        let mut tokenized_words: HashMap<String, Vec<String>> = HashMap::new();
        for word in word_freqs.keys() {
            tokenized_words.insert(word.clone(), Self::tokenize_word(word));
        }
        
        // Step 3: 统计字符频率，添加到词表
        let mut char_freqs: HashMap<String, u32> = HashMap::new();
        for (word, tokens) in &tokenized_words {
            let freq = *word_freqs.get(word).unwrap_or(&0);
            for token in tokens {
                *char_freqs.entry(token.clone()).or_insert(0) += freq;
            }
        }
        
        // 添加达到频率阈值的字符到词表
        for (char_token, freq) in &char_freqs {
            if *freq >= min_frequency && !self.vocab.contains(char_token) {
                self.vocab.add_token(char_token.clone());
            }
        }
        println!("  初始词表大小: {}", self.vocab.len());
        
        // Step 4: 迭代合并最频繁的 pair
        let mut iteration = 0;
        while self.vocab.len() < vocab_size {
            iteration += 1;
            
            // 统计 pair 频率
            let pair_counts = Self::count_pairs(&word_freqs, &tokenized_words);
            
            if pair_counts.is_empty() {
                println!("  没有更多的 pair 可以合并");
                break;
            }
            
            // 找到最频繁的 pair
            let (best_pair, best_freq) = pair_counts
                .iter()
                .max_by_key(|(_, freq)| *freq)
                .unwrap();
            
            if *best_freq < min_frequency {
                println!("  最高频率 {} 低于阈值 {}", best_freq, min_frequency);
                break;
            }
            
            // 合并这个 pair
            let merged_token = format!("{}{}", best_pair.0, best_pair.1);
            Self::merge_pair(&mut tokenized_words, best_pair);
            
            // 添加到词表和合并规则
            self.vocab.add_token(merged_token);
            self.merge_rules.push(best_pair.clone());
            
            // 每 100 次打印进度
            if iteration % 100 == 0 {
                println!("  迭代 {}: 词表大小 = {}", iteration, self.vocab.len());
            }
        }
        
        println!("训练完成！");
        println!("  最终词表大小: {}", self.vocab.len());
        println!("  合并规则数量: {}", self.merge_rules.len());
    }
    
    /// 编码文本为 token id 列表
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        
        for word in text.split_whitespace() {
            let mut tokens = Self::tokenize_word(word);
            
            // 应用所有合并规则
            for (a, b) in &self.merge_rules {
                let merged = format!("{}{}", a, b);
                let mut new_tokens = Vec::with_capacity(tokens.len());
                let mut i = 0;
                
                while i < tokens.len() {
                    if i + 1 < tokens.len() && &tokens[i] == a && &tokens[i + 1] == b {
                        new_tokens.push(merged.clone());
                        i += 2;
                    } else {
                        new_tokens.push(tokens[i].clone());
                        i += 1;
                    }
                }
                tokens = new_tokens;
            }
            
            // 转换为 id
            for token in tokens {
                result.push(self.vocab.get_id(&token));
            }
        }
        
        result
    }
    
    /// 解码 token id 列表为文本
    pub fn decode(&self, ids: &[u32]) -> String {
        let tokens: Vec<&str> = ids.iter()
            .map(|id| self.vocab.get_token(*id))
            .collect();
        
        let text = tokens.join("");
        text.replace("</w>", " ").trim().to_string()
    }
    
    /// 保存 tokenizer 到文件
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        println!("Tokenizer 已保存到: {}", path);
        Ok(())
    }
    
    /// 从文件加载 tokenizer
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let tokenizer: BPETokenizer = serde_json::from_str(&json)?;
        println!("Tokenizer 已从 {} 加载", path);
        Ok(tokenizer)
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vocab() {
        let mut vocab = Vocab::new();
        assert_eq!(vocab.len(), 4);  // PAD, UNK, BOS, EOS
        
        let id = vocab.add_token("hello".to_string());
        assert_eq!(id, 4);
        assert_eq!(vocab.get_id("hello"), 4);
        assert_eq!(vocab.get_token(4), "hello");
    }
    
    #[test]
    fn test_tokenize_word() {
        let tokens = BPETokenizer::tokenize_word("hello");
        assert_eq!(tokens, vec!["h", "e", "l", "l", "o", "</w>"]);
    }
    
    #[test]
    fn test_train_and_encode() {
        let texts = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world of code".to_string(),
        ];
        
        let mut tokenizer = BPETokenizer::new();
        tokenizer.train(&texts, 50, 1);
        
        let encoded = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&encoded);
        
        assert_eq!(decoded, "hello world");
    }
}
