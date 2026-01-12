//! BPE Tokenizer 使用示例

use rust_bpe::BPETokenizer;

fn main() {
    println!("=== Rust BPE Tokenizer Demo ===\n");
    
    // 准备训练数据
    let texts = vec![
        "the cat sat on the mat".to_string(),
        "the dog ran in the park".to_string(),
        "the cat and dog played together".to_string(),
        "hello world".to_string(),
        "hello there".to_string(),
        "world of programming".to_string(),
        "machine learning is amazing".to_string(),
        "deep learning is fun".to_string(),
    ];
    
    // 创建并训练 tokenizer
    let mut tokenizer = BPETokenizer::new();
    tokenizer.train(&texts, 100, 1);
    
    // 测试编码和解码
    println!("\n=== 测试编码/解码 ===");
    let test_texts = vec![
        "hello world",
        "the cat sat",
        "machine learning",
    ];
    
    for text in test_texts {
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        
        println!("原文: \"{}\"", text);
        println!("编码: {:?}", encoded);
        println!("解码: \"{}\"", decoded);
        println!();
    }
    
    // 保存 tokenizer
    tokenizer.save("tokenizer.json").expect("保存失败");
    
    // 加载并验证
    println!("\n=== 测试保存/加载 ===");
    let loaded_tokenizer = BPETokenizer::load("tokenizer.json").expect("加载失败");
    
    let text = "hello world";
    let encoded = loaded_tokenizer.encode(text);
    let decoded = loaded_tokenizer.decode(&encoded);
    println!("加载后测试: \"{}\" -> {:?} -> \"{}\"", text, encoded, decoded);
}
