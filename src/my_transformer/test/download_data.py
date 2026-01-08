# Hugging Face 中英翻译数据集
from datasets import load_dataset

# 使用 OPUS-100 数据集（现代格式，无需脚本）
print("正在下载数据集...")
dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")

# 查看数据
print(f"数据集大小: {len(dataset['train'])}")
print(f"示例数据: {dataset['train'][0]}")

# 提取文本
src_texts = [item["translation"]["en"] for item in dataset["train"]]
tgt_texts = [item["translation"]["zh"] for item in dataset["train"]]

print(f"\n提取完成！共 {len(src_texts)} 条数据")
print(f"英文示例: {src_texts[:3]}")
print(f"中文示例: {tgt_texts[:3]}")
