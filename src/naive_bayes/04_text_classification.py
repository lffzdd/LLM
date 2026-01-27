"""
===========================================
第四课：实战 - 垃圾邮件分类器
===========================================

本课将应用前面学到的知识，实现一个完整的垃圾邮件分类器。

涵盖内容：
1. 数据预处理
2. 特征提取
3. 模型训练
4. 评估指标
5. 模型优化
"""

import numpy as np
import re
from collections import Counter
from typing import List, Tuple, Dict


# ============================================
# 第一部分：数据预处理
# ============================================


class TextPreprocessor:
    """
    文本预处理器

    功能：
    - 转小写
    - 移除标点符号
    - 移除数字
    - 移除停用词
    """

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

        # 常见英文停用词
        self.stopwords = {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
        }

    def preprocess(self, text: str) -> List[str]:
        """
        预处理文本

        参数:
            text: 原始文本

        返回:
            处理后的词列表
        """
        # 转小写
        text = text.lower()

        # 移除标点符号和数字
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # 分词
        words = text.split()

        # 移除停用词
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stopwords]

        # 移除太短的词
        words = [w for w in words if len(w) > 2]

        return words


# ============================================
# 第二部分：特征提取
# ============================================


class TfidfVectorizer:
    """
    TF-IDF 向量化器

    TF-IDF = TF × IDF

    - TF (Term Frequency): 词频，词在文档中出现的次数
    - IDF (Inverse Document Frequency): 逆文档频率
      IDF(t) = log(N / (1 + df(t)))
      N: 文档总数
      df(t): 包含词 t 的文档数
    """

    def __init__(self, max_features: int = None):
        """
        参数:
            max_features: 最大特征数（选择最常见的词）
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}
        self.preprocessor = TextPreprocessor()

    def fit(self, texts: List[str]):
        """
        构建词汇表并计算 IDF
        """
        # 统计词频
        word_counter = Counter()
        doc_counter = Counter()
        n_docs = len(texts)

        for text in texts:
            words = self.preprocessor.preprocess(text)
            word_counter.update(words)
            doc_counter.update(set(words))  # 每个文档只计一次

        # 选择最常见的词
        if self.max_features:
            most_common = word_counter.most_common(self.max_features)
        else:
            most_common = word_counter.most_common()

        # 构建词汇表
        self.vocabulary = {word: i for i, (word, _) in enumerate(most_common)}

        # 计算 IDF
        for word in self.vocabulary:
            df = doc_counter.get(word, 0)
            self.idf[word] = np.log(n_docs / (1 + df))

        print(f"词汇表大小: {len(self.vocabulary)}")

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为 TF-IDF 向量
        """
        vectors = []

        for text in texts:
            words = self.preprocessor.preprocess(text)
            vector = np.zeros(len(self.vocabulary))

            # 计算 TF
            word_counts = Counter(words)

            for word, count in word_counts.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf = count / len(words) if words else 0
                    vector[idx] = tf * self.idf.get(word, 0)

            vectors.append(vector)

        return np.array(vectors)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


# ============================================
# 第三部分：完整的垃圾邮件分类器
# ============================================


class SpamClassifier:
    """
    垃圾邮件分类器

    使用多项式朴素贝叶斯
    """

    def __init__(self, alpha: float = 1.0, max_features: int = 1000):
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(max_features=max_features)

        # 模型参数
        self.class_priors = {}
        self.feature_log_probs = {}

    def fit(self, texts: List[str], labels: np.ndarray):
        """
        训练模型
        """
        print("=" * 60)
        print("🎯 训练垃圾邮件分类器")
        print("=" * 60)

        # 特征提取
        print("\n📝 特征提取...")
        X = self.vectorizer.fit_transform(texts)

        n_samples, n_features = X.shape
        classes = np.unique(labels)

        print(f"训练样本数: {n_samples}")
        print(f"特征维度: {n_features}")

        # 计算类别先验概率和特征条件概率
        print("\n📊 计算概率...")

        for c in classes:
            X_c = X[labels == c]

            # 先验概率（使用对数）
            self.class_priors[c] = np.log(len(X_c) / n_samples)

            # 特征条件概率（使用对数）
            feature_counts = np.sum(X_c, axis=0)
            total_count = np.sum(feature_counts)

            # 拉普拉斯平滑
            smoothed_probs = (feature_counts + self.alpha) / (
                total_count + self.alpha * n_features
            )
            self.feature_log_probs[c] = np.log(smoothed_probs)

            label_name = "🚫 垃圾邮件" if c == 1 else "✉️ 正常邮件"
            print(f"  {label_name}: {len(X_c)} 封 ({len(X_c) / n_samples:.1%})")

        print("\n✅ 训练完成！")

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        预测
        """
        X = self.vectorizer.transform(texts)

        predictions = []
        for x in X:
            scores = {}
            for c in self.class_priors:
                # log P(y=c|x) ∝ log P(y=c) + Σ x_i × log P(x_i|y=c)
                score = self.class_priors[c] + np.sum(x * self.feature_log_probs[c])
                scores[c] = score

            pred = max(scores, key=scores.get)
            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        预测概率
        """
        X = self.vectorizer.transform(texts)

        probas = []
        for x in X:
            log_scores = []
            for c in sorted(self.class_priors.keys()):
                score = self.class_priors[c] + np.sum(x * self.feature_log_probs[c])
                log_scores.append(score)

            # Log-sum-exp 转换为概率
            log_scores = np.array(log_scores)
            max_score = np.max(log_scores)
            exp_scores = np.exp(log_scores - max_score)
            proba = exp_scores / np.sum(exp_scores)
            probas.append(proba)

        return np.array(probas)


# ============================================
# 第四部分：评估指标
# ============================================


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标

    返回:
        - accuracy: 准确率
        - precision: 精确率（垃圾邮件）
        - recall: 召回率（垃圾邮件）
        - f1: F1 分数
    """
    # 混淆矩阵
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negative

    # 计算指标
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def print_metrics(metrics: Dict[str, float]):
    """
    打印评估指标
    """
    print("\n" + "=" * 60)
    print("📊 评估指标")
    print("=" * 60)

    print(f"\n准确率 (Accuracy):  {metrics['accuracy']:.2%}")
    print(f"精确率 (Precision): {metrics['precision']:.2%}")
    print(f"召回率 (Recall):    {metrics['recall']:.2%}")
    print(f"F1 分数:            {metrics['f1']:.2%}")

    cm = metrics["confusion_matrix"]
    print(f"\n混淆矩阵:")
    print(f"              预测正常  预测垃圾")
    print(f"实际正常        {cm['tn']:4d}      {cm['fp']:4d}")
    print(f"实际垃圾        {cm['fn']:4d}      {cm['tp']:4d}")

    print("""
    💡 指标解释:
    - 准确率: 正确预测的比例
    - 精确率: 预测为垃圾邮件中，真正是垃圾邮件的比例
    - 召回率: 真正的垃圾邮件中，被正确识别的比例
    - F1 分数: 精确率和召回率的调和平均
    """)


# ============================================
# 第五部分：主程序 - 完整的演示
# ============================================


def main():
    print("\n" + "🚀 " * 25)
    print("\n    垃圾邮件分类器 - 完整演示")
    print("\n" + "🚀 " * 25 + "\n")

    # 模拟数据集
    # 正常邮件 (label=0)
    ham_emails = [
        "Hi John, let's have a meeting tomorrow at 10am to discuss the project.",
        "Please review the attached document and send your feedback by Friday.",
        "Thank you for your email. I will get back to you shortly.",
        "The team meeting is rescheduled to next Monday at 2pm.",
        "Could you please send me the report when it's ready?",
        "Looking forward to our lunch tomorrow. See you at noon!",
        "I've completed the tasks you assigned. Let me know if you need anything else.",
        "Happy birthday! Hope you have a wonderful day!",
        "The flight is confirmed for next week. Here are your booking details.",
        "Great job on the presentation! The client was very impressed.",
        "Please find the invoice attached. Payment is due by end of month.",
        "Would you like to join us for dinner this Saturday?",
        "I'll be out of office next week. Please contact Sarah for urgent matters.",
        "The product shipment will arrive on Thursday morning.",
        "Let's schedule a call to go over the project timeline.",
    ]

    # 垃圾邮件 (label=1)
    spam_emails = [
        "CONGRATULATIONS! You have won a FREE iPhone! Click here to claim now!",
        "Make money fast! Work from home and earn $5000 per week!",
        "URGENT: Your account has been compromised. Click here to verify!",
        "Get rich quick! Investment opportunity of a lifetime!",
        "FREE GIFT! You have been selected for a special prize!",
        "Lose weight fast with this miracle pill! Order now!",
        "Your lottery ticket has won $1,000,000! Send us your bank details!",
        "Hot singles in your area want to meet you! Click here!",
        "CHEAP VIAGRA and other medications! No prescription needed!",
        "You are our LUCKY WINNER! Claim your FREE vacation now!",
        "Make $10,000 per month from home! No experience required!",
        "URGENT: You have unpaid taxes. Send payment immediately!",
        "Congratulations! You've been selected for a credit card with 0% APR!",
        "FREE casino bonus! Triple your deposit instantly!",
        "Secret to making millions revealed! Limited time offer!",
    ]

    # 合并数据
    all_texts = ham_emails + spam_emails
    all_labels = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

    # 随机打乱
    np.random.seed(42)
    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    all_labels = all_labels[indices]

    # 划分训练集和测试集
    split = int(0.7 * len(all_texts))
    train_texts = all_texts[:split]
    train_labels = all_labels[:split]
    test_texts = all_texts[split:]
    test_labels = all_labels[split:]

    print(f"训练集大小: {len(train_texts)}")
    print(f"测试集大小: {len(test_texts)}")

    # 训练模型
    clf = SpamClassifier(alpha=1.0, max_features=500)
    clf.fit(train_texts, train_labels)

    # 预测
    predictions = clf.predict(test_texts)

    # 评估
    metrics = evaluate(test_labels, predictions)
    print_metrics(metrics)

    # 交互式测试
    print("\n" + "=" * 60)
    print("🔍 测试新邮件")
    print("=" * 60)

    test_emails = [
        "Hi, can we reschedule our meeting to next week?",
        "FREE MONEY! You have won $50,000! Click here to claim!",
        "Please review the quarterly report attached.",
        "URGENT: Your password expires today! Click to reset!",
    ]

    for email in test_emails:
        pred = clf.predict([email])[0]
        proba = clf.predict_proba([email])[0]

        label = "🚫 垃圾邮件" if pred == 1 else "✉️ 正常邮件"
        confidence = proba[pred]

        print(f'\n邮件: "{email[:50]}..."' if len(email) > 50 else f'\n邮件: "{email}"')
        print(f"预测: {label} (置信度: {confidence:.2%})")

    # 总结
    print("\n" + "=" * 60)
    print("📚 总结：朴素贝叶斯分类器的优缺点")
    print("=" * 60)
    print("""
    ✅ 优点:
    - 训练速度快，适合大规模数据
    - 对小样本数据效果好
    - 可解释性强
    - 对缺失数据不敏感
    
    ❌ 缺点:
    - 特征独立假设在现实中往往不成立
    - 对输入数据的分布假设敏感
    - 概率估计可能不准确
    
    💡 改进方向:
    - 使用更好的特征（n-gram、词嵌入）
    - 特征选择
    - 参数调优（不同的平滑参数）
    - 组合模型（投票、集成）
    """)


if __name__ == "__main__":
    main()
