"""
===========================================
第三课：多项式朴素贝叶斯 (Multinomial Naive Bayes)
===========================================

多项式朴素贝叶斯用于处理离散特征，特别适合文本分类。

核心思想：
- 假设特征是词频或词计数
- 每个类别的特征服从多项式分布

关键公式：
P(x_i|y=c) = (count(x_i, c) + α) / (count(c) + α × |V|)

其中：
- count(x_i, c)：类别 c 中特征 i 出现的次数
- count(c)：类别 c 中所有特征的总计数
- α：平滑参数（拉普拉斯平滑）
- |V|：特征词汇表大小
"""

import numpy as np
from collections import Counter, defaultdict


class MultinomialNaiveBayes:
    """
    多项式朴素贝叶斯分类器

    适用场景：
    - 文本分类（垃圾邮件检测、情感分析等）
    - 特征是计数或频率
    """

    def __init__(self, alpha: float = 1.0):
        """
        初始化分类器

        参数:
            alpha: 拉普拉斯平滑参数（默认为 1.0）
                   - α = 0：不平滑（可能导致零概率问题）
                   - α = 1：拉普拉斯平滑
                   - α < 1：Lidstone 平滑
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}  # P(y)
        self.feature_probs = {}  # P(x_i|y)
        self.vocabulary_size = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型

        参数:
            X: 训练数据（词频矩阵），形状 (n_samples, n_features)
            y: 标签，形状 (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y) # unique方法返回唯一的元素,例如[1,1,2,2,3]返回[1,2,3],假设y是[0,0,1,0,1,1],则unique返回[0,1]
        self.vocabulary_size = n_features

        print("=" * 50)
        print("训练多项式朴素贝叶斯分类器")
        print("=" * 50)
        print(f"样本数: {n_samples}")
        print(f"特征数（词汇量）: {n_features}")
        print(f"类别: {self.classes}")
        print(f"平滑参数 α: {self.alpha}")
        print()

        for c in self.classes:
            # 筛选该类别的样本
            X_c = X[y == c]

            # 计算先验概率
            self.class_priors[c] = len(X_c) / n_samples

            # 计算每个特征的条件概率
            # P(x_i|y=c) = (count(x_i, c) + α) / (count(c) + α × |V|)
            feature_counts = np.sum(X_c, axis=0)  # 类别c中出现每个特征的次数,如[0,1,2,3]表示类别c中出现0次特征0,1次特征1,2次特征2,3次特征3
            total_count = np.sum(feature_counts)  # 所有特征的总计数

            # 应用拉普拉斯平滑
            # 假设每个特征至少出现了alpha次,那么总特征次数至少出现了n_features*alpha次
            # 类别c中每个特征的出现频率,用于连乘计算似然
            self.feature_probs[c] = (feature_counts + self.alpha) / (
                total_count + self.alpha * n_features
            )

            print(f"类别 {c}:")
            print(f"  样本数: {len(X_c)}")
            print(f"  先验概率 P(y={c}) = {self.class_priors[c]:.4f}")
            print(f"  特征总计数: {total_count}")
            print()

        print("训练完成！\n")

    def _calculate_log_posterior(self, x: np.ndarray, c) -> float:
        """
        计算对数后验概率

        使用对数概率避免数值下溢：
        log P(y=c|x) ∝ log P(y=c) + Σ x_i × log P(x_i|y=c)

        参数:
            x: 单个样本的特征向量
            c: 类别

        返回:
            对数后验概率（未归一化）
        """
        log_prior = np.log(self.class_priors[c])
        log_likelihood = np.sum(x * np.log(self.feature_probs[c]))

        return log_prior + log_likelihood

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测对数概率

        参数:
            X: 测试数据，形状 (n_samples, n_features)

        返回:
            对数概率矩阵，形状 (n_samples, n_classes)
        """
        log_probas = []

        for x in X:
            log_posteriors = []
            for c in self.classes:
                log_posterior = self._calculate_log_posterior(x, c)
                log_posteriors.append(log_posterior)
            log_probas.append(log_posteriors)

        return np.array(log_probas)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        参数:
            X: 测试数据，形状 (n_samples, n_features)

        返回:
            概率矩阵，形状 (n_samples, n_classes)
        """
        log_probas = self.predict_log_proba(X)

        # 使用 log-sum-exp 技巧避免数值问题
        # P(y=c|x) = exp(log P(c|x)) / Σ exp(log P(c'|x))
        max_log_proba = np.max(log_probas, axis=1, keepdims=True)
        exp_log_probas = np.exp(log_probas - max_log_proba)
        probas = exp_log_probas / np.sum(exp_log_probas, axis=1, keepdims=True)

        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        参数:
            X: 测试数据，形状 (n_samples, n_features)

        返回:
            预测的类别，形状 (n_samples,)
        """
        log_probas = self.predict_log_proba(X)
        return self.classes[np.argmax(log_probas, axis=1)]


# ============================================
# 辅助类：简单的文本向量化器
# ============================================


class SimpleVectorizer:
    """
    简单的词袋模型向量化器

    将文本转换为词频向量
    """

    def __init__(self):
        self.vocabulary = {}
        self.inv_vocabulary = {} # 词汇表的逆映射

    def fit(self, texts: list):
        """
        构建词汇表

        参数:
            texts: 文本列表
        """
        word_set = set()
        for text in texts:
            words = text.lower().split()
            word_set.update(words)

        self.vocabulary = {word: i for i, word in enumerate(sorted(word_set))}
        self.inv_vocabulary = {i: word for word, i in self.vocabulary.items()}

        print(f"词汇表大小: {len(self.vocabulary)}")
        print(f"词汇表: {list(self.vocabulary.keys())[:10]}...")

    def transform(self, texts: list) -> np.ndarray:
        """
        将文本转换为词频向量

        参数:
            texts: 文本列表

        返回:
            词频矩阵，形状 (n_samples, vocabulary_size)
        """
        vectors = []

        for text in texts:
            vector = np.zeros(len(self.vocabulary))
            words = text.lower().split()

            # 统计词频
            for word in words:
                if word in self.vocabulary:
                    vector[self.vocabulary[word]] += 1

            vectors.append(vector)

        return np.array(vectors)

    def fit_transform(self, texts: list) -> np.ndarray:
        """
        先 fit 再 transform
        """
        self.fit(texts)
        return self.transform(texts)


# ============================================
# 示例：简单情感分析
# ============================================


def sentiment_example():
    """
    简单情感分析示例
    """
    # 训练数据
    texts = [
        "I love this movie it is great",
        "This film is wonderful amazing",
        "Excellent movie best ever",
        "I love the acting great job",
        "This movie is terrible bad",
        "Worst film ever boring",
        "I hate this movie awful",
        "Terrible acting bad script",
    ]

    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 1=正面, 0=负面

    print("=" * 50)
    print("简单情感分析示例")
    print("=" * 50)
    print("\n训练数据:")
    for text, label in zip(texts, labels):
        sentiment = "正面 😊" if label == 1 else "负面 😞"
        print(f"  [{sentiment}] {text}")
    print()

    # 向量化
    vectorizer = SimpleVectorizer()
    X_train = vectorizer.fit_transform(texts)

    # 训练
    clf = MultinomialNaiveBayes(alpha=1.0)
    clf.fit(X_train, labels)

    # 测试
    test_texts = [
        "this movie is great love it",
        "terrible movie I hate it",
        "the film is wonderful",
    ]

    X_test = vectorizer.transform(test_texts)
    predictions = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    print("=" * 50)
    print("预测结果")
    print("=" * 50)

    for text, pred, proba in zip(test_texts, predictions, probas):
        sentiment = "正面 😊" if pred == 1 else "负面 😞"
        print(f'\n文本: "{text}"')
        print(f"预测: {sentiment}")
        print(f"概率: 负面={proba[0]:.4f}, 正面={proba[1]:.4f}")


# ============================================
# 详细解释拉普拉斯平滑
# ============================================


def explain_laplace_smoothing():
    """
    解释为什么需要拉普拉斯平滑
    """
    explanation = """
    ============================================
    为什么需要拉普拉斯平滑？
    ============================================
    
    问题：零概率问题
    
    假设训练数据：
    - 正面评论中没有出现过 "terrible" 这个词
    - 测试时遇到: "This is terrible"
    
    如果不平滑：
    P("terrible"|正面) = 0
    P(正面|文本) = P(正面) × ... × P("terrible"|正面) × ... = 0
    
    无论其他证据多强，结果都是 0！这显然不合理。
    
    解决方案：拉普拉斯平滑
    
    P(x_i|y=c) = (count(x_i, c) + α) / (count(c) + α × |V|)
    
    - 给每个词加上一个小的计数 α（通常 α=1）
    - 分母也相应调整，保证概率和为 1
    
    效果：
    - 没见过的词不会导致零概率
    - 相当于假设每个词至少出现过 α 次
    ============================================
    """
    print(explanation)

    # 数值示例
    print("数值示例:")
    print("-" * 50)

    # 假设词汇表大小为 10，某个类别的总词数为 100
    vocab_size = 10
    total_count = 100

    # 某个词在该类别中出现 5 次
    word_count = 5

    # 另一个词没出现过
    unseen_word_count = 0

    # 不使用平滑
    print("\n不使用平滑 (α=0):")
    print(
        f"  P(见过的词|类别) = {word_count}/{total_count} = {word_count / total_count:.4f}"
    )
    print(
        f"  P(没见过的词|类别) = {unseen_word_count}/{total_count} = 0.0000 ❌ 问题！"
    )

    # 使用拉普拉斯平滑
    alpha = 1
    print(f"\n使用拉普拉斯平滑 (α={alpha}):")
    smoothed_total = total_count + alpha * vocab_size
    print(
        f"  P(见过的词|类别) = ({word_count}+{alpha})/({total_count}+{alpha}×{vocab_size}) = {(word_count + alpha) / smoothed_total:.4f}"
    )
    print(
        f"  P(没见过的词|类别) = ({unseen_word_count}+{alpha})/({total_count}+{alpha}×{vocab_size}) = {(unseen_word_count + alpha) / smoothed_total:.4f} ✓ 不再是零！"
    )


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    print("\n" + "📝 " * 20 + "\n")
    print("第三课：多项式朴素贝叶斯分类器")
    print("\n" + "📝 " * 20 + "\n")

    # 解释拉普拉斯平滑
    explain_laplace_smoothing()

    print("\n")

    # 情感分析示例
    sentiment_example()

    print("\n" + "=" * 50)
    print("下一步：运行 04_text_classification.py 进行实战练习")
    print("=" * 50)
