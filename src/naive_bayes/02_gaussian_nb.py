"""
===========================================
第二课：高斯朴素贝叶斯 (Gaussian Naive Bayes)
===========================================

高斯朴素贝叶斯用于处理连续特征。

核心假设：
1. 特征之间相互独立（朴素假设）
2. 每个特征在给定类别下服从高斯（正态）分布

高斯分布公式：
P(x|μ,σ) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

其中：
- μ (mu)：均值
- σ (sigma)：标准差
"""

import numpy as np
from collections import defaultdict


class GaussianNaiveBayes:
    """
    高斯朴素贝叶斯分类器

    适用场景：
    - 特征是连续值（如身高、体重、温度等）
    - 假设特征服从正态分布
    """

    def __init__(self):
        """初始化分类器"""
        self.classes = None  # 所有类别
        self.class_priors = {}  # 先验概率 P(y)
        self.means = {}  # 每个类别每个特征的均值
        self.variances = {}  # 每个类别每个特征的方差

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型

        参数:
            X: 训练数据，形状 (n_samples, n_features)
            y: 标签，形状 (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        print("=" * 50)
        print("训练高斯朴素贝叶斯分类器")
        print("=" * 50)
        print(f"样本数: {n_samples}")
        print(f"特征数: {n_features}")
        print(f"类别: {self.classes}")
        print()

        # 对每个类别
        for c in self.classes:
            # 筛选出属于该类别的样本
            """
            y == c 会返回一个布尔数组，表示 y 中每个元素是否等于 c
            X[y == c] 会返回 X 中对应位置为 True 的行

            X = np.array([
                [5.1, 3.5],  # 样本0，类别0
                [4.9, 3.0],  # 样本1，类别0  
                [7.0, 3.2],  # 样本2，类别1  ← 保留
                [6.4, 3.2],  # 样本3，类别1  ← 保留
                [6.3, 3.3],  # 样本4，类别2
            ])
            y = np.array([0, 0, 1, 1, 2])

            X_c = X[y == 1]
            # 返回：
            # array([[7.0, 3.2],
            #        [6.4, 3.2]])
            """
            X_c = X[y == c]

            # 计算先验概率 P(y=c)
            # len(X_c) / n_samples 表示类别 c 的样本数占总样本数的比例
            self.class_priors[c] = len(X_c) / n_samples

            # 计算每个特征的均值和方差
            """
            axis=0 表示按第0轴（行）计算

            array([[7.0, 3.2],
                   [6.4, 3.2]])
            ->
            mean:array([6.7, 3.2])  # 即 [(7.0+6.4)/2, (3.2+3.2)/2]
            var:array([0.125, 0.0])  # 即 [(7.0-6.7)^2/2, (3.2-3.2)^2/2]

            类别c中,第0个特征的均值为6.7,第1个特征的均值为3.2
            第0个特征的方差为0.125,第1个特征的方差为0.0
            """
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)

            print(f"类别 {c}:")
            print(f"  样本数: {len(X_c)}")
            print(f"  先验概率 P(y={c}) = {self.class_priors[c]:.4f}")
            print(f"  特征均值: {self.means[c]}")
            print(f"  特征方差: {self.variances[c]}")
            print()

        print("训练完成！\n")

    def _gaussian_pdf(self, x: float, mean: float, var: float) -> float:
        """
        计算高斯概率密度函数值

        P(x|μ,σ²) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

        参数:
            x: 输入值
            mean: 均值 μ
            var: 方差 σ²

        返回:
            概率密度值
        """
        # 添加小量避免除零
        eps = 1e-10
        var = var + eps

        coefficient = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))

        return coefficient * exponent

    def _calculate_posterior(self, x: np.ndarray, c) -> float:
        """
        计算后验概率（未归一化）

        P(y=c|x) ∝ P(y=c) × ∏ P(x_i|y=c)

        参数:
            x: 单个样本的特征向量
            c: 类别

        返回:
            未归一化的后验概率
        """
        # 先验概率
        prior = self.class_priors[c]

        # 似然：每个特征的高斯概率密度的乘积
        likelihood = 1.0
        for i, x_i in enumerate(x):
            pdf = self._gaussian_pdf(x_i, self.means[c][i], self.variances[c][i])
            likelihood *= pdf

        return prior * likelihood

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测每个类别的概率

        参数:
            X: 测试数据，形状 (n_samples, n_features)

        返回:
            概率矩阵，形状 (n_samples, n_classes)
        """
        probas = []

        for x in X:
            # 计算每个类别的后验概率,即每个类别在x出现时的概率
            posteriors = []
            for c in self.classes:
                posterior = self._calculate_posterior(x, c)
                posteriors.append(posterior)

            # 归一化
            total = sum(posteriors)
            posteriors = [p / total for p in posteriors]
            probas.append(posteriors)

        return np.array(probas)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        参数:
            X: 测试数据，形状 (n_samples, n_features)

        返回:
            预测的类别，形状 (n_samples,)
        """
        probas = self.predict_proba(X)
        # np.argmax(probas, axis=1) 返回每个样本的预测类别
        # 例如，如果 probas = [[0.2, 0.1, 0.7], [0.6, 0.2, 0.2]]，则返回 [2, 0]
        return self.classes[np.argmax(probas, axis=1)]


# ============================================
# 示例：使用鸢尾花数据集
# ============================================


def iris_example():
    """
    使用鸢尾花数据集演示高斯朴素贝叶斯
    """
    # 简化版鸢尾花数据（部分样本）
    # 特征：花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度
    # 类别：0=setosa, 1=versicolor, 2=virginica

    # 训练数据
    X_train = np.array(
        [
            # Setosa (类别 0)
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            # Versicolor (类别 1)
            [7.0, 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4.0, 1.3],
            [6.5, 2.8, 4.6, 1.5],
            # Virginica (类别 2)
            [6.3, 3.3, 6.0, 2.5],
            [5.8, 2.7, 5.1, 1.9],
            [7.1, 3.0, 5.9, 2.1],
            [6.3, 2.9, 5.6, 1.8],
            [6.5, 3.0, 5.8, 2.2],
        ]
    )

    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    # 创建并训练分类器
    clf = GaussianNaiveBayes()
    clf.fit(X_train, y_train)

    # 测试数据
    X_test = np.array(
        [
            [5.0, 3.4, 1.5, 0.2],  # 应该是 setosa
            [6.0, 2.9, 4.5, 1.5],  # 应该是 versicolor
            [6.7, 3.0, 5.5, 2.0],  # 应该是 virginica
        ]
    )

    print("=" * 50)
    print("预测结果")
    print("=" * 50)

    # 预测概率
    probas = clf.predict_proba(X_test)
    predictions = clf.predict(X_test)

    class_names = ["Setosa", "Versicolor", "Virginica"]

    for i, (x, pred, proba) in enumerate(zip(X_test, predictions, probas)):
        print(f"\n样本 {i + 1}: {x}")
        print(f"预测类别: {class_names[pred]}")
        print("各类别概率:")
        for j, name in enumerate(class_names):
            bar = "█" * int(proba[j] * 20)
            print(f"  {name:12s}: {proba[j]:.4f} {bar}")


# ============================================
# 与 sklearn 对比验证
# ============================================


def compare_with_sklearn():
    """
    与 sklearn 的实现对比，验证我们的实现是否正确
    """
    try:
        from sklearn.naive_bayes import GaussianNB
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        print("\n" + "=" * 50)
        print("与 sklearn 对比验证")
        print("=" * 50)

        # 加载完整的鸢尾花数据集
        iris = load_iris()
        X, y = iris.data, iris.target

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 我们的实现
        our_clf = GaussianNaiveBayes()
        our_clf.fit(X_train, y_train)
        our_predictions = our_clf.predict(X_test)
        our_accuracy = accuracy_score(y_test, our_predictions)

        # sklearn 的实现
        sklearn_clf = GaussianNB()
        sklearn_clf.fit(X_train, y_train)
        sklearn_predictions = sklearn_clf.predict(X_test)
        sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

        print(f"\n我们的实现准确率: {our_accuracy:.4f}")
        print(f"sklearn 准确率:   {sklearn_accuracy:.4f}")

        if np.allclose(our_accuracy, sklearn_accuracy, atol=0.01):
            print("\n✅ 验证通过！我们的实现与 sklearn 结果一致！")
        else:
            print("\n⚠️ 结果有差异，可能是因为方差计算方式略有不同")

    except ImportError:
        print("\n💡 提示：安装 sklearn 可以进行对比验证")
        print("   pip install scikit-learn")


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    print("\n" + "🌸 " * 20 + "\n")
    print("第二课：高斯朴素贝叶斯分类器")
    print("\n" + "🌸 " * 20 + "\n")

    # 运行鸢尾花示例
    iris_example()

    # 与 sklearn 对比
    compare_with_sklearn()

    print("\n" + "=" * 50)
    print("下一步：运行 03_multinomial_nb.py 学习多项式朴素贝叶斯")
    print("=" * 50)
