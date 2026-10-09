# -*- coding: utf-8 -*-
"""
LDA主题建模简明脚本 - 确定最佳主题数
修复了SSL证书和多进程问题
"""

import ssl
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import nltk

# 修复SSL证书问题
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 确保必要的NLTK数据已下载
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def main():
    print("=== 步骤1: 数据加载和预处理 ===")

    # 加载数据
    file_path = "/Users/irisbao/Desktop/AI-blended questions/学生日志-ENG/AI-blended questions-sum.xlsx"
    df = pd.read_excel(file_path, engine='openpyxl')

    # 提取reflection列数据
    texts = df['reflection'].dropna().tolist()
    print(f"成功提取 {len(texts)} 条反思文本")

    # 文本预处理函数
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # 移除非字母字符

        tokens = word_tokenize(text)

        # 移除停用词和短词
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        # 词形还原
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    # 应用预处理
    processed_texts = [preprocess_text(text) for text in texts]
    print("文本预处理完成")

    print("\n=== 步骤2: 创建词典和语料库 ===")

    # 创建词典并过滤极端值
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.85)
    print(f"词典创建完成，包含 {len(dictionary)} 个词语")

    # 创建语料库
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    print(f"语料库创建完成，包含 {len(corpus)} 个文档")

    print("\n=== 步骤3: 确定最佳主题数量 ===")

    # 测试不同的主题数量 (2到10个主题)
    topic_range = range(2, 11)
    coherence_scores = []
    perplexity_scores = []

    print("正在计算不同主题数的模型性能...")
    for num_topics in topic_range:
        # 训练LDA模型
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto'
        )

        # 计算一致性分数 - 禁用多进程以避免错误
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=processed_texts,
            dictionary=dictionary,
            coherence='c_v',
            processes=1  # 禁用多进程
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)

        # 计算困惑度
        perplexity = lda_model.log_perplexity(corpus)
        perplexity_scores.append(perplexity)

        print(f"主题数: {num_topics}, 一致性: {coherence_score:.4f}, 困惑度: {perplexity:.4f}")

    print("\n=== 步骤4: 可视化评估结果 ===")

    # 创建评估结果DataFrame
    eval_df = pd.DataFrame({
        'num_topics': topic_range,
        'coherence': coherence_scores,
        'perplexity': perplexity_scores
    })

    # 绘制一致性分数曲线
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(topic_range, coherence_scores, 'bo-')
    plt.xlabel('Number of topics ')
    plt.ylabel('Coherence')
    plt.title('Coherence vs. Number of topics')
    plt.grid(True)

    # 绘制困惑度曲线
    plt.subplot(1, 2, 2)
    plt.plot(topic_range, perplexity_scores, 'ro-')
    plt.xlabel('Number of topics')
    plt.ylabel('Perplexity (log)')
    plt.title('Perplexity vs. Number of topics')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('topic_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 找出最佳主题数（基于一致性分数）
    best_idx = np.argmax(coherence_scores)
    best_topic_num = topic_range[best_idx]
    print(f"\n建议的最佳主题数量: {best_topic_num}")
    print(f"对应的一致性分数: {coherence_scores[best_idx]:.4f}")

    # 保存评估结果
    eval_df.to_csv('topic_evaluation_results.csv', index=False)
    print("评估结果已保存到 topic_evaluation_results.csv")

    print("\n=== 分析完成 ===")
    print("请查看生成的主题评估图表，并根据一致性分数曲线确定最佳主题数")
    print("确定最佳主题数后，您可以使用该数值运行最终的LDA建模")


if __name__ == '__main__':
    main()