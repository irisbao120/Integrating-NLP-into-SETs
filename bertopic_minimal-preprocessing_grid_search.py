# -*- coding: utf-8 -*-
"""
BERTopic parameter search on minimally processed text.
Computes C_v coherence, IRBO, outlier ratio, and top words for each configuration.
"""

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
import nltk
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import json

# 下载 NLTK 数据
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 设置中文字体支持（如需）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. IRBO 计算函数
# ============================================================
def rbo(list1, list2, p=0.9, n=10):
    """Ranked-Biased Overlap"""
    list1 = list1[:n]
    list2 = list2[:n]
    rbo_sum = 0.0
    for d in range(1, n + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        overlap = len(set1 & set2)
        rbo_sum += (p ** (d - 1)) * (overlap / d)
    return (1 - p) * rbo_sum


def irbo(list1, list2, p=0.9, n=10):
    """Inverted Ranked-Biased Overlap (1 - RBO)"""
    return 1.0 - rbo(list1, list2, p=p, n=n)


def compute_mean_irbo(topics_words, p=0.9, n=10):
    """Compute mean IRBO across all topic pairs."""
    n_topics = len(topics_words)
    if n_topics < 2:
        return np.nan
    irbo_vals = []
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            irbo_vals.append(irbo(topics_words[i], topics_words[j], p=p, n=n))
    return np.mean(irbo_vals)


# ============================================================
# 2. 主流程
# ============================================================
def main():
    print("=== Step 1: Data Loading and Minimal Preprocessing ===")

    # 修改为你的数据路径
    file_path = "/Users/irisbao/Desktop/AI-blended questions/学生日志-ENG/AI-blended questions-sum.xlsx"
    df = pd.read_excel(file_path, engine='openpyxl')

    # 提取 reflection 列
    texts = df['reflection'].dropna().tolist()
    print(f"Successfully extracted {len(texts)} reflection texts")

    # ---------- 最小预处理 ----------
    def minimal_preprocess(text):
        text = str(text).strip()
        # 将多个空白字符（包括换行）替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text

    processed_texts = [minimal_preprocess(t) for t in texts]
    print("Minimal preprocessing completed (punctuation, casing, stopwords, inflections preserved).")

    # ---------- 为 C_v coherence 准备分词文本 ----------
    def tokenize_for_coherence(text):
        tokens = word_tokenize(text.lower())
        # 去除纯标点
        tokens = [t for t in tokens if t not in string.punctuation and not all(c in string.punctuation for c in t)]
        return tokens

    tokenized_docs = [tokenize_for_coherence(t) for t in processed_texts]
    print("Tokenized documents prepared for coherence calculation.")

    print("\n=== Step 2: Precompute Embeddings ===")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(processed_texts, show_progress_bar=True)
    print("Embeddings computed.")

    print("\n=== Step 3: Parameter Search ===")
    min_cluster_sizes = [3, 5, 7, 10, 15]
    min_samples_list = [1, 2, 3]

    results = []

    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            print(f"\nRunning min_cluster_size={min_cluster_size}, min_samples={min_samples}")

            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=False
            )

            topic_model = BERTopic(
                language="english",
                top_n_words=10,
                calculate_probabilities=False,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=False
            )

            topics, _ = topic_model.fit_transform(processed_texts, embeddings=embeddings)

            topic_info = topic_model.get_topic_info()
            n_topics = len(topic_info) - 1  # 排除 -1
            outlier_count = topic_info[topic_info['Topic'] == -1]['Count'].values[0] if -1 in topic_info['Topic'].values else 0
            outlier_ratio = outlier_count / len(processed_texts)

            # 提取每个主题的 top words
            topics_words = []
            for topic_id in range(n_topics):
                topic_keywords = topic_model.get_topic(topic_id)
                if topic_keywords:
                    words = [word for word, _ in topic_keywords[:10]]
                    topics_words.append(words)

            # 计算 C_v 和 IRBO
            if n_topics >= 2:
                dictionary = Dictionary(tokenized_docs)
                coherence_model = CoherenceModel(
                    topics=topics_words,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                cv_score = coherence_model.get_coherence()
                mean_irbo = compute_mean_irbo(topics_words, p=0.9, n=10)
            else:
                cv_score = np.nan
                mean_irbo = np.nan

            # 格式化 top words 为字符串
            top_words_str = "; ".join(
                [f"T{i}: {', '.join(words)}" for i, words in enumerate(topics_words)]
            )

            results.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_topics': n_topics,
                'outlier_count': outlier_count,
                'outlier_ratio': outlier_ratio,
                'cv_coherence': cv_score,
                'mean_irbo': mean_irbo,
                'top_words': top_words_str
            })

            print(f"  -> n_topics={n_topics}, outlier_ratio={outlier_ratio:.2%}, C_v={cv_score:.4f}, IRBO={mean_irbo:.4f}")

    # ---------- 保存结果 ----------
    results_df = pd.DataFrame(results)
    results_df.to_csv('bertopic_minimal_param_search_results.csv', index=False)

    print("\n=== Parameter Search Complete ===")
    print("Summary table:")
    print(results_df[['min_cluster_size', 'min_samples', 'n_topics', 'outlier_ratio', 'cv_coherence', 'mean_irbo']].to_string(index=False))
    print("\nFull results (including top words) saved to 'bertopic_minimal_param_search_results.csv'")

    # 可选：保存最佳配置的模型（按 C_v 最大且主题数合理）
    # 这里不自动选择，留给你根据表格手动决定。


if __name__ == '__main__':
    main()