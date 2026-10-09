# -*- coding: utf-8 -*-
"""
简化版 BERTopic 主题建模脚本
与 LDA 和 TF-IDF 保持一致的预处理流程
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import json

# 设置中文字体支持 (用于图表)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 下载 NLTK 数据
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def main():
    print("=== Step 1: Data Loading and Preprocessing ===")

    # 加载数据
    file_path = "/Users/irisbao/Desktop/AI-blended questions/学生日志-ENG/AI-blended questions-sum.xlsx"
    df = pd.read_excel(file_path, engine='openpyxl')

    # 提取 reflection 列数据
    texts = df['reflection'].dropna().tolist()
    print(f"Successfully extracted {len(texts)} reflection texts")

    # 文本预处理函数（与 LDA 和 TF-IDF 保持一致）
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

        return ' '.join(tokens)  # 返回字符串格式

    # 应用预处理
    processed_texts = [preprocess_text(text) for text in texts]
    print("Text preprocessing completed")

    print("\n=== Step 2: Prepare BERTopic Model ===")

    # 设置随机种子以确保可复现性
    seed = 42
    np.random.seed(seed)

    # 初始化 BERTopic 模型
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    topic_model = BERTopic(
        language="english",
        top_n_words=10,  # 每个主题提取 20 个关键词
        calculate_probabilities=True,
        verbose=True
    )

    print("BERTopic model prepared with consistent preprocessing")

    print("\n=== Step 3: Train BERTopic Model ===")

    # 训练模型
    topics, probabilities = topic_model.fit_transform(processed_texts)
    print("BERTopic model training completed")

    # 获取主题信息
    topic_info = topic_model.get_topic_info()
    print(f"\nNumber of topics identified: {len(topic_info) - 1}")  # 减去 -1 主题(离群点)
    print("Topic distribution:")
    print(topic_info)

    # 保存主题信息
    topic_info.to_csv('bertopic_topic_info.csv', index=False)
    print("Topic info saved to 'bertopic_topic_info.csv'")

    print("\n=== Step 4: Extract Topic Keywords ===")

    # 创建关键词 DataFrame
    keywords_data = []
    for topic_id in range(len(topic_info)):
        if topic_id != -1:  # 排除离群点主题
            topic_keywords = topic_model.get_topic(topic_id)
            if topic_keywords:
                for i, (word, score) in enumerate(topic_keywords):
                    keywords_data.append({
                        'topic_id': topic_id,
                        'rank': i + 1,
                        'keyword': word,
                        'score': score
                    })

    keywords_df = pd.DataFrame(keywords_data)
    keywords_df.to_csv('bertopic_topic_keywords.csv', index=False)
    print("Topic keywords saved to 'bertopic_topic_keywords.csv'")

    # 打印每个主题的前 10 个关键词
    for topic_id in range(len(topic_info)):
        if topic_id != -1:  # 排除离群点主题
            topic_keywords = topic_model.get_topic(topic_id)
            if topic_keywords:
                print(f"\nTopic {topic_id} keywords:")
                for i, (word, score) in enumerate(topic_keywords[:10], 1):
                    print(f"{i:2d}. {word}: {score:.4f}")

    print("\n=== Step 5: Document-Topic Analysis ===")

    # 为每个文档分配主题
    doc_topics = []
    for i, (topic, prob) in enumerate(zip(topics, probabilities)):
        doc_topics.append({
            'doc_id': i,
            'topic': topic,
            'topic_probability': prob[topic] if topic != -1 else 0,
        })

    # 转换为 DataFrame
    topics_df = pd.DataFrame(doc_topics)

    # 统计主题分布
    topic_counts = topics_df['topic'].value_counts().sort_index()

    # 保存文档-主题分配
    topics_df.to_csv('bertopic_document_assignments.csv', index=False)
    print("Document-topic assignments saved to 'bertopic_document_assignments.csv'")

    print("\n=== Step 6: Visualizations ===")

    # 1. 主题分布条形图
    plt.figure(figsize=(12, 6))
    topic_counts_df = topic_info[topic_info['Topic'] != -1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(topic_counts_df)))

    plt.bar(topic_counts_df['Topic'].astype(str), topic_counts_df['Count'], color=colors)
    plt.xlabel('Topic ID', fontsize=12)
    plt.ylabel('Number of Documents', fontsize=12)
    plt.title('BERTopic Distribution', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # 在柱子上添加数值标签
    for i, count in enumerate(topic_counts_df['Count']):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('bertopic_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 每个主题的关键词水平条形图
    n_topics = len(topic_info) - 1
    fig, axes = plt.subplots((n_topics + 1) // 2, 2, figsize=(15, 5 * ((n_topics + 1) // 2)))

    if n_topics <= 2:
        axes = np.array([axes])

    axes = axes.flatten()

    for idx, topic_id in enumerate(range(n_topics)):
        # 获取主题关键词
        topic_keywords = topic_model.get_topic(topic_id)
        if topic_keywords:
            words = [word for word, _ in topic_keywords[:10]]  # 取前10个关键词
            scores = [score for _, score in topic_keywords[:10]]

            # 创建水平条形图
            axes[idx].barh(words, scores, color=plt.cm.Set3(topic_id / n_topics))
            axes[idx].set_title(f'Topic {topic_id}', fontsize=12)
            axes[idx].set_xlabel('Relevance', fontsize=10)

    # 隐藏多余的子图
    for idx in range(n_topics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('bertopic_keywords.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 尝试生成交互式可视化 (HTML格式)
    try:
        # 主题间距离图
        fig = topic_model.visualize_topics()
        fig.write_html("bertopic_topics.html")
        print("Topic distance map saved to 'bertopic_topics.html'")

        # 主题层次聚类图
        fig = topic_model.visualize_hierarchy()
        fig.write_html("bertopic_hierarchy.html")
        print("Topic hierarchy saved to 'bertopic_hierarchy.html'")

        # 主题关键词柱状图 (BERTopic自带)
        fig = topic_model.visualize_barchart(n_words=10)
        fig.update_layout(
            height=800,  # 显著增加图形高度
            width=1000,  # 根据需要增加宽度
            margin=dict(l=150)  # 增加左侧边距，为长关键词留出空间
        )
        fig.write_html("bertopic_barchart.html")
        print("BERTopic barchart saved to 'bertopic_barchart.html'")

        # 文档-主题可视化
        fig = topic_model.visualize_documents(processed_texts)
        fig.write_html("bertopic_documents.html")
        print("Document-topic visualization saved to 'bertopic_documents.html'")

        # 热力图可视化
        fig = topic_model.visualize_heatmap()
        fig.write_html("bertopic_heatmap.html")
        print("Topic heatmap saved to 'bertopic_heatmap.html'")

    except Exception as e:
        print(f"Note: Interactive visualizations could not be generated: {e}")
        print("This is expected if Chrome is not installed on your system.")


    print("\n=== Step 7: Save Model and Summary ===")

    # 保存 BERTopic 模型
    topic_model.save("bertopic_model")
    print("BERTopic model saved to 'bertopic_model'")

    # 创建分析摘要
    # 创建分析摘要 - 修复JSON序列化问题
    summary = {
        'dataset_info': {
            'total_documents': int(len(texts)),  # 转换为Python int
            'preprocessed_documents': int(len(processed_texts))  # 转换为Python int
        },
        'topic_info': {
            'total_topics': int(len(topic_info) - 1),  # 转换为Python int
            'outlier_documents': int(topic_info[topic_info['Topic'] == -1]['Count'].values[0]) if -1 in topic_info[
                'Topic'].values else 0,  # 转换为Python int
        },
        'output_files': [
            'bertopic_model',
            'bertopic_topic_info.csv',
            'bertopic_topic_keywords.csv',
            'bertopic_document_assignments.csv',
            'bertopic_distribution.png',
            'bertopic_keywords.png',
            'bertopic_topics.html',
            'bertopic_hierarchy.html',
            'bertopic_barchart.html',
            'bertopic_documents.html',
            'bertopic_heatmap.html',
        ]
    }

    with open('bertopic_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Analysis summary saved to 'bertopic_analysis_summary.json'")

    print("\n=== BERTopic Analysis Complete ===")
    print("All results have been saved.")
    print(f"Key findings: {len(topic_info) - 1} topics identified")
    print("Next steps:")
    print("1. Review the topic keywords and assign meaningful names to each topic")
    print("2. Examine the document-topic assignments for deeper insights")
    print("3. Compare with LDA results to identify consistent patterns")


if __name__ == '__main__':
    main()