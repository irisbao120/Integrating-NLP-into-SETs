# -*- coding: utf-8 -*-
"""
TF-IDF Analysis for Student Reflection Journals
加入K-means聚类，多次运行取平均结果，强制8个簇
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.cluster import KMeans
from scipy.stats import mode

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 设置中文字体支持 (用于图表显示)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 初始化词形还原器
lemmatizer = WordNetLemmatizer()

# 加载数据
file_path = "/Users/irisbao/LU/学术/学术-DAN/ai-blended learning/SDT+bertopic/Dan-comparison改/学生日志-ENG-1-43/reflection 43-344.xlsx"
df = pd.read_excel(file_path)

# 检查数据
print(f"数据集形状: {df.shape}")
print("\n前几行数据:")
print("\n列名:")
print(df.columns.tolist())

# 假设文本数据在名为'text'或'response'的列中，请根据实际情况调整
# 如果列名不同，请修改下面的变量
text_column = 'reflection'  # 可能需要更改为您的实际列名

# 文本预处理函数
def preprocess_text(text):
    if pd.isna(text):
        return ""

    # 转换为小写
    text = text.lower()

    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 分词
    tokens = word_tokenize(text)

    # 移除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    # 词形还原
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

# 应用文本预处理
print("\n正在预处理文本...")
df['processed_text'] = df[text_column].apply(preprocess_text)

# 计算TF-IDF
print("计算TF-IDF...")
vectorizer = TfidfVectorizer(max_features=100, max_df=0.85, min_df=2)
tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

# K-means聚类
print("K-means聚类...")
num_clusters = 7  # 与人工编码主题数保持一致
num_runs = 10  # 运行次数

cluster_labels = np.zeros((len(df), num_runs), dtype=int)  # 存储每次运行的簇标签

for i in range(num_runs):
    print(f"运行 K-means 第 {i+1} 次...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=i, n_init=10)  # 每次设置不同的random_state
    cluster_labels[:, i] = kmeans.fit_predict(tfidf_matrix)

# 计算平均簇标签 (选择最常见的标签)
print("计算平均簇标签...")
average_cluster = mode(cluster_labels, axis=1)[0].flatten()
df['tfidf_cluster'] = average_cluster.astype(int)

# 显示簇标签分布
print("\n簇标签分布:")
print(df['tfidf_cluster'].value_counts())

# 检查簇的数量
if len(np.unique(df['tfidf_cluster'])) != num_clusters:
    print(f"\n警告：K-means 聚类结果不是 {num_clusters} 个簇，请检查数据和参数设置！")

# 保存结果到CSV文件
df.to_csv('tfidf_analysis_results_with_clusters.csv', index=False)
print("\n结果已保存到 'tfidf_analysis_results_with_clusters.csv'")

print("\n分析完成!")
