import ssl
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from gensim import corpora
from gensim.models import LdaModel
import plotly.graph_objs as go
import plotly.offline as pyo

# 解决SSL证书问题
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 下载NLTK数据
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 读入数据
file_path = "/Users/irisbao/LU/学术/学术-DAN/ai-blended learning/SDT+bertopic/Dan-comparison改/学生日志-ENG-1-43/reflection 43-344.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
texts = df['reflection'].dropna().tolist()
print(f"成功提取 {len(texts)} 条反思文本")


# 文本预处理
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


processed_texts = [preprocess_text(t) for t in texts]
print("文本预处理完成")

# 构建字典与语料
dictionary = corpora.Dictionary(processed_texts)
dictionary.filter_extremes(no_below=2, no_above=0.85)
corpus = [dictionary.doc2bow(text) for text in processed_texts]
print(f"字典词数：{len(dictionary)}，语料数：{len(corpus)}")

# 训练LDA模型
num_topics = 7
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=15, alpha='auto')
print("LDA模型训练完成")

# 获取每个文档的主题分布
doc_topic_distributions = []
for doc in corpus:
    topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
    doc_topic_distributions.append([prob for _, prob in topic_dist])

doc_topic_distributions = np.array(doc_topic_distributions)

# 获取每个文档的主要主题
dominant_topics = np.argmax(doc_topic_distributions, axis=1)
print("文档主题分配完成")

# 打印每个主题的关键词
print("\n=== 主题关键词 ===")
for topic_id in range(num_topics):
    topic_words = lda_model.show_topic(topic_id, topn=10)
    print(f"\n主题 {topic_id}:")
    for i, (word, prob) in enumerate(topic_words, 1):
        print(f"  {i:2d}. {word}: {prob:.4f}")

# 保存主题关键词到CSV
keywords_data = []
for topic_id in range(num_topics):
    topic_words = lda_model.show_topic(topic_id, topn=10)
    for i, (word, prob) in enumerate(topic_words):
        keywords_data.append({
            'topic_id': topic_id,
            'rank': i + 1,
            'keyword': word,
            'score': prob
        })

keywords_df = pd.DataFrame(keywords_data)
keywords_df.to_csv('lda_topic_keywords.csv', index=False)
print("主题关键词已保存到 'lda_topic_keywords.csv'")

# 保存文档-主题分配
doc_assignments = []
for i, (doc, dominant_topic) in enumerate(zip(corpus, dominant_topics)):
    doc_assignments.append({
        'doc_id': i,
        'dominant_topic': dominant_topic,
        'topic_probability': doc_topic_distributions[i, dominant_topic]
    })

doc_assignments_df = pd.DataFrame(doc_assignments)
doc_assignments_df.to_csv('lda_document_assignments.csv', index=False)
print("文档-主题分配已保存到 'lda_document_assignments.csv'")

# 主题分布统计
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
print("\n=== 主题分布 ===")
for topic_id in range(num_topics):
    count = topic_counts.get(topic_id, 0)
    print(f"主题 {topic_id}: {count} 个文档")

# 可视化主题分布
plt.figure(figsize=(10, 6))
colors = plt.cm.Set3(np.linspace(0, 1, num_topics))
plt.bar(range(num_topics), [topic_counts.get(i, 0) for i in range(num_topics)], color=colors)
plt.xlabel('主题 ID')
plt.ylabel('文档数量')
plt.title('LDA 主题分布')
plt.xticks(range(num_topics))
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, count in enumerate([topic_counts.get(i, 0) for i in range(num_topics)]):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('lda_topic_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 生成 t-SNE 3D 可视化 ===")

# 使用文档-主题分布作为特征进行t-SNE降维
tsne_3d = TSNE(n_components=3, random_state=42,
               perplexity=min(30, len(doc_topic_distributions) - 1),
               n_iter=1000)
embeddings_3d = tsne_3d.fit_transform(doc_topic_distributions)

print(f"t-SNE 降维完成，形状: {embeddings_3d.shape}")

# 使用与BERTopic相同的颜色序列（前7种颜色）
colors_7 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2'
]

# 创建Plotly图形
fig = go.Figure()

# 为每个主题添加散点
for topic_id in range(num_topics):
    topic_mask = dominant_topics == topic_id
    if np.any(topic_mask):
        # 获取该主题的文档索引
        topic_indices = np.where(topic_mask)[0]

        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[topic_indices, 0],
            y=embeddings_3d[topic_indices, 1],
            z=embeddings_3d[topic_indices, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=colors_7[topic_id],
                opacity=0.8
            ),
            name=f'Topic {topic_id}',
            text=[f"Topic {topic_id}<br>概率: {doc_topic_distributions[i, topic_id]:.3f}<br>{texts[i][:100]}..."
                  for i in topic_indices],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

# 更新布局
fig.update_layout(
    title={
        'text': 'LDA t-SNE 3D Visualization',
        'x': 0.5,
        'xanchor': 'center'
    },
    scene=dict(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        zaxis_title='t-SNE Dimension 3'
    ),
    width=1000,
    height=800,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# 保存为HTML文件
pyo.plot(fig, filename='lda_tsne_3d.html', auto_open=False)
print("LDA t-SNE 3D 可视化已保存到 'lda_tsne_3d.html'")

print("\n=== LDA 分析完成 ===")
print("输出文件:")
print("1. lda_topic_keywords.csv - 主题关键词")
print("2. lda_document_assignments.csv - 文档主题分配")
print("3. lda_topic_distribution.png - 主题分布图")
print("4. lda_tsne_3d.html - t-SNE 3D 可视化")