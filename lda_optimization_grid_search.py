# -*- coding: utf-8 -*-
import ssl
import re
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def main():
    # 1. 解决SSL验证问题
    try:
        _create_unverified_https_context = ssl._create_unverified_https_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # 2. 下载NLTK资源
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    # 3. 载入数据
    filepath = "/Users/irisbao/LU/学术/学术-DAN/ai-blended learning/SDT+bertopic/Dan-comparison改/学生日志-ENG-1-43/reflection 43-344.xlsx"  # 替换为你的路径
    df = pd.read_excel(filepath, engine='openpyxl')
    texts = df['reflection'].dropna().astype(str).tolist()
    print(f"Preprocessing done, total {len(texts)} texts.")

    # 4. 文本预处理
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    processed_texts = [preprocess_text(t) for t in texts]
    print(f"Preprocessing complete. {len(processed_texts)} processed texts.")

    # 5. 字典和语料
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.85)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    print(f"Dictionary size: {len(dictionary)}, corpus size: {len(corpus)}")

    # 6. 载入参数组合（确保 JSON 文件存在且格式正确）
    with open('/Users/irisbao/LU/学术/学术-DAN/ai-blended learning/SDT+bertopic/Dan-comparison改/optimization of LDA/k=7/parameter_grid.json', 'r', encoding='utf-8') as f:
        param_list = json.load(f)

    # 7. 定义指标函数（仅用coherence）
    def rbo_corrected(list1, list2, p=0.9, depth=None):
        if depth is None:
            depth = min(len(list1), len(list2))
        if depth == 0:
            return 0
        weighted_overlap = 0
        for d in range(1, depth + 1):
            set1 = set(list1[:d])
            set2 = set(list2[:d])
            overlap_d = len(set1.intersection(set2))
            weighted_overlap += overlap_d * (p ** (d - 1))
        rbo_score = (1 - p) * weighted_overlap / depth
        return max(0, min(1, rbo_score))

    def irbo_corrected(topics, top_n=10, p=0.9):
        num_topics = len(topics)
        if num_topics < 2:
            return 1.0
        topic_tops = [topic[:top_n] for topic in topics]
        rbo_scores = []
        for i in range(num_topics):
            for j in range(i+1, num_topics):
                score = rbo_corrected(topic_tops[i], topic_tops[j], p=p, depth=top_n)
                rbo_scores.append(score)
        mean_rbo = np.mean(rbo_scores)
        return max(0, min(1, 1 - mean_rbo))

    # 8. 训练+存结果
    results = []
    for idx, param in enumerate(param_list):
        k = 7  # 固定主题数
        alpha = param['alpha']
        eta = param['eta']
        passes = param.get('passes', 15)
        print(f"Training set {idx+1}/{len(param_list)}: k={k}, alpha={alpha}, eta={eta}, passes={passes}")
        try:
            # 训练LDA模型
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                random_state=42,
                passes=passes,
                alpha=alpha,
                eta=eta
            )
            # 计算coherence
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=processed_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()

            # 提取主题关键词，用于IRBO完整性（虽然我们不用它做排序的依据）
            topics_words = []
            for t in range(k):
                top_words = [word for word, _ in lda_model.show_topic(t, topn=20)]
                topics_words.append(top_words)

            # 仅用coherence指标
            results.append({
                'k': k,
                'alpha': alpha,
                'eta': eta,
                'passes': passes,
                'coherence': coherence_score
            })

            print(f"Finished: coherence={coherence_score:.4f}")

        except Exception as e:
            print(f"Error: {e}")
            continue

    # 转成DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv('lda_coherence_results.csv', index=False)
    print("所有参数调优的结果已保存到 'lda_coherence_results.csv'。")

    # 9. 通过Plotly做交互式图表
    import plotly.express as px

    # 按照coherence排序
    df_sorted = df_results.sort_values('coherence', ascending=False).reset_index()

    # 打散散点图
    fig_scatter = px.scatter(
        df_sorted,
        x=df_sorted.index,
        y='coherence',
        labels={'x': 'Parameter Set (Sorted)', 'coherence': 'Coherence'},
        title='Parameter Set vs. Coherence (Scatter Plot)'
    )
    # 折线图
    fig_line = px.line(
        df_sorted,
        x=df_sorted.index,
        y='coherence',
        labels={'x': 'Parameter Set (Sorted)', 'coherence': 'Coherence'},
        title='Parameter Set vs. Coherence (Line Plot)'
    )
    # 参数关系的散点矩阵（仅示例，具体可以传入参数列）
    # 这里选择显示所有连续参数
    # 转换成数字或类别
    df_pivot = df_sorted.copy()
    # 可选：转换alpha, eta成类别或数值
    # 例如：df_pivot['alpha']是字符串，我们可以保持，但显示可能不够直观
    # 这里直接用
    import plotly.express as px
    fig_pair = px.scatter_matrix(
        df_pivot,
        dimensions=['coherence'],  # 可以加入多个参数来分析关系
        labels={col: col for col in ['coherence']},
        title='Parameter Relations Scatter Matrix (示意)'
    )

    # 输出为交互式HTML
    fig_scatter.write_html('coherence_scatter.html')
    fig_line.write_html('coherence_line.html')
    fig_pair.write_html('parameter_correlation.html')
    print("交互式图表已保存：coherence_scatter.html、coherence_line.html、parameter_correlation.html")

    # 10. 打印最优的参数
    top_result = df_sorted.iloc[0]
    print("\n最高 Coherence 的参数组合：")
    print(top_result)

if __name__ == '__main__':
    main()
