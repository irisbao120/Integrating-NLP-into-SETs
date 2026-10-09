# -*- coding: utf-8 -*-
"""
LDA Topic Modeling with k=7 with Coherence and IRBO Evaluation
Modified to output document-topic distribution for each response
"""

import ssl
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import nltk

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Corrected RBO and IRBO calculation functions
def rbo_corrected(list1, list2, p=0.9, depth=None):
    """
    Calculate Rank-Biased Overlap (RBO) between two ranked lists using the correct formula.
    """
    if depth is None:
        depth = min(len(list1), len(list2))

    if depth == 0:
        return 0

    # Calculate overlap at each depth
    weighted_overlap = 0

    for d in range(1, depth + 1):
        # Get the sets of items up to depth d
        set1 = set(list1[:d]) if d <= len(list1) else set(list1)
        set2 = set(list2[:d]) if d <= len(list2) else set(list2)

        # Calculate overlap at this depth
        overlap_d = len(set1.intersection(set2))

        # Weight and accumulate
        weighted_overlap += overlap_d * (p ** (d - 1))

    # Calculate RBO (simplified but more stable formula)
    rbo_score = (1 - p) * weighted_overlap / depth

    # Ensure the value is between 0 and 1
    return max(0, min(1, rbo_score))

def irbo_corrected(topics, top_n=10, p=0.9):
    """
    Calculate Inverted Rank-Biased Overlap (IRBO) for multiple topics with corrected RBO.
    """
    num_topics = len(topics)
    if num_topics < 2:
        return 1.0

    # Extract top N words for each topic
    topic_tops = [topic[:top_n] for topic in topics]

    # Calculate RBO for all unique topic pairs
    rbo_scores = []
    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            score = rbo_corrected(topic_tops[i], topic_tops[j], p, depth=top_n)
            rbo_scores.append(score)

    # Calculate mean RBO
    mean_rbo = np.mean(rbo_scores) if rbo_scores else 0.0

    # IRBO is 1 minus the mean RBO
    irbo_score = 1 - mean_rbo

    # Ensure the value is between 0 and 1
    return max(0, min(1, irbo_score))

def main():
    print("=== Step 1: Data Loading and Preprocessing ===")

    # Load data
    file_path = "/Users/irisbao/LU/学术/学术-DAN/ai-blended learning/SDT+bertopic/Dan-comparison改/学生日志-ENG-1-43/reflection 43-344.xlsx"
    df = pd.read_excel(file_path, engine='openpyxl')

    # Extract reflection column data
    texts = df['reflection'].dropna().tolist()
    print(f"Successfully extracted {len(texts)} reflection texts")

    # Text preprocessing function
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters

        tokens = word_tokenize(text)

        # Remove stopwords and short words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    # Apply preprocessing
    processed_texts = [preprocess_text(text) for text in texts]
    print("Text preprocessing completed")

    print("\n=== Step 2: Create Dictionary and Corpus ===")

    # Create dictionary and filter extremes
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.85)
    print(f"Dictionary created with {len(dictionary)} words")

    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    print(f"Corpus created with {len(corpus)} documents")

    print("\n=== Step 3: Train LDA Model with k=7 ===")

    # Train LDA model with optimal topic number
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=7,  # Using the optimal topic number
        random_state=42,
        passes=15,  # Increased iterations for better quality
        alpha='auto'
    )

    print("LDA model training completed")

    print("\n=== Step 4: Extract Topic Keywords and Calculate Evaluation Metrics ===")

    # Extract top 20 keywords for each topic
    topic_keywords = {}
    all_topics_words = []  # For IRBO calculation

    for topic_id in range(7):
        keywords = lda_model.show_topic(topic_id, topn=20)
        topic_keywords[topic_id] = [(word, round(prob, 4)) for word, prob in keywords]

        # Extract just the words for IRBO calculation
        words_only = [word for word, prob in keywords]
        all_topics_words.append(words_only)

        print(f"\nTopic {topic_id} keywords:")
        for i, (word, prob) in enumerate(keywords, 1):
            print(f"{i:2d}. {word}: {prob:.4f}")

    # Calculate Coherence Score
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=processed_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    print(f"\nCoherence Score (C_V): {coherence_score:.4f}")

    # Calculate IRBO Score with corrected function
    irbo_score = irbo_corrected(all_topics_words, top_n=10)
    print(f"IRBO Score (corrected): {irbo_score:.4f}")

    # Save topic keywords to CSV
    keywords_data = []
    for topic_id, keywords in topic_keywords.items():
        for rank, (word, prob) in enumerate(keywords, 1):
            keywords_data.append({
                'topic_id': topic_id,
                'rank': rank,
                'keyword': word,
                'probability': prob
            })

    keywords_df = pd.DataFrame(keywords_data)
    keywords_df.to_csv('topic_keywords_top20.csv', index=False)
    print("\nTopic keywords saved to 'topic_keywords_top20.csv'")

    print("\n=== Step 5: Document-Topic Analysis ===")

    # Assign dominant topic to each document
    doc_topics = []
    for i, doc_bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(doc_bow)
        dominant_topic = max(topic_dist, key=lambda x: x[1])
        doc_topics.append({
            'doc_id': i,
            'dominant_topic': dominant_topic[0],
            'confidence': round(dominant_topic[1], 4),
            'topic_distribution': [round(prob, 4) for _, prob in topic_dist]
        })

    # Convert to DataFrame and analyze
    topics_df = pd.DataFrame(doc_topics)

    # Select and reorder columns
    topics_df = topics_df[['doc_id', 'dominant_topic', 'confidence', 'topic_distribution']]

    topic_counts = topics_df['dominant_topic'].value_counts().sort_index()

    # Calculate topic proportions
    topic_proportions = [(count / len(topics_df)) * 100 for count in topic_counts]

    print("\nTopic distribution:")
    for topic_id, count, proportion in zip(topic_counts.index, topic_counts.values, topic_proportions):
        print(f"Topic {topic_id}: {count} documents ({proportion:.1f}%)")

    # Save document-topic assignments
    topics_df.to_csv('document_topic_assignments.csv', index=False)
    print("Document-topic assignments saved to 'document_topic_assignments.csv'")

    print("\n=== Step 6: Visualizations ===")

    # 1. Topic Distribution Bar Chart
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, 7))
    bars = plt.bar(range(7), topic_counts.values, color=colors)

    plt.xlabel('Topic ID', fontsize=12)
    plt.ylabel('Number of Documents', fontsize=12)
    plt.title('Topic Distribution Across Documents', fontsize=14)
    plt.xticks(range(7))
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, topic_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('topic_distribution.pdf', bbox_inches='tight')
    plt.show()

    # 2. Topic Keywords Visualization (Horizontal Bar Chart for each topic)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()

    for topic_id in range(7):
        # Get top 10 keywords for this topic
        keywords = lda_model.show_topic(topic_id, topn=10)
        words = [word for word, _ in keywords]
        probabilities = [prob for _, prob in keywords]

        # Create horizontal bar chart
        axes[topic_id].barh(words, probabilities, color=plt.cm.Set3(topic_id / 7))
        axes[topic_id].set_title(f'Topic {topic_id} - Top Keywords', fontsize=12)
        axes[topic_id].set_xlabel('Probability', fontsize=10)

    # Remove the empty subplot
    fig.delaxes(axes[7])
    plt.tight_layout()
    plt.savefig('topic_keywords_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig('topic_keywords_visualization.pdf', bbox_inches='tight')
    plt.show()

    # 3. Topic Confidence Distribution
    plt.figure(figsize=(10, 6))
    confidence_data = [topics_df[topics_df['dominant_topic'] == i]['confidence'] for i in range(7)]

    plt.boxplot(confidence_data, labels=range(7))
    plt.xlabel('Topic ID', fontsize=12)
    plt.ylabel('Confidence Score', fontsize=12)
    plt.title('Confidence Distribution by Topic', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('topic_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('topic_confidence_distribution.pdf', bbox_inches='tight')
    plt.show()

    # 4. Inter-topic Distance Map (using pyLDAvis)
    print("Generating intertopic distance map (pyLDAvis)...")
    try:
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, 'intertopic_distance_map.html')
        print("Intertopic distance map saved to 'intertopic_distance_map.html'")
    except Exception as e:
        print(f"Error generating intertopic distance map: {e}")

    # 5. Topic Evolution Visualization (if applicable)
    # This would typically require multiple time slices of data
    # For now, we'll create a simple topic correlation heatmap

    # Create topic-document matrix
    doc_topic_matrix = np.zeros((len(corpus), 7))
    for i, doc_bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        for topic_id, prob in topic_dist:
            doc_topic_matrix[i, topic_id] = prob

    # Calculate topic correlation
    topic_correlation = np.corrcoef(doc_topic_matrix.T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(topic_correlation,
                xticklabels=range(7),
                yticklabels=range(7),
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True)
    plt.title('Topic Correlation Heatmap', fontsize=14)
    plt.xlabel('Topic ID', fontsize=12)
    plt.ylabel('Topic ID', fontsize=12)
    plt.tight_layout()
    plt.savefig('topic_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('topic_correlation_heatmap.pdf', bbox_inches='tight')
    plt.show()

    print("\n=== Step 7: Save Model and Results ===")

    # Save LDA model
    lda_model.save('lda_model_k7.gensim')
    print("LDA model saved to 'lda_model_k7.gensim'")

    # Save dictionary and corpus
    dictionary.save('dictionary.gensim')
    with open('corpus.pkl', 'wb') as f:
        import pickle
        pickle.dump(corpus, f)
    print("Dictionary and corpus saved")

    # Create summary report
    summary_report = {
        'model_parameters': {
            'num_topics': 7,
            'passes': 15,
            'alpha': 'auto',
            'random_state': 42
        },
        'dataset_info': {
            'total_documents': len(texts),
            'dictionary_size': len(dictionary),
            'corpus_size': len(corpus)
        },
        'evaluation_metrics': {
            'coherence_score': round(coherence_score, 4),
            'irbo_score': round(irbo_score, 4)
        },
        'topic_distribution': topic_counts.to_dict(),
        'topic_proportions': {f"topic_{i}": f"{p:.1f}%" for i, p in enumerate(topic_proportions)},
        'output_files': [
            'topic_keywords_top20.csv',
            'document_topic_assignments.csv',
            'topic_distribution.png',
            'topic_keywords_visualization.png',
            'topic_confidence_distribution.png',
            'intertopic_distance_map.html',
            'topic_correlation_heatmap.png',
            'lda_model_k7.gensim',
            'dictionary.gensim',
            'corpus.pkl'
        ]
    }

    import json
    with open('lda_analysis_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    print("Analysis summary saved to 'lda_analysis_summary.json'")

    print("\n=== LDA Analysis Complete ===")
    print("All visualizations and results have been saved.")
    print(f"Evaluation Metrics - Coherence: {coherence_score:.4f}, IRBO: {irbo_score:.4f}")
    print("Next steps:")
    print("1. Review the topic keywords and assign meaningful names to each topic")
    print("2. Examine the intertopic distance map to understand topic relationships")
    print("3. Analyze document-topic assignments for deeper insights")
    print("4. Compare these metrics with BERTopic results")

if __name__ == '__main__':
    main()
