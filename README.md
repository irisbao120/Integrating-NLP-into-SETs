# Integrating-NLP-into-SETs
This is the repository for demonstrating technical results of TF-IDF, LDA and BERTopic processing based on our collected data of SETs.
# Replication Package for "Human-in-the-loop AI for Educational Feedback Analysis"

## 1. Overview
This repository contains the data, scripts, and prompts used for the analysis in the study. 

## 2. Data Availability & Ethical Statement
IMPORTANT: Due to institutional ethics restrictions and student privacy, the raw student reflective journals are NOT publicly available. All analyses presented in the paper were performed on de-identified data. The repository contains the model outputs, derived results, and the exact code needed to reproduce the analyses. 

## 3. Software Dependencies
- Python 3.13
- Dependencies are listed in `requirements.txt`

## 4. Directory Structure & Execution Order
Please execute the scripts in the following order:

### Data Preprocessing & Baseline Models
- `tf-idf kmeans.py`: Runs TF-IDF and k-means baseline.
- `lda_baseline.py`: Runs baseline LDA (k=7).
- `bertopic_baseline.py`: Runs baseline BERTopic.
- Outputs: `1_tfidf_kmeans_with_clusters.csv`, `2_lda_topic_keywords_visualization.png`, `3_bertopic_topic_info.xlsx`

### Model Optimization & Sensitivity Analysis
- `lda_optimal number of topics.py`: Finds optimal k for LDA.
- `lda_optimization_grid_search.py`: Runs the 105 hyperparameter combinations for LDA.
- `lda_optimization_results.py`: Processes the grid search results.
- `bertopic_minimal-preprocessing_grid_search`: Runs BERTopic sensitivity analysis.
- Outputs: `4_lda_tuning_results.csv`, `bertopic_minimal_param_search_results.csv`

### Statistical & Design Effect Analysis
- `lda_question and week permutation test.xlsx` (Raw results)
- `topic_prompt_time_assosiation.py`: Runs the chi-square and permutation tests.
- `Results of bootstrap 95%CI.xlsx`: Bootstrap results.

### Generative AI Interpretation
- `Prompts for ChatGPT generation.txt`: Contains the exact prompt used for GenAI interpretation.

### Visualization
- `lda_tsne.py` / `bertopic_tsne.py`: Generates t-SNE visualizations.

## 5. Computational Parameters
- LDA Grid Search: alpha (auto, symmetric, asymmetric, 0.01, 0.1, 0.5, and 1.0), eta (auto, 0.01, 0.1, 0.5, and 1.0), passes (15, 20, 30). Total 105 combinations.
- BERTopic Grid Search: min_cluster_size (3, 5, 7, 10, 15), min_samples (1, 2, 3).
- Random seeds and iterations are documented in the respective scripts.
