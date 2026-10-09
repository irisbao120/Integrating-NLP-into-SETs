import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# 1. 读取数据
df = pd.read_excel('/Users/irisbao/LU/学术/学术-DAN/ai-blended learning/SDT+bertopic/Dan-comparison改/computers/lda_reflection 43-344.xlsx', sheet_name='Sheet1')

# 2. 只保留需要的列，并确保类型正确
df = df[['doc_id', 'student_id', 'question_id', 'week_id', 'dominant_topic']].copy()
df['student_id'] = df['student_id'].astype(int)
df['question_id'] = df['question_id'].astype('category')
df['week_id'] = df['week_id'].astype('category')
df['dominant_topic'] = df['dominant_topic'].astype('category')

print('数据形状:', df.shape)
print('学生数:', df['student_id'].nunique())
print('主题分布:')
print(df['dominant_topic'].value_counts().sort_index())

# 3. 观察交叉表
ct_q = pd.crosstab(df['dominant_topic'], df['question_id'])
ct_w = pd.crosstab(df['dominant_topic'], df['week_id'])

print('\nTopic × Question 交叉表:')
print(ct_q)
print('\nTopic × Week 交叉表:')
print(ct_w)

# 4. 观察卡方与渐近 p 值
chi2_q, p_asym_q, dof_q, exp_q = chi2_contingency(ct_q, correction=False)
chi2_w, p_asym_w, dof_w, exp_w = chi2_contingency(ct_w, correction=False)

print(f'\nQuestion: chi2 = {chi2_q:.4f}, dof = {dof_q}, asymptotic p = {p_asym_q:.4f}')
print(f'Week:     chi2 = {chi2_w:.4f}, dof = {dof_w}, asymptotic p = {p_asym_w:.4f}')

# 5. Cramér's V 效应量
def cramers_v(ct):
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

print(f'\nCramér\'s V (Question) = {cramers_v(ct_q):.4f}')
print(f'Cramér\'s V (Week)     = {cramers_v(ct_w):.4f}')

# 6. 学生聚类置换检验
def cluster_permutation_chi2(df, topic_col, factor_col, student_col,
                             n_perm=10000, seed=42):
    """
    topic_col: 主题列名
    factor_col: 要置换的因子列名（question_id 或 week_id）
    student_col: 学生 ID 列名
    """
    rng = np.random.default_rng(seed)

    # 观察卡方
    ct_obs = pd.crosstab(df[topic_col], df[factor_col])
    obs_chi2 = chi2_contingency(ct_obs, correction=False)[0]

    perm_chi2 = np.empty(n_perm)

    for i in range(n_perm):
        df_perm = df.copy()
        # 在每个学生内部置换 factor_col
        df_perm[factor_col] = df_perm.groupby(student_col)[factor_col].transform(
            lambda x: pd.Series(rng.permutation(x.values), index=x.index)
        )
        ct = pd.crosstab(df_perm[topic_col], df_perm[factor_col])
        # 防止极端情况：某行或某列全为 0
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            perm_chi2[i] = np.nan
        else:
            perm_chi2[i] = chi2_contingency(ct, correction=False)[0]

    perm_chi2 = perm_chi2[~np.isnan(perm_chi2)]
    p_perm = (np.sum(perm_chi2 >= obs_chi2) + 1) / (len(perm_chi2) + 1)

    return obs_chi2, p_perm, perm_chi2

# 对 question_id 做置换检验
obs_q, p_q, perm_q = cluster_permutation_chi2(
    df, topic_col='dominant_topic', factor_col='question_id',
    student_col='student_id', n_perm=10000, seed=42
)

# 对 week_id 做置换检验
obs_w, p_w, perm_w = cluster_permutation_chi2(
    df, topic_col='dominant_topic', factor_col='week_id',
    student_col='student_id', n_perm=10000, seed=42
)

print(f'\n=== 置换检验结果 ===')
print(f'Question: obs chi2 = {obs_q:.4f}, permutation p = {p_q:.4f}')
print(f'Week:     obs chi2 = {obs_w:.4f}, permutation p = {p_w:.4f}')

# 7. 标准化残差：看哪些格子贡献大
def std_residuals(ct):
    chi2, p, dof, exp = chi2_contingency(ct, correction=False)
    return (ct - exp) / np.sqrt(exp)

print('\nTopic × Question 标准化残差:')
print(std_residuals(ct_q).round(2))
print('\nTopic × Week 标准化残差:')
print(std_residuals(ct_w).round(2))