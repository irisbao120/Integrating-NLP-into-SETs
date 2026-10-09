import pandas as pd
import plotly.express as px

def plot_param_effect(df, param_name):
    """
    画出某个参数对coherence的影响散点图。
    param_name: 'alpha', 'eta', 或 'passes'
    """
    # 复制一份
    df_plot = df.copy()

    # 根据参数类型调整横轴
    if df_plot[param_name].dtype == object:
        # 如alpha可能是字符串
        # 保持原样，可能需转换
        df_plot[param_name] = df_plot[param_name]
    else:
        # 数值类型
        df_plot[param_name] = df_plot[param_name].astype(str)

    # 生成悬浮信息
    df_plot['Hover_Info'] = df_plot.apply(
        lambda row: f"k=7, alpha={row['alpha']}, eta={row['eta']}, passes={row['passes']}, Coherence={row['coherence']:.3f}", axis=1
    )

    # 画散点图
    fig = px.scatter(
        df_plot,
        x=param_name,
        y='coherence',
        hover_name='Hover_Info',
        title=f'Impact of {param_name} on Coherence',
        labels={param_name: param_name, 'coherence': 'Coherence'}
    )
    fig.update_traces(marker=dict(size=12))
    filename = f'{param_name}_impact_coherence.html'
    fig.write_html(filename)
    print(f"已生成：{filename}")

def main():
    # 载入结果
    df_results = pd.read_csv('lda_coherence_results.csv')

    # 仅考察k=7，即你的所有配置都固定k=7所以不用看k参数变化
    # 推荐观察 alpha、eta、passes
    for param in ['alpha', 'eta', 'passes']:
        plot_param_effect(df_results, param)

if __name__ == '__main__':
    main()
