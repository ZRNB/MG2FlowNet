import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties

def load_data(path):
    df = pd.read_excel(path)
    df = df.dropna(subset=['round']).copy()
    df['round'] = pd.to_numeric(df['round'], errors='coerce')
    df = df.dropna(subset=['round'])
    df['round'] = df['round'].astype(int)
    df = df.sort_values('round').reset_index(drop=True)
    return df

def series_label_and_xy(df, col):
    x_all = df['round'].values
    y_all = df[col].values
    first_cell = y_all[0] if len(y_all) > 0 else None

    # 判断第一行是不是模型名字
    if first_cell is not None and not pd.isna(first_cell):
        try:
            float(first_cell)
            is_label_row = False
        except (ValueError, TypeError):
            is_label_row = True
    else:
        is_label_row = False

    if is_label_row:
        label = str(first_cell).strip()
        x_plot = x_all[1:]
        y_plot = pd.to_numeric(y_all[1:], errors='coerce').astype(float)
    else:
        label = col.strip()
        x_plot = x_all
        y_plot = pd.to_numeric(y_all, errors='coerce').astype(float)

    mask = ~np.isnan(y_plot)
    return label, x_plot[mask], y_plot[mask]

def plot_dataframe(ax, df, ylabel, title, style_map):
    columns = [c for c in df.columns if c != 'round']
    for col in columns:
        label, x_plot, y_plot = series_label_and_xy(df, col)

        if col in style_map:
            marker, color, ls = style_map[col]
        else:
            marker, color, ls = ('o', None, '-')

        ax.plot(x_plot, y_plot,
                marker=marker, color=color, linestyle=ls,
                markersize=5, linewidth=1.6,
                markevery=max(len(x_plot)//10, 1) if len(x_plot) > 0 else 1,
                label=label)

    ax.set_xlabel('Trajectories', fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    x_min, x_max = ax.get_xlim()
    ax.set_xticks(np.arange(0, int(x_max)+1, 5000))
    font_props = FontProperties(size=14, weight='bold')
    ax.legend(loc='lower right', prop=font_props)
    ax.tick_params(axis='both', labelsize=14, width=1.2)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')

def main():
    df_tanimoto = load_data('tanimoto.xlsx')

    style_map = {
        'TB': ('o', 'black', '--'),
        'MG2FlowNet': ('s', 'red', '-'),
        'MCMC': ('*', 'cyan', '-.'),
        'PPO': ('p', 'orange', '-'),
        'RANDOM-TRAJ': ('D', 'purple', '-'),
    }

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=False)

    plot_dataframe(ax, df_tanimoto, ylabel='Tanimoto similarity',title='', style_map=style_map)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig('tanimoto_plot.png', dpi=300, bbox_inches='tight')
    fig.savefig('tanimoto_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
