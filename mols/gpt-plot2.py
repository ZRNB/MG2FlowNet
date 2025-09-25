
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
def load_data(path):
    df = pd.read_excel(path)
    # Only use rows where 'round' is present; do NOT forward-fill
    df = df.dropna(subset=['round']).copy()
    # Convert round to numeric if possible; drop non-numeric
    df['round'] = pd.to_numeric(df['round'], errors='coerce')
    df = df.dropna(subset=['round'])
    df['round'] = df['round'].astype(int)
    df = df.sort_values('round').reset_index(drop=True)
    return df

def series_label_and_xy(df, col):
    # Decide the display label for a column:
    # - If the first cell is a non-numeric string, use it as the label,
    #   and use data starting from row 1.
    # - Otherwise, use the column name as label and all rows as data.
    # X values follow the same slicing as Y to keep alignment.

    x_all = df['round'].values
    y_all = df[col].values

    first_cell = y_all[0] if len(y_all) > 0 else None

    is_label_row = False
    if first_cell is not None and not pd.isna(first_cell):
        try:
            float(first_cell)
        except (ValueError, TypeError):
            is_label_row = True

    if is_label_row:
        label = str(first_cell)
        x_plot = x_all[1:]
        y_plot = pd.to_numeric(y_all[1:], errors='coerce').astype(float)
    else:
        label = col
        x_plot = x_all
        y_plot = pd.to_numeric(y_all, errors='coerce').astype(float)

    if label.strip() == 'MCTS-GFN':
        label = 'MG2FlowNet'

    mask = ~np.isnan(y_plot)
    x_plot = x_plot[mask]
    y_plot = y_plot[mask]

    return label, x_plot, y_plot

def plot_dataframe(ax, df, ylabel, title, style_map):
    columns = [c for c in df.columns if c != 'round']
    for col in columns:
        label, x_plot, y_plot = series_label_and_xy(df, col)

        if label == 'MG2FlowNet':
            marker, color, ls = ('D', 'red', '-')
        elif col in style_map:
            marker, color, ls = style_map[col]
        else:
            marker, color, ls = ('o', None, '-')

        ax.plot(x_plot, y_plot,
                marker=marker, color=color, linestyle=ls,
                markersize=5, linewidth=1.6,
                markevery=max(len(x_plot)//10, 1) if len(x_plot) > 0 else 1,
                label=label)

    ax.set_xlabel('Round', fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    font_props = FontProperties(size=14, weight='bold')
    ax.legend(loc='upper left', prop=font_props)
    ax.tick_params(axis='both', labelsize=14, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
def main():
    df75 = load_data('num_of_modes_75.xlsx')
    df80 = load_data('num_of_modes_80.xlsx')

    style_map = {
        'DB': ('o', 'black', '--'),
        'TB': ('s', 'purple', '-'),
        'SubTB': ('*', 'cyan', '-.'),
        'QGFN': ('p', 'orange', '-'),
        'MCTS-GFN': ('D', 'red', '-'),
    }
   

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    plot_dataframe(ax1, df75, ylabel='Number of modes', title='(a) Modes with reward > 7.5', style_map=style_map)
    plot_dataframe(ax2, df80, ylabel='Number of modes', title='(b) Modes with reward > 8.0', style_map=style_map)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, wspace=0.35)  # 增大子图间距
    fig.savefig('modes_plot_labels_from_firstrow.png', dpi=300, bbox_inches='tight')
    fig.savefig('modes_plot_labels_from_firstrow.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
