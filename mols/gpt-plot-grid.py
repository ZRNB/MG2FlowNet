
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.font_manager import FontProperties

# def load_data(path):
#     df = pd.read_excel(path)
#     df = df.dropna(subset=['States visited']).copy()
#     df['States visited'] = pd.to_numeric(df['States visited'], errors='coerce')
#     df = df.dropna(subset=['States visited'])
#     df['States visited'] = df['States visited'].astype(int)
#     df = df.sort_values('States visited').reset_index(drop=True)
#     return df

# def series_label_and_xy(df, col):
#     x_all = df['States visited'].values
#     y_all = df[col].values
#     first_cell = y_all[0] if len(y_all) > 0 else None

#     # 判断第一行是不是模型名字
#     if first_cell is not None and not pd.isna(first_cell):
#         try:
#             float(first_cell)
#             is_label_row = False
#         except (ValueError, TypeError):
#             is_label_row = True
#     else:
#         is_label_row = False

#     if is_label_row:
#         label = str(first_cell).strip()
#         x_plot = x_all[1:]
#         y_plot = pd.to_numeric(y_all[1:], errors='coerce').astype(float)
#     else:
#         label = col.strip()
#         x_plot = x_all
#         y_plot = pd.to_numeric(y_all, errors='coerce').astype(float)

#     # 替换 MCTS-GFN → MG2FlowNet
#     if label == 'MCTS-GFN':
#         label = 'MG2FlowNet'

#     mask = ~np.isnan(y_plot)
#     return label, x_plot[mask], y_plot[mask]

# def plot_dataframe(ax, df, ylabel, title, style_map):
#     columns = [c for c in df.columns if c != 'States visited']
#     handles = []
#     labels = []
#     for col in columns:
#         label, x_plot, y_plot = series_label_and_xy(df, col)

#         if label == 'MG2FlowNet':
#             marker, color, ls = ('D', 'red', '-')
#         elif col in style_map:
#             marker, color, ls = style_map[col]
#         else:
#             marker, color, ls = ('o', None, '-')

#         line, = ax.plot(x_plot, y_plot,
#                         marker=marker, color=color, linestyle=ls,
#                         markersize=5, linewidth=1.6,
#                         markevery=max(len(x_plot)//20, 1) if len(x_plot) > 0 else 1,
#                         label=label)
#         handles.append(line)
#         labels.append(label)

#     ax.set_xlabel('States visited', fontsize=16, fontweight='bold')
#     ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
#     ax.set_title(title, fontsize=18, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.6)

#     ax.tick_params(axis='both', labelsize=14, width=1.2)
#     for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
#         tick_label.set_fontweight('bold')

#     return handles, labels

# def main():
#     df_error = load_data('grid_distribution_error.xlsx')
#     df_modes = load_data('grid_num_of_modes.xlsx')

#     style_map = {
#         'TB': ('o', 'black', '--'),
#         'MCTS-GFN': ('D', 'red', '-'),   # 会映射成 MG2FlowNet
#         'MCMC': ('*', 'cyan', '-.'),
#         'PPO': ('p', 'orange', '-'),
#         'RANDOM-TRAJ': ('s', 'purple', '-'),
#     }

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

#     # ✅ 指定横坐标刻度
#     xticks_error = [10000, 30000, 50000, 80000, 100000]
#     xticks_modes = [10000, 20000, 30000, 40000]

#     ax1.set_xticks(xticks_error)
#     ax2.set_xticks(xticks_modes)

#     handles1, labels1 = plot_dataframe(ax1, df_error, ylabel='Distribution error',
#                                        title='', style_map=style_map)

#     handles2, labels2 = plot_dataframe(ax2, df_modes, ylabel='Number of modes',
#                                        title='', style_map=style_map)

#     # ✅ 合并 legend (保证不重复)
#     handles = handles1
#     labels = labels1
#     by_label = dict(zip(labels, handles))

#     font_props = FontProperties(size=14, weight='bold')
#     fig.legend(by_label.values(), by_label.keys(),
#                loc='lower center', ncol=len(by_label),
#                prop=font_props, frameon=False, bbox_to_anchor=(0.5, -0.02))

#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.22, wspace=0.35)  # 为 legend 腾出空间

#     fig.savefig('grid_plot.png', dpi=300, bbox_inches='tight')
#     fig.savefig('grid_plot.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == '__main__':
#     main()
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.font_manager import FontProperties
# from matplotlib import gridspec

# def load_data(path):
#     df = pd.read_excel(path)
#     df = df.dropna(subset=['States visited']).copy()
#     df['States visited'] = pd.to_numeric(df['States visited'], errors='coerce')
#     df = df.dropna(subset=['States visited'])
#     df['States visited'] = df['States visited'].astype(int)
#     df = df.sort_values('States visited').reset_index(drop=True)
#     return df

# def series_label_and_xy(df, col):
#     x_all = df['States visited'].values
#     y_all = df[col].values
#     first_cell = y_all[0] if len(y_all) > 0 else None

#     # 判断第一行是不是模型名字
#     if first_cell is not None and not pd.isna(first_cell):
#         try:
#             float(first_cell)
#             is_label_row = False
#         except (ValueError, TypeError):
#             is_label_row = True
#     else:
#         is_label_row = False

#     if is_label_row:
#         label = str(first_cell).strip()
#         x_plot = x_all[1:]
#         y_plot = pd.to_numeric(y_all[1:], errors='coerce').astype(float)
#     else:
#         label = col.strip()
#         x_plot = x_all
#         y_plot = pd.to_numeric(y_all, errors='coerce').astype(float)

#     # 替换 MCTS-GFN → MG2FlowNet
#     if label == 'MCTS-GFN':
#         label = 'MG2FlowNet'

#     mask = ~np.isnan(y_plot)
#     return label, x_plot[mask], y_plot[mask]

# def plot_dataframe(ax, df, ylabel, title, style_map, ylimit=None):
#     columns = [c for c in df.columns if c != 'States visited']
#     for col in columns:
#         label, x_plot, y_plot = series_label_and_xy(df, col)

#         if label == 'MG2FlowNet':
#             marker, color, ls = ('D', 'red', '-')
#         elif col in style_map:
#             marker, color, ls = style_map[col]
#         else:
#             marker, color, ls = ('o', None, '-')

#         ax.plot(x_plot, y_plot,
#                 marker=marker, color=color, linestyle=ls,
#                 markersize=5, linewidth=1.6,
#                 markevery=max(len(x_plot)//20, 1) if len(x_plot) > 0 else 1,
#                 label=label)

#     ax.set_xlabel('States visited', fontsize=16, fontweight='bold')
#     ax.set_ylabel(ylabel, fontsize=16, fontweight='bold', labelpad=15)
#     ax.set_title(title, fontsize=18, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.6)

#     if ylimit is not None:
#         ax.set_ylim(ylimit)

#     ax.tick_params(axis='both', labelsize=14, width=1.2)
#     for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
#         tick_label.set_fontweight('bold')

# def main():
#     df_error = load_data('grid_distribution_error.xlsx')
#     df_modes = load_data('grid_num_of_modes.xlsx')

#     style_map = {
#         'TB': ('o', 'black', '--'),
#         'MCTS-GFN': ('D', 'red', '-'),   # 会映射成 MG2FlowNet
#         'MCMC': ('*', 'cyan', '-.'),
#         'PPO': ('p', 'orange', '-'),
#         'RANDOM-TRAJ': ('s', 'purple', '-'),
#     }

#     # ✅ 使用 gridspec 手动控制高度比例
#     fig = plt.figure(figsize=(14, 6))
#     gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])  # 上小下大
#     ax1_top = fig.add_subplot(gs[0, 0])
#     ax1_bottom = fig.add_subplot(gs[1, 0], sharex=ax1_top)
#     ax2 = fig.add_subplot(gs[:, 1])  # 右边占满两行

#     # ✅ Distribution Error 拆成上下两部分
#     plot_dataframe(ax1_top, df_error, ylabel='',
#                    title='', style_map=style_map, ylimit=(0.5, 3.5))
#     plot_dataframe(ax1_bottom, df_error, ylabel='Distribution error',
#                    title='', style_map=style_map, ylimit=(0.0, 0.5))

#     # 去掉多余的 x 轴刻度
#     plt.setp(ax1_top.get_xticklabels(), visible=False)

#     # ✅ Number of Modes
#     plot_dataframe(ax2, df_modes, ylabel='Number of modes',
#                    title='', style_map=style_map)

#     # ✅ 指定横坐标刻度
#     xticks_error = [10000, 30000, 50000, 80000, 100000]
#     xticks_modes = [10000, 20000, 30000, 40000]

#     ax1_bottom.set_xticks(xticks_error)
#     ax2.set_xticks(xticks_modes)

#     # ✅ 合并 legend
#     handles, labels = ax1_bottom.get_legend_handles_labels()
#     font_props = FontProperties(size=14, weight='bold')
#     fig.legend(handles, labels,
#                loc='lower center', ncol=len(labels),
#                prop=font_props, frameon=False, bbox_to_anchor=(0.5, -0.02))

#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.22, wspace=0.35, hspace=0.05)

#     fig.savefig('grid_plot_broken.png', dpi=300, bbox_inches='tight')
#     fig.savefig('grid_plot_broken.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == '__main__':
#     main()
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.font_manager import FontProperties
# from matplotlib import gridspec

# def load_data(path):
#     df = pd.read_excel(path)
#     df = df.dropna(subset=['States visited']).copy()
#     df['States visited'] = pd.to_numeric(df['States visited'], errors='coerce')
#     df = df.dropna(subset=['States visited'])
#     df['States visited'] = df['States visited'].astype(int)
#     df = df.sort_values('States visited').reset_index(drop=True)
#     return df

# def series_label_and_xy(df, col):
#     x_all = df['States visited'].values
#     y_all = df[col].values
#     first_cell = y_all[0] if len(y_all) > 0 else None

#     # 判断第一行是不是模型名字
#     if first_cell is not None and not pd.isna(first_cell):
#         try:
#             float(first_cell)
#             is_label_row = False
#         except (ValueError, TypeError):
#             is_label_row = True
#     else:
#         is_label_row = False

#     if is_label_row:
#         label = str(first_cell).strip()
#         x_plot = x_all[1:]
#         y_plot = pd.to_numeric(y_all[1:], errors='coerce').astype(float)
#     else:
#         label = col.strip()
#         x_plot = x_all
#         y_plot = pd.to_numeric(y_all, errors='coerce').astype(float)

#     # 替换 MCTS-GFN → MG2FlowNet
#     if label == 'MCTS-GFN':
#         label = 'MG2FlowNet'

#     mask = ~np.isnan(y_plot)
#     return label, x_plot[mask], y_plot[mask]

# def plot_dataframe(ax, df, ylabel, title, style_map, ylimit=None, add_legend=False):
#     columns = [c for c in df.columns if c != 'States visited']
#     for col in columns:
#         label, x_plot, y_plot = series_label_and_xy(df, col)

#         if label == 'MG2FlowNet':
#             marker, color, ls = ('D', 'red', '-')
#         elif col in style_map:
#             marker, color, ls = style_map[col]
#         else:
#             marker, color, ls = ('o', None, '-')

#         ax.plot(x_plot, y_plot,
#                 marker=marker, color=color, linestyle=ls,
#                 markersize=5, linewidth=1.6,
#                 markevery=max(len(x_plot)//20, 1) if len(x_plot) > 0 else 1,
#                 label=label)

#     ax.set_xlabel('States visited', fontsize=16, fontweight='bold')
#     ax.set_ylabel(ylabel, fontsize=16, fontweight='bold', labelpad=15)
#     ax.set_title(title, fontsize=18, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.6)

#     if ylimit is not None:
#         ax.set_ylim(ylimit)

#     ax.tick_params(axis='both', labelsize=14, width=1.2)
#     for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
#         tick_label.set_fontweight('bold')

#     # ✅ 在每张图内部加 legend
#     if add_legend:
#         font_props = FontProperties(size=12, weight='bold')
#         ax.legend(loc='upper right', prop=font_props, frameon=False)

# def main():
#     df_error = load_data('grid_distribution_error.xlsx')
#     df_modes = load_data('grid_num_of_modes.xlsx')

#     style_map = {
#         'TB': ('o', 'black', '--'),
#         'MCTS-GFN': ('D', 'red', '-'),   # 会映射成 MG2FlowNet
#         'MCMC': ('*', 'cyan', '-.'),
#         'PPO': ('p', 'orange', '-'),
#         'RANDOM-TRAJ': ('s', 'purple', '-'),
#     }

#     # ✅ 使用 gridspec 手动控制高度比例
#     fig = plt.figure(figsize=(14, 6))
#     gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])
#     ax1_top = fig.add_subplot(gs[0, 0])
#     ax1_bottom = fig.add_subplot(gs[1, 0], sharex=ax1_top)
#     ax2 = fig.add_subplot(gs[:, 1])

#     # Distribution Error 拆成上下两部分
#     plot_dataframe(ax1_top, df_error, ylabel='',
#                    title='', style_map=style_map, ylimit=(0.5, 3.5))
#     plot_dataframe(ax1_bottom, df_error, ylabel='Distribution error',
#                    title='', style_map=style_map, ylimit=(0.0, 0.5), add_legend=True)

#     plt.setp(ax1_top.get_xticklabels(), visible=False)

#     # Number of Modes
#     plot_dataframe(ax2, df_modes, ylabel='Number of modes',
#                    title='', style_map=style_map, add_legend=True)

#     # 指定横坐标刻度
#     xticks_error = [10000, 30000, 50000, 80000, 100000]
#     xticks_modes = [10000, 20000, 30000, 40000]

#     ax1_bottom.set_xticks(xticks_error)
#     ax2.set_xticks(xticks_modes)

#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.15, wspace=0.35, hspace=0.05)

#     fig.savefig('grid_plot.png', dpi=300, bbox_inches='tight')
#     fig.savefig('grid_plot.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == '__main__':
#     main()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec

def load_data(path):
    df = pd.read_excel(path)
    df = df.dropna(subset=['States visited']).copy()
    df['States visited'] = pd.to_numeric(df['States visited'], errors='coerce')
    df = df.dropna(subset=['States visited'])
    df['States visited'] = df['States visited'].astype(int)
    df = df.sort_values('States visited').reset_index(drop=True)
    return df

def series_label_and_xy(df, col):
    x_all = df['States visited'].values
    y_all = df[col].values
    first_cell = y_all[0] if len(y_all) > 0 else None

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

    if label == 'MCTS-GFN':
        label = 'MG2FlowNet'

    mask = ~np.isnan(y_plot)
    return label, x_plot[mask], y_plot[mask]

def plot_dataframe(ax, df, ylabel, title, style_map, ylimit=None, add_legend=False, legend_loc='upper right'):
    columns = [c for c in df.columns if c != 'States visited']
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
                markevery=max(len(x_plot)//20, 1) if len(x_plot) > 0 else 1,
                label=label)

    ax.set_xlabel('States visited', fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    if ylimit is not None:
        ax.set_ylim(ylimit)

    ax.tick_params(axis='both', labelsize=14, width=1.2)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')

    # ✅ 单独设置 legend 的位置
    if add_legend:
        font_props = FontProperties(size=14, weight='bold')
        ax.legend(loc=legend_loc, prop=font_props, frameon=False)

def main():
    df_error = load_data('grid_distribution_error.xlsx')
    df_modes = load_data('grid_num_of_modes.xlsx')

    style_map = {
        'TB': ('o', 'black', '--'),
        'MCTS-GFN': ('D', 'red', '-'),
        'MCMC': ('*', 'cyan', '-.'),
        'PPO': ('p', 'orange', '-'),
        'RANDOM-TRAJ': ('s', 'purple', '-'),
    }

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])
    ax1_top = fig.add_subplot(gs[0, 0])
    ax1_bottom = fig.add_subplot(gs[1, 0], sharex=ax1_top)
    ax2 = fig.add_subplot(gs[:, 1])

    # 左边上下两部分
    plot_dataframe(ax1_top, df_error, ylabel='',
                   title='', style_map=style_map, ylimit=(0.5, 3.5))
    plot_dataframe(ax1_bottom, df_error, ylabel='Distribution error',
                   title='', style_map=style_map, ylimit=(0.0, 0.5),
                   add_legend=True, legend_loc='upper right')

    plt.setp(ax1_top.get_xticklabels(), visible=False)

    # 右边
    plot_dataframe(ax2, df_modes, ylabel='Number of modes',
                   title='', style_map=style_map,
                   add_legend=True, legend_loc='upper left')

    # 横坐标刻度
    xticks_error = [10000, 30000, 50000, 80000, 100000]
    xticks_modes = [10000, 20000, 30000, 40000]

    ax1_bottom.set_xticks(xticks_error)
    ax2.set_xticks(xticks_modes)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, wspace=0.35, hspace=0.05)

    fig.savefig('grid_plot.png', dpi=300, bbox_inches='tight')
    fig.savefig('grid_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
