import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Reset to default (light) background
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Load CSV data
df = pd.read_csv('num_of_modes_75.xlsx')
df2 = pd.read_csv('num_of_modes_80.xlsx')

# Extract x-axis (Active round) and selected columns
x_full = df['#Modes (R>7.5)'].values
columns_to_plot = [
    'DB',
    'TB',
    'LSGFN',
    'CBGFN',
    'randGFN',
    'QGFN',
    'SUBTB',
    'Teacher'
]
labels = ['DB', 'TB', 'LSGFN', 'CBGFN', 'RandGFN', 'QGFN', 'SUBTB', 'Teacher']

# Select 10 evenly spaced indices
num_points = 10
indices = np.linspace(0, len(x_full) - 1, num_points, dtype=int)

# Downsample x-axis and data
x = x_full[indices]
print(indices, x, df2['Teacher'].values[indices])
data = [df[col].values[indices] for col in columns_to_plot]
data2 = [df2[col].values[indices] for col in columns_to_plot]

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

# Define line styles, markers, and colors
lines = [
    (data[0], 'o', 'black', '--', labels[0]),
    (data[1], 's', 'purple', '-', labels[1]),
    (data[2], '^', 'red', '-.', labels[2]),
    (data[3], 'D', 'blue', ':', labels[3]),
    (data[4], 'v', 'green', '--', labels[4]),
    (data[5], 'p', 'orange', '-', labels[5]),
    (data[6], '*', 'cyan', '-.', labels[6]),
    (data[7], '*', 'grey', '-.', labels[7]),
]

lines2 = [
    (data2[0], 'o', 'black', '--', labels[0]),
    (data2[1], 's', 'purple', '-', labels[1]),
    (data2[2], '^', 'red', '-.', labels[2]),
    (data2[3], 'D', 'blue', ':', labels[3]),
    (data2[4], 'v', 'green', '--', labels[4]),
    (data2[5], 'p', 'orange', '-', labels[5]),
    (data2[6], '*', 'cyan', '-.', labels[6]),
    (data2[7], '*', 'grey', '-.', labels[7]),
]

# Plot on both axes
for acc, marker, color, ls, label in lines:
    ax1.plot(x, acc, marker=marker, color=color, linestyle=ls,
             markersize=4, markevery=1, label=label)
for acc, marker, color, ls, label in lines2:
    ax2.plot(x, acc, marker=marker, color=color, linestyle=ls,
             markersize=4, markevery=1, label=label)

# Calculate and set y-axis limits for each subplot
def set_dynamic_ylim(ax, data_list, margin=0.05):
    all_data = np.concatenate([d for d in data_list])
    y_min, y_max = np.nanmin(all_data), np.nanmax(all_data)
    y_range = y_max - y_min
    y_min -= y_range * margin
    y_max += y_range * margin
    if y_min < 0 and all(all_data >= 0):
        y_min = 0
    ax.set_ylim(y_min, y_max)

# Apply dynamic y-axis limits
set_dynamic_ylim(ax1, data)
set_dynamic_ylim(ax2, data2)

# Set titles and labels
ax1.set_xlabel('Active Round', fontsize=12)
ax2.set_xlabel('Active Round', fontsize=12)
# ax1.set_ylabel('Number of Modes', fontsize=12)
# ax2.set_ylabel('Average Reward', fontsize=12)

# Add (a), (b) labels
ax1.text(0.5, -0.18, '(a) Number of modes with reward > 7.5', fontsize=12, transform=ax1.transAxes, ha='center', va='top')
ax2.text(0.5, -0.18, '(b) Number of modes with reward > 8', fontsize=12, transform=ax2.transAxes, ha='center', va='top')

# Add grid
ax1.grid(True, linestyle='--', alpha=0.6)
ax2.grid(True, linestyle='--', alpha=0.6)

# Create shared legend below subplots

fig.legend([l[4] for l in lines], loc='lower center', ncol=8, fontsize=10,
           bbox_to_anchor=(0.5, 0.05))

# Adjust layout
fig.tight_layout()
fig.subplots_adjust(bottom=0.3)

# Save figure
plt.savefig('performance_plots_from_csv_10_points_two_subplots.pdf', dpi=300, bbox_inches='tight')
plt.show()
