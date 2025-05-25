import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# --- Data for Performance Graphs ---
# These are placeholder values. Replace them with your actual experimental averages.
scramble_depths_labels = ['3', '5', '7', '9', '11'] # Scramble depths tested

# Average Nodes Explored Data (Illustrative)
avg_nodes_bfs = [100, 500, 2000, 8000, 25000]
avg_nodes_dfs = [50, 200, 800, 3000, 9000]
avg_nodes_astar = [20, 50, 150, 400, 1000]

# Average Time Taken Data (seconds) (Illustrative)
avg_time_bfs = [0.01, 0.05, 0.2, 0.8, 2.0]
avg_time_dfs = [0.005, 0.02, 0.1, 0.5, 1.5]
avg_time_astar = [0.002, 0.008, 0.03, 0.1, 0.25]

# --- Plotting Function for Performance Comparison ---
def plot_performance_chart(data_bfs, data_dfs, data_astar, labels, title, ylabel, filename, y_log_scale=False):
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7)) # Wider figure
    rects1 = ax.bar(x - width, data_bfs, width, label='BFS', color='#1f77b4', edgecolor='black') # Blue
    rects2 = ax.bar(x, data_dfs, width, label='DFS (Limit 11)', color='#ff7f0e', edgecolor='black') # Orange
    rects3 = ax.bar(x + width, data_astar, width, label='A*', color='#2ca02c', edgecolor='black') # Green

    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel("Scramble Depth (Number of Moves)", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add data labels on top of bars
    def autolabel(rects, data_values):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            # Format labels: integers for nodes, float for time
            label_text = f'{int(data_values[i])}' if 'Nodes' in ylabel else f'{data_values[i]:.3f}'
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0) # Smaller font for bar labels

    autolabel(rects1, data_bfs)
    autolabel(rects2, data_dfs)
    autolabel(rects3, data_astar)

    if y_log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter()) # Ensure standard number format for log scale

    ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Performance chart '{filename}' saved.")
    # plt.show()

# --- Data for State Space Explosion Graph ---
cube_dimensions = ['2x2x2', '3x3x3', '4x4x4', '5x5x5']
# Exact or commonly cited approximate state counts
state_counts = [
    3.674160e6,    # 2x2x2
    4.3252003e19,  # 3x3x3
    7.4011968e45,  # 4x4x4
    2.8287094e74   # 5x5x5
]

# --- Plotting Function for State Space Explosion ---
def plot_state_space_chart(dimensions, counts, title, ylabel, filename):
    x = np.arange(len(dimensions))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(x, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black') # Different color per bar

    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel("Cube Dimension (N x N x N)", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    
    ax.set_yscale('log') # Logarithmic scale is essential here
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation()) # Scientific notation for log scale
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2e}', va='bottom', ha='center', fontsize=9) # Scientific notation

    ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"State space chart '{filename}' saved.")
    # plt.show()


if __name__ == '__main__':
    # Call table generation functions (already defined above to run if script is main)
    # Example:
    # performance_data_for_table_script = [...] # define or load your data here
    # generate_performance_comparison_table_markdown(performance_data_for_table_script)
    # generate_state_space_table_markdown()


    # --- Generate and Save Performance Graphs ---
    plot_performance_chart(
        avg_nodes_bfs, avg_nodes_dfs, avg_nodes_astar,
        scramble_depths_labels,
        'Average Nodes Explored vs. Scramble Depth (2x2x2 Cube)',
        'Average Nodes Explored (Log Scale)',
        'avg_nodes_explored_chart.png',
        y_log_scale=True # Use log scale for nodes if variance is high
    )

    plot_performance_chart(
        avg_time_bfs, avg_time_dfs, avg_time_astar,
        scramble_depths_labels,
        'Average Time Taken vs. Scramble Depth (2x2x2 Cube)',
        'Average Time Taken (seconds, Log Scale)',
        'avg_time_taken_chart.png',
        y_log_scale=True # Use log scale for time if variance is high
    )

    # --- Generate and Save State Space Explosion Graph ---
    plot_state_space_chart(
        cube_dimensions,
        state_counts,
        'State Space Size vs. Cube Dimension (N x N x N)',
        'Approximate Number of Reachable States (Log Scale)',
        'state_space_explosion_chart.png'
    )
