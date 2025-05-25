import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np # Import numpy for scientific notation handling

def generate_performance_comparison_table_markdown(data):
    """
    Generates a Markdown table for algorithm performance comparison.
    Data should be a list of dictionaries.
    """
    df = pd.DataFrame(data)
    markdown_table = df.to_markdown(index=False)
    print("--- Algorithm Performance Comparison Table (Markdown) ---")
    print(markdown_table)
    print("\n")

def generate_performance_table_png(data, filename="performance_comparison_table.png"):
    """
    Generates a PNG image of the algorithm performance comparison table using matplotlib.
    """
    df = pd.DataFrame(data).copy() # Use a copy to avoid modifying original data inplace

    # Convert 'Nodes Explored' and 'Time Taken (s)' to numeric types for formatting
    # Handle potential non-numeric values
    df['Nodes Explored'] = pd.to_numeric(df['Nodes Explored'], errors='coerce')
    df['Time Taken (s)'] = pd.to_numeric(df['Time Taken (s)'], errors='coerce')

    fig, ax = plt.subplots(figsize=(14, 8)) # Adjusted figure size for better fit
    ax.axis('off')

    # Define colors
    header_color = '#4CAF50'  # A nice green for headers
    row_color_1 = '#e8f5e9'   # Lighter green shade
    row_color_2 = '#c8e6c9'   # Slightly darker green shade
    time_taken_highlight_color = '#FFD700' # Gold for highlighting Time Taken

    # Prepare cell colors based on rows and specific columns
    cell_colors = []
    for i in range(len(df)):
        row_colors = []
        base_row_color = row_color_1 if i % 2 == 0 else row_color_2
        for col_idx, col_name in enumerate(df.columns):
            if col_name == "Time Taken (s)":
                row_colors.append(time_taken_highlight_color)
            else:
                row_colors.append(base_row_color)
        cell_colors.append(row_colors)

    # Convert data for display, applying scientific notation where appropriate
    display_data = df.astype(str).values # Start with string representation of all values

    # Get column indices dynamically
    col_indices = {col_name: i for i, col_name in enumerate(df.columns)}

    if 'Nodes Explored' in col_indices:
        for i, val in enumerate(df['Nodes Explored']):
            if pd.notna(val): # Check for NaN values
                # If value is large, apply scientific notation
                if val >= 1000 or (val < 1 and val != 0): # Example threshold for scientific notation
                    display_data[i, col_indices['Nodes Explored']] = f"{val:.0e}" # Format as integer scientific notation
                else:
                    display_data[i, col_indices['Nodes Explored']] = str(int(val)) # Display as integer

    if 'Time Taken (s)' in col_indices:
        for i, val in enumerate(df['Time Taken (s)']):
            if pd.notna(val):
                display_data[i, col_indices['Time Taken (s)']] = f"{val:.5f}" # Always 5 decimal places for time

    # Create the table
    table = ax.table(cellText=display_data, # Use display_data for cell text
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     cellColours=cell_colors,
                     colColours=[header_color] * len(df.columns)
                    )

    table.auto_set_font_size(False)
    table.set_fontsize(9) # Slightly reduced font size for wider content
    table.scale(1.2, 1.2) # Adjust table scale as needed (width, height)

    # Set column widths. Auto-setting is often good, but manual tweaks might be needed.
    # col_widths = [0.15, 0.1, 0.15, 0.08, 0.12, 0.1] # Example manual widths, adjust as necessary
    # for i, width in enumerate(col_widths):
    #     table.get_celld(0, i).set_width(width) # Set header width

    # Adjust column widths based on content or desired layout
    table.auto_set_column_width(col=list(range(len(df.columns))))


    # Header text color and bolding
    for (row, col), cell in table.get_celld().items():
        if row == 0: # Header row
            cell.set_text_props(color='white', fontweight='bold')
        # Time Taken (s) values bold
        if row > 0 and 'Time Taken (s)' in col_indices and col == col_indices['Time Taken (s)']:
             cell.set_text_props(fontweight='bold')


    output_dir = "tables"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"--- Algorithm Performance Comparison Table (PNG) saved to {filepath} ---")
    print("\n")


def generate_state_space_table_markdown():
    """
    Generates a Markdown table for Rubik's Cube state space complexity.
    """
    state_space_data = [
        {"Cube Dimension (N x N x N)": "2x2x2", "Approximate Number of States": "3,674,160", "Order of Magnitude": "Millions ($3.67 \\times 10^6$)"},
        {"Cube Dimension (N x N x N)": "3x3x3", "Approximate Number of States": "43,252,003,274,489,856,000", "Order of Magnitude": "Quintillions ($4.33 \\times 10^{19}$)"},
        {"Cube Dimension (N x N x N)": "4x4x4", "Approximate Number of States": "7,401,196,841,564,901,869,874,093,974,498,574,336,000,000,000", "Order of Magnitude": "Septillions ($7.40 \\times 10^{45}$)"},
        {"Cube Dimension (N x N x N)": "5x5x5", "Approximate Number of States": "282,870,942,277,741,856,536,180,333,107,150,328,293,127,731,985,672,134,721,536,000,000,000,000,000", "Order of Magnitude": "Duovigintillions ($2.83 \\times 10^{74}$)"}
    ]
    df_states = pd.DataFrame(state_space_data)
    markdown_table_states = df_states.to_markdown(index=False)
    print("--- State Space Complexity Table (Markdown) ---")
    print(markdown_table_states)
    print("\n")

def generate_state_space_table_png(data, filename="state_space_complexity_table.png"):
    """
    Generates a PNG image of the Rubik's Cube state space complexity table using matplotlib.
    """
    df_states = pd.DataFrame(data).copy()

    # Convert 'Approximate Number of States' to numeric for proper formatting
    # This column contains strings with commas, so we need a conversion function
    def convert_large_number_string_to_float(s):
        try:
            return float(s.replace(',', ''))
        except ValueError:
            return np.nan

    df_states['Approximate Number of States'] = df_states['Approximate Number of States'].apply(convert_large_number_string_to_float)


    fig, ax = plt.subplots(figsize=(18, 5)) # Adjusted figure size for state space table
    ax.axis('off')

    # Define colors
    header_color = '#2196F3' # A nice blue for headers
    row_color_1 = '#e3f2fd'  # Lighter blue shade
    row_color_2 = '#bbdefb'  # Slightly darker blue shade

    # Prepare cell colors based on rows
    cell_colors = []
    for i in range(len(df_states)):
        row_colors = [row_color_1 if i % 2 == 0 else row_color_2] * len(df_states.columns)
        cell_colors.append(row_colors)

    # Convert data for display, applying scientific notation where appropriate
    display_data = df_states.astype(str).values # Start with string representation of all values

    # Get column indices dynamically
    col_indices = {col_name: i for i, col_name in enumerate(df_states.columns)}

    if 'Approximate Number of States' in col_indices:
        for i, val in enumerate(df_states['Approximate Number of States']):
            if pd.notna(val):
                display_data[i, col_indices['Approximate Number of States']] = f"{val:.2e}" # Format as scientific notation with 2 decimal places

    table = ax.table(cellText=display_data, # Use display_data for cell text
                     colLabels=df_states.columns,
                     cellLoc='center',
                     loc='center',
                     cellColours=cell_colors,
                     colColours=[header_color] * len(df_states.columns)
                    )

    table.auto_set_font_size(False)
    table.set_fontsize(9) # Slightly reduced font size
    table.scale(1.2, 1.2)

    # Adjust column widths dynamically
    table.auto_set_column_width(col=list(range(len(df_states.columns))))

    # Header text color
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')

    output_dir = "tables"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"--- State Space Complexity Table (PNG) saved to {filepath} ---")
    print("\n")

# Your data from the previous interaction
performance_data_source = [
    {"Scramble Sequence": "F' U D U'", "Algorithm": "A*", "Solution Path": "D' F", "Path Length": 2, "Nodes Explored": 3, "Time Taken (s)": "0.00020"},
    {"Scramble Sequence": "F' U D U'", "Algorithm": "BFS (Illustrative)", "Solution Path": "D' F", "Path Length": 2, "Nodes Explored": 10, "Time Taken (s)": "0.00050"},
    {"Scramble Sequence": "F' U D U'", "Algorithm": "DFS (Illustrative)", "Solution Path": "D' F", "Path Length": 2, "Nodes Explored": 5, "Time Taken (s)": "0.00030"},
    {"Scramble Sequence": "B R F F", "Algorithm": "A*", "Solution Path": "F F R' B'", "Path Length": 4, "Nodes Explored": 5, "Time Taken (s)": "0.00040"},
    {"Scramble Sequence": "B R F F", "Algorithm": "BFS (Illustrative)", "Solution Path": "F F R' B'", "Path Length": 4, "Nodes Explored": 30, "Time Taken (s)": "0.00180"},
    {"Scramble Sequence": "B R F F", "Algorithm": "DFS (Illustrative)", "Solution Path": "F F R' B'", "Path Length": 4, "Nodes Explored": 18, "Time Taken (s)": "0.00100"},
    {"Scramble Sequence": "L R' L U", "Algorithm": "A*", "Solution Path": "U' L L R", "Path Length": 4, "Nodes Explored": 33, "Time Taken (s)": "0.00370"},
    {"Scramble Sequence": "L R' L U", "Algorithm": "BFS (Illustrative)", "Solution Path": "U' L L R", "Path Length": 4, "Nodes Explored": 150, "Time Taken (s)": "0.00900"},
    {"Scramble Sequence": "L R' L U", "Algorithm": "DFS (Illustrative)", "Solution Path": "U' L L R", "Path Length": 4, "Nodes Explored": 70, "Time Taken (s)": "0.00650"},
    {"Scramble Sequence": "L R' F F'", "Algorithm": "A*", "Solution Path": "L' R", "Path Length": 2, "Nodes Explored": 3, "Time Taken (s)": "0.00020"},
    {"Scramble Sequence": "L R' F F'", "Algorithm": "BFS (Illustrative)", "Solution Path": "L' R", "Path Length": 2, "Nodes Explored": 12, "Time Taken (s)": "0.00060"},
    {"Scramble Sequence": "L R' F F'", "Algorithm": "DFS (Illustrative)", "Solution Path": "L' R", "Path Length": 2, "Nodes Explored": 6, "Time Taken (s)": "0.00030"},
    {"Scramble Sequence": "L' R B' L", "Algorithm": "A*", "Solution Path": "R' U", "Path Length": 2, "Nodes Explored": 3, "Time Taken (s)": "0.00030"},
    {"Scramble Sequence": "L' R B' L", "Algorithm": "BFS (Illustrative)", "Solution Path": "R' U", "Path Length": 2, "Nodes Explored": 15, "Time Taken (s)": "0.00070"},
    {"Scramble Sequence": "L' R B' L", "Algorithm": "DFS (Illustrative)", "Solution Path": "R' U", "Path Length": 2, "Nodes Explored": 7, "Time Taken (s)": "0.00040"},
]

# Ensure Time Taken is formatted as string to preserve trailing zeros if desired from source
# This formatting is now applied within the generate_performance_table_png function for floats
# For the source data, it's already strings, which is fine for the initial DataFrame creation.

state_space_data_source = [
    {"Cube Dimension (N x N x N)": "2x2x2", "Approximate Number of States": "3,674,160", "Order of Magnitude": "Millions ($3.67 \\times 10^6$)"},
    {"Cube Dimension (N x N x N)": "3x3x3", "Approximate Number of States": "43,252,003,274,489,856,000", "Order of Magnitude": "Quintillions ($4.33 \\times 10^{19}$)"},
    {"Cube Dimension (N x N x N)": "4x4x4", "Approximate Number of States": "7,401,196,841,564,901,869,874,093,974,498,574,336,000,000,000", "Order of Magnitude": "Septillions ($7.40 \\times 10^{45}$)"},
    {"Cube Dimension (N x N x N)": "5x5x5", "Approximate Number of States": "282,870,942,277,741,856,536,180,333,107,150,328,293,127,731,985,672,134,721,536,000,000,000,000,000", "Order of Magnitude": "Duovigintillions ($2.83 \\times 10^{74}$)"}
]

if __name__ == '__main__':
    generate_performance_table_png(performance_data_source)
    generate_state_space_table_png(state_space_data_source)
    # The markdown tables generated by the Python code are no longer needed for the report itself,
    # but the functions are kept for demonstration if text-based markdown is desired.
    # generate_performance_comparison_table_markdown(performance_data_source)
    # generate_state_space_table_markdown()
