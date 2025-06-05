import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# Configure matplotlib to use DM Mono font
font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'DMMono-Regular.ttf')
font_prop = fm.FontProperties(fname=font_path, size=18)  # Increased base font size
title_font_prop = fm.FontProperties(fname=font_path, size=18)  # Larger font for titles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DM Mono']

# Register the font with matplotlib
fm.fontManager.addfont(font_path)

# Mapping from space group numbers to labels
sg_number_to_label = {
    1: 'P-1',
    12: 'C2/m',
    62: 'Pnma',
    63: 'Cmcm',
    129: 'P4/nmm',
    139: 'I4/mmm',
    166: 'R-3m',
    194: 'P6_3/mmc',
    221: 'Pm-3m',
    225: 'Fm-3m'
}

# Function to plot the bar graph

def plot_overlaid_sg_bar_graph(df1, df2, df3, label1, label2, label3, title):
    # Create a figure
    plt.figure(figsize=(12, 6))
    
    # Define the space group labels and their colors
    sg_labels = list(sg_number_to_label.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f', '#76d7c4', '#f7b7a3', '#d3a4ff']
    
    # Map the actual_sg numbers to labels
    df1['sg_label'] = df1['actual_sg'].map(sg_number_to_label)
    df2['sg_label'] = df2['actual_sg'].map(sg_number_to_label)
    df3['sg_label'] = df3['actual_sg'].map(sg_number_to_label)
    
    # Calculate the frequency of each space group in the datasets
    freq1 = df1['sg_label'].value_counts(normalize=True).reindex(sg_labels, fill_value=0)
    freq2 = df2['sg_label'].value_counts(normalize=True).reindex(sg_labels, fill_value=0)
    freq3 = df3['sg_label'].value_counts(normalize=True).reindex(sg_labels, fill_value=0)
    
    # Define the bar width
    bar_width = 0.25
    
    # Set the positions of the bars
    r1 = np.arange(len(sg_labels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create the bars
    plt.bar(r1, freq1, color=colors, width=bar_width, edgecolor='grey', label=label1)
    plt.bar(r2, freq2, color=colors, width=bar_width, edgecolor='grey', label=label2)
    plt.bar(r3, freq3, color=colors, width=bar_width, edgecolor='grey', label=label3)
    
    # Add labels and title
    plt.xlabel('Space Group', fontproperties=font_prop)
    plt.ylabel('Frequency', fontproperties=font_prop)
    plt.title(title, fontproperties=title_font_prop)
    plt.xticks([r + bar_width for r in range(len(sg_labels))], sg_labels, rotation=45)
    
    # Add legend
    plt.legend(prop=font_prop, fontsize=12)
    
    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f"visualizations/graphs/{title}.png", dpi=300)
    plt.show()

# Example usage
if __name__ == "__main__":
    df_sg_label = pd.read_csv("visualizations/analysis/finetune_dataset.csv")
    df_sg_label_2 = pd.read_csv("visualizations/analysis/7b-dpo-wyckoff-histogram.csv")
    df_sg_label_3 = pd.read_csv("visualizations/analysis/7b-non-wyckoff-histogram.csv")
    plot_overlaid_sg_bar_graph(df_sg_label, df_sg_label_2, df_sg_label_3, "Train", "PLaID", "PLaID Base", "final-paper-bar") 