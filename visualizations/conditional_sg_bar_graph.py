import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pymatgen.symmetry.groups import SpaceGroup

# Directory containing the CSV files
csv_directory = 'evals/results/sg_sun'

# List of space groups to consider
space_groups_to_plot = [1, 15, 38, 119, 143, 194, 216]

# Dictionary to hold num_ssun values for each model and space group
model_data = {}

# Iterate over each file in the directory
for filename in os.listdir(csv_directory):
    # Check if the file matches the pattern and is one of the specified space groups
    for N in space_groups_to_plot:
        if filename == f'sun_{N}_esen_metastability.csv':
            file_path = os.path.join(csv_directory, filename)
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Iterate over each row in the DataFrame
            for _, row in df.iterrows():
                model_name = row['model_name']
                num_ssun = row['num_ssun'] / 1000 * 100  # Convert to percentage
                if model_name not in model_data:
                    model_data[model_name] = {N: num_ssun}
                else:
                    model_data[model_name][N] = num_ssun

# Configure matplotlib to use DM Mono font
font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'DMMono-Regular.ttf')
font_prop = fm.FontProperties(fname=font_path, size=18)  # Increased base font size
title_font_prop = fm.FontProperties(fname=font_path, size=18)  # Larger font for titles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DM Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = 12.5  # Increased tick label size
plt.rcParams['ytick.labelsize'] = 12.5  # Increased tick label size

# Register the font with matplotlib
fm.fontManager.addfont(font_path)

# Dictionary to map space group numbers to labels
sg_number_to_label = {
    1:   'P1',
    15:  'C2/c',
    38:  'Amm2',
    119: 'I4m2',
    143: 'P3',
    194: 'P6_3/mmc',
    216: 'F-43m'
}

# Use the dictionary to get space group labels for the x-axis
space_group_labels = [sg_number_to_label[N] for N in space_groups_to_plot]

# Define the desired order of models
model_order = ['qwen_7b_wyckoff_sg', 'qwen_7b_wyckoff_dpo_sg_it3_temp_0.7', 'qwen_7b_wyckoff_dpo_sg_combined_t2_it3']

# Define the color scheme from alex_mp_analysis_fast.py
colors = ['#76c7c0', '#c78fd6', '#9966ff']

# Plotting the data for each model as a bar chart
plt.figure(figsize=(12, 8))
bar_width = 0.1
index = range(len(space_groups_to_plot))

# Define the legend labels for each model
legend_labels = {
    'qwen_7b_wyckoff_sg': 'PLaID++ (Wyckoff Base)',
    'qwen_7b_wyckoff_dpo_sg_it3_temp_0.7': 'PLaID++ (Spacegroup Only)',
    'qwen_7b_wyckoff_dpo_sg_combined_t2_it3': 'PLaID++ (Spacegroup + Stability)'
}

# Create a bar for each model in the specified order with the corresponding color and legend label
for i, model in enumerate(model_order):
    if model in model_data:
        data = model_data[model]
        plt.bar([x + i * bar_width for x in index], data.values(), bar_width, label=legend_labels[model], color=colors[i])

# Update plot labels and title to use the new font properties
plt.xlabel('Space Group', fontproperties=font_prop)
plt.ylabel('Percent S.S.U.N. (%)', fontproperties=font_prop)
plt.xticks([x + bar_width * (len(model_data) / 2) for x in index], space_group_labels)
plt.legend(prop=font_prop)
plt.tight_layout()
plt.savefig('visualizations/graphs/percent_ssun_models_across_space_groups_histogram.png')
plt.show()