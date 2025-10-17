import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import pandas as pd

# Set up custom font
font_path = "./visualizations/DMMono-Regular.ttf"
custom_font = FontProperties(fname=font_path)

# Load data from CSV files
def load_data(filename, column_name):
    data = pd.read_csv(filename)
    return data[column_name].tolist()

# Load both datasets
base_data = load_data("evals/results/qwen_7b_wyckoff_temp_0.7_esen_ehull_results.csv", "e_above_hull")
wyckoff_dpo_data = load_data("evals/results/qwen_7b_dpo_wyckoff_tiered_v2_it3_temp0.7_esen_ehull_results.csv", "e_above_hull")

# Set up the plot
plt.figure(figsize=(6, 4), dpi=300)

# Define colors to match the original image
colors = ["#76c7c0", "#c78fd6"]  # CDVAE-like cyan, LLAMA-like purple

# Calculate the range of the data
min_value = min(min(base_data), min(wyckoff_dpo_data))
max_value = max(max(base_data), max(wyckoff_dpo_data))

# Define a fixed bin width
bin_width = 0.035

# Create bin edges
bins = np.arange(min_value, max_value + bin_width, bin_width)

# Plot histograms with fixed bin width
sns.histplot(base_data, bins=bins, color=colors[0], alpha=0.7, label="Base Model")
sns.histplot(wyckoff_dpo_data, bins=bins, color=colors[1], alpha=0.7, label="DPO Iteration 3")

# Formatting the plot
plt.xlabel(r"$\hat{E}_{hull}$ (eV/atom)", fontsize=12, fontproperties=custom_font)
plt.ylabel("# structures", fontsize=12, fontproperties=custom_font)
plt.xticks(fontproperties=custom_font)
plt.yticks(fontproperties=custom_font)
plt.legend(frameon=True, loc="upper right", prop=custom_font)
plt.xlim(-0.25, 1.0)
plt.ylim(0, 4000)  # Adjust based on your data range
plt.grid(False)  # Remove grid for a cleaner look
plt.box(True)  # Keep box outline for professional style

# Save the figure
plt.savefig("visualizations/graphs/ehull/it_3_vs_base_new_dims.png", dpi=300, bbox_inches="tight")
#plt.savefig("hull_energy_histogram.pdf", bbox_inches="tight")

# Show the plot
plt.show()