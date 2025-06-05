import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the CSV files
csv_directory = 'evals/results/qwen_7b_wyckoff_sg'

# Dictionary to hold space group counts
space_group_counts = {}

# Iterate over each file in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_directory, filename)
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Assuming the space group information is in a column named 'space_group'
        for space_group in df['actual_spacegroup']:
            if space_group in space_group_counts:
                space_group_counts[space_group] += 1
            else:
                space_group_counts[space_group] = 1

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.bar(space_group_counts.keys(), space_group_counts.values())
plt.xlabel('Space Group')
plt.ylabel('Number of Crystals')
plt.title('Number of Crystals in Each Space Group')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('visualizations/graphs/qwen_7b_wyckoff_sg_histogram.png')
plt.show()
