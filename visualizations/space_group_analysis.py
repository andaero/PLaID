import pandas as pd
from alex_mp_analysis_fast import get_actual_sg, apply_parallel
import os

# Load the CSV file
df = pd.read_csv("data/basic/train.csv")

# Define a function to get the actual space group
def get_space_group(row):
    return get_actual_sg(row, cif=True)

# Apply get_actual_sg in parallel to each row to get the actual space group
actual_sg_series = apply_parallel(get_space_group, df, num_workers=os.cpu_count())

# Count the occurrences of each space group
actual_sg_counts = pd.Series(actual_sg_series).value_counts()

# Convert to DataFrame with space group numbers as index
actual_sg_df = pd.DataFrame({'count': actual_sg_counts})
actual_sg_df.index.name = 'actual_space_group'

# Save to CSV file
actual_sg_df.to_csv('visualizations/analysis/train_actual_sg_histogram.csv')