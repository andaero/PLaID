import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import linregress

# Load the custom TTF font
custom_font_path = "vis/DMMono-Regular.ttf"
custom_font = fm.FontProperties(fname=custom_font_path)

relaxed_csv = "../crystal-text-llm/evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.7_uncon_esen_ehull_results.csv"
dft_json = "ehulls/qwen-dpo-wyckoff-combined-t2.json"

df = pd.read_csv(relaxed_csv)
dft_df = pd.read_json(dft_json)
dft_df = dft_df.sample(n=1000, random_state=420)

dft_df = dft_df[(dft_df['e_above_hull_per_atom_dft_corrected'] >= -0.2) & 
                (dft_df['e_above_hull_per_atom_dft_corrected'] <= 0.2)]

df = pd.merge(df, dft_df, left_index=True, right_on="original_index", how="inner")
df = df.dropna(subset=['e_above_hull', 'e_above_hull_per_atom_dft_corrected'])

x = df['e_above_hull_per_atom_dft_corrected']
y = df['e_above_hull']
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Plot with custom font
plt.figure(figsize=(6, 6))
plt.scatter(x, y, label='Sampled Structures', color='#c9aeff')
plt.plot(x, slope * x + intercept, label=f'Fit line', color='#1188ff', linewidth=2)
plt.plot([-0.2, 0.25], [-0.2, 0.25], linestyle='--', color='gray', linewidth=1, label='y=x Reference Line')
plt.axvline(x=0.1, linestyle='--', color='#ff7777', linewidth=1, label='Metastability x=0.1')
plt.xlabel('E Above Hull, DFT (eV/atom)', fontproperties=custom_font, fontsize=14)
plt.ylabel('E Above Hull, eqV2 (eV/atom)', fontproperties=custom_font, fontsize=14)
plt.legend(prop=custom_font, fontsize=16, loc='lower right')

# Add R^2 value as text annotation (not in legend)
r_squared_text = f"${{R^2 = {r_value**2:.2f}}}$"
plt.text(0.05, 0.95, r_squared_text,
         transform=plt.gca().transAxes,
         fontsize=16,
         fontproperties=custom_font,
         verticalalignment='top')

plt.xlim(-0.2, 0.25)
plt.ylim(-0.2, 0.25)
plt.xticks(ticks=[-0.2, -0.1, 0, 0.1, 0.2])
plt.yticks(ticks=[-0.2, -0.1, 0, 0.1, 0.2])

plt.tight_layout()  # Automatically adjusts padding to minimize whitespace
plt.grid(True)

# Save and show
plt.savefig('r2_plot.pdf')
plt.savefig('r2_plot.png')
plt.show()