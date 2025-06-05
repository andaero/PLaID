import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.symmetry.groups import SpaceGroup
from pyxtal import pyxtal
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.font_manager as fm

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

def get_sg_label(x):
    sg_des = x["space_group"]
    return sg_des

def get_actual_sg(x, cif=True):
    if cif:
        structure = Structure.from_str(x["cif"], fmt="cif")
    else:
        structure = Structure.from_dict(x["structure"])

    """
    print(structure)
    structure_data = x.get("structure", {})
    if "lattice" not in structure_data: # lattices are needed for space group analysis
        # print(f"Missing lattice in structure for entry: {x}")
        return None
    """

    pyx = pyxtal()
    for tol in (0.01, 0.0001):          # try two tolerances
        try:
            pyx.from_seed(structure, tol=tol)
            return pyx.group.number
        except Exception:
            continue                    # try next tol
    

'''
Workers go brr
'''

def apply_parallel(func, df, num_workers):
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(func, [row for _, row in df.iterrows()]), total=len(df), desc="Processing", ncols=100):
            if result is not None:
                results.append(result)
    print(f"Processed {len(results)} rows successfully.")
    return results

def get_histogram_csv(file, output_file):
    # df = pd.read_json(f"/cs/cs152/individual/flowmm/relaxed_structures/{file}")
    df = pd.read_csv(file)
    print("length", len(df))
    print(df.head(5))
    print("DONE READING CSV, EXTRACTING SG LABEL AND ACTUAL SG")


    df_sg_label= pd.DataFrame({"actual_sg": apply_parallel(get_actual_sg, df, num_workers=os.cpu_count())})

    df_sg_label.to_csv(f"visualizations/analysis/{output_file}.csv", index=False)
    print("exported to .csv")

def plot_sg_histogram(df, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(df, bins=230)
    plt.title(title, fontproperties=title_font_prop)
    plt.xlabel("Space Group Number", fontproperties=font_prop)
    plt.ylabel("Frequency", fontproperties=font_prop)
    plt.grid()
    plt.savefig(f"visualizations/graphs/{title}.png", dpi=300)
    plt.show()

def plot_overlaid_sg_histogram(df1, df2, df3, label1, label2, label3, title):
    plt.figure(figsize=(10, 6))
    plt.hist(
        df1,
        bins=230,
        label=label1,
        alpha=0.5,
        weights=[1.0 / len(df1)] * len(df1),
        color="#33cc33"
    )
    plt.hist(
        df2,
        bins=230,
        label=label2,
        alpha=0.5,
        weights=[1.0 / len(df2)] * len(df2),
        color="#3399ff"
    )
    plt.hist(
        df3,
        bins=230,
        label=label3,
        alpha=0.5,
        weights=[1.0 / len(df3)] * len(df3),
        color="#9966ff"
    )
    plt.xlabel("Space Group Number", fontproperties=font_prop)
    plt.ylabel("Frequency", fontproperties=font_prop)
    plt.legend(prop=font_prop, fontsize=12)
    plt.grid()
    plt.savefig(f"visualizations/graphs/{title}.png", dpi=300)
    plt.show()

def plot_overlaid_sg_histogram_2(df1, df2, df3, label1, label2, label3, title):
    plt.figure(figsize=(10, 6))
    
    # Plot each histogram on the same axes, with discrete bins 1 through 230
    plt.hist(
        df1,
        bins=range(1, 232),
        label=label1,
        alpha=0.5,
        weights=[1.0 / len(df1)] * len(df1),
        color="#33cc33",
        rwidth=0.8
    )
    plt.hist(
        df2,
        bins=range(1, 232),
        label=label2,
        alpha=0.5,
        weights=[1.0 / len(df2)] * len(df2),
        color="#3399ff",
        rwidth=0.8
    )
    plt.hist(
        df3,
        bins=range(1, 232),
        label=label3,
        alpha=0.5,
        weights=[1.0 / len(df3)] * len(df3),
        color="#9966ff",
        rwidth=0.8
    )

    # Labels and legend
    plt.xlabel("Space Group Number", fontproperties=font_prop)
    plt.ylabel("Frequency", fontproperties=font_prop)
    plt.legend(prop=font_prop, fontsize=12)

    # Turn off grid lines
    plt.grid(False)

    # Add more ticks for the x-axis, for instance every 10 space groups
    plt.xticks(np.arange(0, 231, 10))

    # Make sure our x limits cover 1 through 230 fully
    plt.xlim(1, 230)

    plt.tight_layout()
    plt.savefig(f"visualizations/graphs/{title}.png", dpi=300)
    plt.show()

def plot_overlaid_sg_histogram_3(df1, df2, df3, label1, label2, label3, title):
    # Make the figure wider
    plt.figure(figsize=(12, 6))
    
    # Plot each histogram on the same axes, with discrete bins from 1 to 230
    plt.hist(
        df1,
        bins=range(1, 232),
        label=label1,
        alpha=0.5,
        weights=[1.0 / len(df1)] * len(df1),
        color="#33cc33",
        rwidth=0.9  # Make bars wider
    )
    plt.hist(
        df2,
        bins=range(1, 232),
        label=label2,
        alpha=0.5,
        weights=[1.0 / len(df2)] * len(df2),
        color="#e666ff",
        rwidth=0.9
    )
    plt.hist(
        df3,
        bins=range(1, 232),
        label=label3,
        alpha=0.5,
        weights=[1.0 / len(df3)] * len(df3),
        color="#9966ff",
        rwidth=0.9
    )
    
    # Labels and legend
    plt.xlabel("Space Group Number", fontproperties=font_prop)
    plt.ylabel("Frequency", fontproperties=font_prop)
    plt.legend(prop=font_prop, fontsize=12)
    
    # Turn off grid lines
    plt.grid(False)
    
    # Add more ticks on the x-axis, e.g., every 10 space groups
    plt.xticks(np.arange(0, 231, 10))
    
    # Make sure our x-limits cover 1 through 230 fully
    plt.xlim(1, 230)
    
    # Tidy layout
    plt.tight_layout()
    plt.savefig(f"visualizations/graphs/{title}.png", dpi=300)
    plt.show()

def plot_overlaid_sg_histogram_4(df1, df2, df3, label1, label2, label3, title):
    # Wider figure
    plt.figure(figsize=(12, 6))
    
    # Histograms with discrete bins from 1 to 230
    plt.hist(
        df1,
        bins=range(1, 232),
        label=label1,
        alpha=0.5,
        weights=[1.0 / len(df1)] * len(df1),
        color="#33cc33",
        rwidth=0.9
    )
    plt.hist(
        df2,
        bins=range(1, 232),
        label=label2,
        alpha=0.5,
        weights=[1.0 / len(df2)] * len(df2),
        color="#3399ff",
        rwidth=0.9
    )
    plt.hist(
        df3,
        bins=range(1, 232),
        label=label3,
        alpha=0.5,
        weights=[1.0 / len(df3)] * len(df3),
        color="#9966ff",
        rwidth=0.9
    )

    # Labels and legend
    plt.xlabel("Space Group Number", fontproperties=font_prop)
    plt.ylabel("Frequency", fontproperties=font_prop)
    plt.legend(prop=font_prop, fontsize=12)

    # Turn off grid lines
    plt.grid(False)

    # Use ticks every 10 space groups
    plt.xticks(np.arange(0, 231, 10))

    # Adjust x-limits so the first bar (centered at x=1) does not touch the y-axis
    # 0.5 is just an example offset; you can tweak it as needed
    plt.xlim(0.5, 230.5)

    plt.tight_layout()
    plt.savefig(f"visualizations/graphs/{title}.png", dpi=300)
    plt.show()

def analyze_finetune_dataset():
    print("reading dataset")
    df = pd.read_csv("data/basic/train.csv")
    from functools import partial
    func = partial(get_actual_sg, cif=True)
    print("DONE READING CSV, EXTRACTING SG LABEL ANfinetune_datasetD ACTUAL SG")
    df_sg_label = pd.DataFrame({"actual_sg": apply_parallel(func, df, num_workers=os.cpu_count())})
    df_sg_label.to_csv("visualizations/analysis/.csv", index=False)

def identify_top_5_sg(plaid_model):
    df1 = pd.read_csv("visualizations/analysis/finetune_dataset.csv")
    df2 = pd.read_csv(f"visualizations/analysis/{plaid_model}-histogram.csv")
    
    df1["actual_sg"] = df1["actual_sg"].astype(int)
    top_5_sg_1 = df1["actual_sg"].value_counts().nlargest(5).index.tolist()
    print("Top 5 Space Groups in finetune dataset:", top_5_sg_1)

    df2["actual_sg"] = df2["actual_sg"].astype(int)
    top_5_sg_2 = df2["actual_sg"].value_counts().nlargest(5).index.tolist()
    print(f"Top 5 Space Groups in {plaid_model} dataset:", top_5_sg_2)

def plot_spacegroup_stack(dfs, labels, title, top_n=10, figsize=(10, 4)):
    """
    dfs    : list of pandas.Series, each containing integer SG numbers
    labels : list of str, same length as dfs
    title  : str, plot title (also used for filename)
    top_n  : how many top SGs (by frequency in dfs[0]) to show
    """

    # 1) compute normalized frequency for each model
    freq_dict = {
        lbl: df.value_counts(normalize=True).sort_index()
        for df, lbl in zip(dfs, labels)
    }

    # 2) pick top_n SGs by freq in the first series (e.g. your Train set)
    top_sgs = freq_dict[labels[0]].nlargest(top_n).index.tolist()

    # 3) build a matrix [models × top_sgs]
    mat = np.zeros((len(dfs), len(top_sgs)))
    for i, lbl in enumerate(labels):
        for j, sg in enumerate(top_sgs):
            mat[i, j] = freq_dict[lbl].get(sg, 0.0)

    # 4) map SG numbers → Hermann–Mauguin symbols
    sg_symbols = []
    for sg in top_sgs:
        try:
            sg_symbols.append(SpaceGroup(sg).hm_name)
        except Exception:
            sg_symbols.append(f"{sg}")
    # optional: color-cycle for each SG
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(top_sgs))]

    # 5) plot
    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(dfs))

    for j, (sg, col) in enumerate(zip(sg_symbols, colors)):
        ax.barh(labels, mat[:, j],
                left=left,
                height=0.6,
                color=col,
                label=sg)
        left += mat[:, j]

    # 6) formatting
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of structures", fontproperties=font_prop)
    ax.set_title(title, fontproperties=title_font_prop)
    ax.set_xticks([])            # hide default x-ticks
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=font_prop)
    ax.invert_yaxis()            # match your ordering: top = first entry

    # 7) group separators & bottom labels
    cum = np.cumsum(mat, axis=1)
    for j in range(len(top_sgs)-1):
        # draw a thin white line between segments
        ax.vlines(cum[0,j], -0.5, len(dfs)-0.5, color='white', linewidth=2)

    # now place the SG symbols below the zero line
    for j, x in enumerate(cum[0]):
        ax.text(x - mat[0,j]/2, -0.1,
                sg_symbols[j],
                rotation=30,
                ha='center',
                va='top',
                fontproperties=font_prop,
                color=colors[j])

    # 8) legend (if you really need one)
    # ax.legend(ncol=5, bbox_to_anchor=(0.5, -0.3), loc="upper center", prop=font_prop)

    plt.tight_layout()
    plt.savefig(f"visualizations/graphs/{title}.png", dpi=300)
    plt.show()

def main():
# analyze_finetune_dataset()
    df_sg_label= pd.read_csv("visualizations/analysis/finetune_dataset.csv")
    df_sg_label_2 = pd.read_csv("visualizations/analysis/qwen-7b-dpo-wyckoff-histogram.csv.csv")
    df_sg_label_3 = pd.read_csv("visualizations/analysis/qwen-7b-base-histogram.csv.csv")
    plot_overlaid_sg_histogram_4(df_sg_label, df_sg_label_2, df_sg_label_3, "Train", "PLaID++", "PLaID++ (Wyckoff Base)", "final-paper")

    plot_spacegroup_stack(
        dfs   = [df_sg_label, df_sg_label_2, df_sg_label_3],
        labels= ["Train", "PLaID", "PLaID Base"],
        title = "spacegroup_stack"
    )

    
# find spikes:
    # identify_top_5_sg("7b-dpo-wyckoff")
    # identify_top_5_sg("7b-wyckoff")
    # identify_top_5_sg("8b-dpo-wyckoff")
    # identify_top_5_sg("8b-wyckoff")
    # identify_top_5_sg("8b-non-wyckoff")
    # identify_top_5_sg("7b-non-wyckoff")

# follow the procedure to get the csv histogram data on PLaID
    # get_histogram_csv("8b-dpo-wyckoff.json", "8b-dpo-wyckoff-histogram")
    # get_histogram_csv("8b-non-wyckoff.json", "8b-non-wyckoff-histogram")
    # get_histogram_csv("8b-wyckoff.json", "8b-wyckoff-histogram.csv")
    # get_histogram_csv("7b-wyckoff.json", "7b-wyckoff-histogram.csv")
    # get_histogram_csv("7b-dpo-wyckoff.json", "7b-dpo-wyckoff-histogram.csv")
    # get_histogram_csv("llm_samples_relaxed.json", "7b-non-wyckoff-histogram.csv")
    
    # get_histogram_csv("../crystal-text-llm/evals/results/qwen_7b_wyckoff_dpo_sg_combined_t2_it3_temp_0.7_uncon_esen_ehull_results.csv", "qwen-7b-dpo-wyckoff-histogram.csv")
    # get_histogram_csv("../crystal-text-llm/evals/results/qwen_7b_base_run_qwen_temp_0.7_esen_ehull_results.csv", "qwen-7b-base-histogram.csv")
    

# uncomment the following lines to plot the histograms
    # print("8b-dpo-wyckoff-histogram")
    # df_sg_label = pd.read_csv("visualizations/analysis/8b-dpo-wyckoff-histogram.csv")
    # plot_sg_histogram(df_sg_label, "8b-dpo-wyckoff-histogram")

    # print("8b-non-wyckoff-histogram")
    # df_sg_label = pd.read_csv("visualizations/analysis/8b-non-wyckoff-histogram.csv")
    # plot_sg_histogram(df_sg_label, "8b-non-wyckoff-histogram")

    # print("8b-wyckoff-histogram")
    # df_sg_label = pd.read_csv("visualizations/analysis/8b-wyckoff-histogram.csv")
    # plot_sg_histogram(df_sg_label, "8b-wyckoff-histogram")

    # print("7b-wyckoff-histogram")
    # df_sg_label = pd.read_csv("visualizations/analysis/7b-wyckoff-histogram.csv")
    # plot_sg_histogram(df_sg_label, "7b-wyckoff-histogram")
    
    # print("7b-dpo-wyckoff-histogram")
    # df_sg_label = pd.read_csv("visualizations/analysis/7b-dpo-wyckoff-histogram.csv")
    # plot_sg_histogram(df_sg_label, "7b-dpo-wyckoff-histogram")
    

if __name__ == "__main__":
    main()
