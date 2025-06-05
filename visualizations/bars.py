import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.symmetry.groups import SpaceGroup
from pyxtal import pyxtal
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.font_manager as fm

# Configure matplotlib to use DM Mono font
font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'DMMono-Regular.ttf')
font_prop = fm.FontProperties(fname=font_path, size=18)
title_font_prop = fm.FontProperties(fname=font_path, size=18)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DM Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = 12.5
plt.rcParams['ytick.labelsize'] = 12.5
fm.fontManager.addfont(font_path)

def get_actual_sg(x, cif=False):
    """Return the numeric space-group for a pymatgen Structure or CIF."""
    if cif:
        struct = Structure.from_str(x["cif"], fmt="cif")
    else:
        struct = Structure.from_dict(x["structure"])
    if "lattice" not in x.get("structure", {}):
        print(f"Missing lattice for entry; skipping")
        return None
    px = pyxtal()
    try:
        px.from_seed(struct, tol=0.01)
    except:
        px.from_seed(struct, tol=0.0001)
    return px.group.number

def apply_parallel(func, df, num_workers):
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        for res in tqdm(exe.map(func, [row for _, row in df.iterrows()]),
                        total=len(df), desc="Processing"):
            if res is not None:
                results.append(res)
    return results

def get_histogram_csv(file, output_file):
    df = pd.read_json(f"/cs/cs152/individual/flowmm/relaxed_structures/{file}")
    vals = apply_parallel(get_actual_sg, df, num_workers=os.cpu_count())
    pd.DataFrame({"actual_sg": vals}).to_csv(
        f"visualizations/analysis/{output_file}.csv", index=False)

def plot_spacegroup_stack_exact(dfs, labels, title, figsize=(5.5, 1.5), freq_threshold=2.1e-2):
    """
    dfs            : dict of {label: pd.Series of SG ints}
    labels         : list of str (keys into dfs, in desired order)
    title          : filename (without extension)
    figsize        : tuple for fig size in inches
    freq_threshold : only label SGs above this fraction in the reference set
    """
    # 1) compute fractions
    fractions = {
        method: data.value_counts().sort_index() / len(data)
        for method, data in dfs.items()
    }

    # 2) pick reference (last in labels) to choose which SGs to display
    ref = labels[-1]
    choosen_sgs = fractions[ref].loc[fractions[ref] > freq_threshold].index.tolist()

    # 3) precompute cumulative mid-points for x-ticks
    ref_cumsum = fractions[ref].cumsum()
    ref_midpts = ref_cumsum - (fractions[ref] / 2)

    # 4) start plotting
    matplotlib.rc('font', size=7)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = matplotlib.colormaps['tab10']

    for i, method in enumerate(labels):
        segments = fractions[method]
        start = 0
        for sg, frac in segments.items():
            ax.barh(i,
                    frac,
                    left=start,
                    color=cmap(sg % cmap.N),
                    edgecolor='none')
            start += frac

    # 5) styling axes
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for sp in ['top','right','bottom','left']:
        ax.spines[sp].set_visible(False)

    # 6) x-ticks only for chosen SGs, colored to match segment
    xticks = ref_midpts.loc[choosen_sgs]
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [SpaceGroup(g).hm_name for g in choosen_sgs],
        rotation=26
    )
    for tick, sg in zip(ax.xaxis.get_ticklabels(), choosen_sgs):
        tick.set_color(cmap(sg % cmap.N))

    ax.grid(axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"visualizations/graphs/{title}.pdf")
    plt.show()

def main():
    df_train = pd.read_csv("visualizations/analysis/finetune_dataset.csv")["actual_sg"].astype(int)
    df_plaid = pd.read_csv("visualizations/analysis/7b-dpo-wyckoff-histogram.csv")["actual_sg"].astype(int)
    df_base  = pd.read_csv("visualizations/analysis/7b-non-wyckoff-histogram.csv")["actual_sg"].astype(int)

    dfs = {
        "Train": df_train,
        "PLaID": df_plaid,
        "MP–20": df_base  # use whatever your reference label is
    }

    plot_spacegroup_stack_exact(
        dfs, 
        labels=["Train", "PLaID", "MP–20"],
        title="space_groups_horizontal"
    )


if __name__ == "__main__":
    main()
