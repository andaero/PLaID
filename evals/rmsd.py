import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm
import matplotlib.pyplot as plt

def rmsd_structure_matcher(cif_init: str, cif_relaxed: str, matcher: StructureMatcher) -> float:
    """
    Compute RMSD between generated and ground-state structures using pymatgen's StructureMatcher.
    Returns None if structures do not match.
    """
    # Parse CIF strings
    s_init    = Structure.from_str(cif_init, fmt="cif")
    s_relaxed = Structure.from_str(cif_relaxed, fmt="cif")
    
    # Attempt matching
    if not matcher.fit(s_init, s_relaxed):
        return None
    
    # Get RMS distance
    return matcher.get_rms_dist(s_init, s_relaxed)

def rmsd_statistics_structure_matcher(fn: str, threshold: float = None, plot: bool = False):
    df = pd.read_csv(fn)
    matcher = StructureMatcher()
    rmsd_list = []
    unmatched = 0
    
    for ci, cr in tqdm(zip(df["cif"], df["relaxed_cif"]),
                       total=len(df), desc="Calculating RMSDs"):
        rmsd = rmsd_structure_matcher(ci, cr, matcher)
        if rmsd is None:
            unmatched += 1
        else:
            rmsd_list.append(rmsd)
    
    arr = np.array(rmsd_list)
    
    if unmatched:
        print(f"Warning: {unmatched} structure pairs did not match and were skipped.")
    
    mean_rmsd = arr.mean()
    p95_rmsd  = np.percentile(arr, 95)
    
    print(f"Average RMSD:      {mean_rmsd:.4f} Å")
    print(f"95th-percentile:   {p95_rmsd:.4f} Å   (95% of matched samples < {p95_rmsd:.4f} Å)")
    
    if threshold is not None:
        pct_below = (arr < threshold).mean() * 100
        print(f"Samples < {threshold:.3f} Å: {pct_below:.1f}%")
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.hist(arr, bins=30, alpha=0.7, edgecolor='black')
        plt.title('StructureMatcher RMSD Histogram')
        plt.xlabel('RMSD (Å)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig("rmsd_histogram_structure_matcher.png")

    return mean_rmsd, p95_rmsd

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate mean & 95th-percentile RMSD using StructureMatcher")
    parser.add_argument("csv_file", help="CSV with columns 'cif' and 'relaxed_cif'")
    parser.add_argument("--check", type=float,
                        help="report percentage below this RMSD threshold")
    parser.add_argument("--plot", action="store_true",
                        help="plot histogram of RMSDs")
    args = parser.parse_args()
    
    rmsd_statistics_structure_matcher(args.csv_file, threshold=args.check, plot=args.plot)