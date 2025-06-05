import pandas as pd
import os

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.lattice import Lattice

def get_sg_label(x):
    sg_des = x['space_group']
    return sg_des

def get_actual_sg(x):
    structure = Structure.from_str(x['cif'], fmt="cif")
    pyx = pyxtal()
    try:
        # TODO: Tested higher tols, not worth
        pyx.from_seed(structure, tol=0.01)
        # pyx.from_seed(structure, tol=0.1)
    except:
        print('failed pyxtal 0.01 trying 0.0001 tol')
        pyx.from_seed(structure, tol=0.0001)
    return pyx.group.number
    # try:
    #     sga = SpacegroupAnalyzer(structure, tol)
    #     sg = sga.get_space_group_number()
    # except:
    #     return False
    # return sg

    
def analyze_alex_mp():
    df = pd.read_csv("alex_mp_20/train.csv")
    df['sg_label'] = df.apply(get_sg_label, axis=1)
    df['actual_sg'] = df.apply(get_actual_sg, axis=1)

    stats_sg_label = df.group_by('sg_label')
    stats_actual_sg = df.group_by('actual_sg')

    print("Summary statistics for space group labels:")
    print(stats_sg_label)
    print("\nSummary statistics for actual space groups:")
    print(stats_actual_sg)
    
    # Plot histograms for both sg_label and actual_sg
    plt.figure(figsize=(14, 6))

    # Plot for space group labels
    plt.subplot(1, 2, 1)
    sns.histplot(df['sg_label'], kde=False, bins=20, color='skyblue')
    plt.title('Histogram of Space Group Labels')
    plt.xlabel('Space Group Label')
    plt.ylabel('Count')

    # Plot for actual space groups
    plt.subplot(1, 2, 2)
    sns.histplot(df['actual_sg'], kde=False, bins=20, color='salmon')
    plt.title('Histogram of Actual Space Groups')
    plt.xlabel('Space Group Number')
    plt.ylabel('Count')

    # Show the plots
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    analyze_alex_mp()