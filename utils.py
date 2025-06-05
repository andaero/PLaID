from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.lattice import Lattice
import numpy as np


def get_crystal_string_wyckoff_pyx(cif_str, tol=0.01, translate=False):
    structure = Structure.from_str(cif_str, fmt="cif")

    # Randomly translate within the unit cell
    if translate:
        structure_og = structure.copy()
        structure.translate_sites(
            indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
        )

    # optimize to fit space group
    sga = SpacegroupAnalyzer(structure, tol)
    structure = sga.get_refined_structure()

    pyx = pyxtal()
    try:
        # TODO: Tested higher tols, not worth
        pyx.from_seed(structure, tol=0.01)
        # pyx.from_seed(structure, tol=0.1)
    except:
        print("failed pyxtal 0.01 trying 0.0001 tol")
        pyx.from_seed(structure, tol=0.0001)

    # Output space group and lattice parameters
    outs = [
        f"{pyx.formula}",
        f"Spacegroup: {pyx.group.symbol}",
        f"abc: {' '.join(f'{val:.2f}' for val in pyx.lattice.get_para()[:3])}",
        f"angles: {' '.join(f'{val:.2f}' for val in pyx.lattice.get_para(degree=True)[3:])}",
    ]
    outs.append(f"Sites ({len(structure)})")

    # Gather site and Wyckoff position info from pyxtal atom_sites
    data = []
    for site in pyx.atom_sites:
        species = site.specie
        coord = site.position

        # Extract Wyckoff position and label
        wp_label = site.wp.get_label()
        # Note 3f for lossless structure matching between og and text
        # However use 2f for init training
        row = f"{species} " + " ".join([f"{j:.3f}" for j in coord]) + f" {wp_label}"
        data.append(row)

    # # Combine header and site data into the final output string
    wyckoff_string = "\n".join(outs + data)
    # print('wyck str pyxtal: ', wyckoff_string)

    # TO TEST IF TEXT STRING MATCHES ORIGINAL PYMATGEN STRUCTURE
    # wyck_pyxtal = parse_fn_wyckoff(wyckoff_string, to_cif=False)
    # if not wyck_pyxtal.matches(structure):
    #     print('NOT SAME')
    #     print('pyxtal: ', wyck_pyxtal)
    #     print('pymatgen: ', structure)
    return wyckoff_string


def get_crystal_string_wyckoff(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")

    # Randomly translate within the unit cell
    # structure.translate_sites(
    #     indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
    # )

    # print('og structure: ', structure)
    # get wycoff positions
    sga = SpacegroupAnalyzer(structure, 0.01)
    # used looser tolerance of 0.1 vs default 0.01 in accordance with MaterialsProject
    structure = sga.get_symmetrized_structure()
    # print(structure.to(fmt='cif'))

    outs = [
        f"{structure.formula}",
        f"Spacegroup: {structure.spacegroup.int_symbol}",
        f"abc: {' '.join(f'{val:.2f}' for val in structure.lattice.abc)}",
        f"angles: {' '.join(f'{val:.2f}' for val in structure.lattice.angles)}",
    ]
    outs.append(f"Sites ({len(structure)})")
    data = []
    for idx, sites in enumerate(structure.equivalent_sites):
        site = sites[0]
        row = (
            f"{site.species_string} "
            + " ".join([f"{j:.2f}" for j in site.frac_coords])
            + f" {structure.wyckoff_symbols[idx]}"
        )
        data.append(row)
    wycoff_string = "\n".join(outs + data)
    return wycoff_string


def parse_fn_wyckoff(gen_str, to_cif=True):
    # Split the input into lines and remove empty ones
    lines = [x.strip() for x in gen_str.split("\n") if len(x.strip()) > 0]

    # Extract lattice parameters
    spacegroup_symbol = lines[1].split(":")[1].strip()
    lengths = [float(x) for x in lines[2].split(":")[1].strip().split()]
    angles = [float(x) for x in lines[3].split(":")[1].strip().split()]

    # Convert space group symbol to number using pymatgen
    spacegroup_number = Group(spacegroup_symbol).number

    # Extract site information
    sites_lines = lines[5:]

    elem_dict = {}
    for site in sites_lines:
        parts = site.split()
        element = parts[0]
        x, y, z = [float(coord) for coord in parts[1:4]]  # list [x, y, z]
        # Extract the multiplicity and letter (e.g., "2a")
        wyckoff_label = parts[4]
        multiplicity = int(wyckoff_label[:-1])
        wyckoff_letter = wyckoff_label[-1]

        site_tuple = (wyckoff_label, x, y, z)
        if element not in elem_dict:
            site_list = [site_tuple]
            elem_dict[element] = {
                "sites": site_list,
                "numIons": multiplicity,
            }
        else:
            # add tuple to list of tuples
            elem_dict[element]["sites"].append(site_tuple)
            elem_dict[element]["numIons"] += multiplicity

    # Create a pymatgen Lattice object from lengths and angles
    lattice = Lattice.from_para(*lengths, *angles, radians=False)

    # print(elem_dict)
    elements = list(elem_dict.keys())
    num_ions = [elem["numIons"] for elem in elem_dict.values()]
    sites = [elem["sites"] for elem in elem_dict.values()]
    # print(elements)
    # print(num_ions)
    # print(sites)

    # Create a pyxtal structure using from_random
    crystal = pyxtal()
    crystal.from_random(
        dim=3,
        group=spacegroup_number,
        species=elements,
        numIons=num_ions,
        lattice=lattice,
        sites=sites,
        max_count=1,
    )
    # print(crystal)
    # Convert to pymatgen Structure
    structure = crystal.to_pymatgen()

    if to_cif:
        return structure.to(fmt="cif")

    return structure
