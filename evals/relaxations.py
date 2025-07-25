from pymatgen.core import Structure
from pymatgen.core.structure import Structure

# IMPORTS FOR M3GNET
import matgl
from matgl.ext.ase import Relaxer

# IMPORTS FOR OMAT
import io
from ase.io import read
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import OCPCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import CHGNet
import torch
from chgnet_ import prerelax_with_chgnet
from evals.llm_utils import cif_str_to_crystal
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc  # Add garbage collection module


def m3gnet_relaxed_energy(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")

    pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    relaxer = Relaxer(potential=pot)  # This loads the default pre-trained model

    relax_results = relaxer.relax(structure)
    final_structure = relax_results["final_structure"]
    final_energy_per_atom = float(relax_results["trajectory"].energies[-1])
    return final_energy_per_atom, final_structure


def chgnet_relaxed_energy(cif_str):
    """Calculate the relaxed energy of a structure using CHGNet.

    Args:
        cif_str: CIF string representation of the structure

    Returns:
        tuple: (initial_energy, relaxed_energy, relaxed_structure_dict)
    """
    # Load CHGNet model
    chgnet = CHGNet.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chgnet.to(device)

    # Convert CIF string to structure
    crystal = cif_str_to_crystal(cif_str)
    if crystal is None or not crystal.valid:
        return None, None

    structure = Structure.from_str(cif_str, fmt="cif")
    if len(structure) == 1:
        return None, None

    # Relax structure with CHGNet
    pair = prerelax_with_chgnet(structure=structure, chgnet=chgnet, steps=1500)

    e_gen = pair.energies[0]  # initial structure energy
    e_relax = pair.energies[1]  # relaxed structure energy
    relaxed_structure = pair.structure_dicts[-1]
    # Return initial energy, relaxed energy, and relaxed structure
    return e_relax, relaxed_structure


def eqv2_relaxed_energy(cif_str):
    # 1. Convert CIF string to ASE Atoms
    cif_buffer = io.StringIO(cif_str)
    atoms = read(cif_buffer, format="cif")

    # 2. Attach the OCP calculator
    calc = OCPCalculator(
        checkpoint_path="evals/OCPCalcs/eqV2_86M_omat_mp_salex.pt", cpu=False
    )
    atoms.calc = calc

    # 3. Relax both cell and atomic positions using FIRE + FrechetCellFilter
    dyn = FIRE(FrechetCellFilter(atoms))
    dyn.run(fmax=0.02, steps=500)

    # 4. Final energy in eV, then convert to per-atom
    final_energy_eV = atoms.get_potential_energy()
    # final_energy_per_atom = final_energy_eV / len(atoms)

    # 5. Convert relaxed ASE Atoms -> Pymatgen Structure
    final_structure = AseAtomsAdaptor.get_structure(atoms)

    return final_energy_eV, final_structure


def chgnet_relaxed_energy_batch(
    gen_file: Path, steps: int = 1500, index: np.ndarray = None
):
    """Calculate the relaxed energy of multiple structures using CHGNet.

    Args:
        gen_file: Path to CSV file containing structures
        steps: Maximum number of relaxation steps
        index: Array of indices to process (if None, process all structures)
    """
    # Load CHGNet model
    chgnet = CHGNet.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chgnet.to(device)

    # Import file from CSV
    df = pd.read_csv(gen_file)

    # If index is None, process all structures
    if index is None:
        index = np.arange(len(df))
    res = []
    for i in tqdm(index.tolist(), desc=f"Processing structures {index[0]}"):
        try:
            r = df.iloc[i]
            cif_str = r["cif"]

            crystal = cif_str_to_crystal(cif_str)
            if crystal is None or not crystal.valid:
                continue

            structure = Structure.from_str(cif_str, fmt="cif")
            if len(structure) == 1:
                continue

            # Relax structure with CHGNet
            pair = prerelax_with_chgnet(structure=structure, chgnet=chgnet, steps=steps)

            e_gen = pair.energies[0]  # initial structure energy
            e_relax = pair.energies[1]  # relaxed structure energy
            relaxed_structure = pair.structure_dicts[-1]

            # add initial energy, relaxed energy, and relaxed structure to df
            r["initial_energy"] = e_gen
            r["relaxed_energy"] = e_relax
            r["relaxed_cif"] = relaxed_structure
            res.append(r)
        except Exception as e:
            print(f"Error processing structure {i}: {e}")
            continue

    # Save the updated DataFrame to a new CSV file
    return pd.DataFrame(res)


def eqv2_relaxed_energy_batch(
    gen_file: Path, fmax: float = 0.02, index: np.ndarray = None
):
    """Calculate the relaxed energy of multiple structures using EQV2 OCP calculator.

    Args:
        gen_file: Path to CSV file containing structures.
        fmax: Maximum force convergence criterion for FIRE optimizer.
        index: Array of indices to process (if None, process all structures).

    Returns:
        pd.DataFrame: DataFrame containing the results with relaxed energy and CIF.
    """
    # Load OCP calculator
    # Consider making the checkpoint path an argument if needed
    calc = OCPCalculator(
        checkpoint_path="evals/OCPCalcs/eqV2_86M_omat_mp_salex.pt", cpu=False
    )

    # Import file from CSV
    df = pd.read_csv(gen_file)

    # If index is None, process all structures
    if index is None:
        index = np.arange(len(df))

    res = []
    # Use a generic description or pass it if needed
    desc = (
        f"Processing structures {index.min()}-{index.max()}"
        if index.size > 0
        else "Processing structures"
    )
    iter_count = 0

    for i in tqdm(index.tolist(), desc=desc):
        try:
            r = df.iloc[i].to_dict()  # Work with a copy
            cif_str = r["cif"]

            crystal = cif_str_to_crystal(cif_str)
            if crystal is None or not crystal.valid:
                print(f"Skipping structure {i}: Invalid crystal")
                continue

            # Use pymatgen Structure to check length easily
            structure_pmg = Structure.from_str(cif_str, fmt="cif")
            if len(structure_pmg) == 1:
                print(f"Skipping structure {i}: Single atom structure")
                continue

            # 1. Convert CIF string to ASE Atoms
            cif_buffer = io.StringIO(cif_str)
            atoms = read(cif_buffer, format="cif")

            # 2. Attach the OCP calculator
            atoms.calc = calc

            # 3. Relax both cell and atomic positions using FIRE + FrechetCellFilter
            dyn = FIRE(
                FrechetCellFilter(atoms), logfile=None
            )  # Set logfile to None to suppress output
            dyn.run(fmax=fmax, steps=500)

            # 4. Final energy in eV
            final_energy_eV = atoms.get_potential_energy()

            # 5. Convert relaxed ASE Atoms -> Pymatgen Structure -> CIF
            final_structure_pmg = AseAtomsAdaptor.get_structure(atoms)
            final_relaxed_cif = final_structure_pmg.to(fmt="cif")

            # Add results to the dictionary
            r["relaxed_energy"] = final_energy_eV
            r["relaxed_cif"] = final_relaxed_cif
            # Add other initial info if needed, e.g., initial energy requires separate calc
            # r["initial_energy"] = atoms.get_potential_energy() # Would need calc before relax
            res.append(r)
            # Perform garbage collection every 10 iterations
            iter_count += 1
            if iter_count % 10 == 0:
                # Run Python's garbage collector
                gc.collect()
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"Memory cleaned after processing {iter_count} structures")

        except Exception as e:
            print(f"Error processing structure {i}: {e}")
            continue

    # Final garbage collection before returning
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return the results as a DataFrame
    return pd.DataFrame(res)


def esen_relaxed_energy_batch(
    gen_file: Path, fmax: float = 0.02, index: np.ndarray = None
):
    """Calculate the relaxed energy of multiple structures using ESEN OCP calculator.

    Args:
        gen_file: Path to CSV file containing structures.
        fmax: Maximum force convergence criterion for FIRE optimizer.
        index: Array of indices to process (if None, process all structures).

    Returns:
        pd.DataFrame: DataFrame containing the results with relaxed energy and CIF.
    """
    # Load OCP calculator
    # Consider making the checkpoint path an argument if needed
    calc = OCPCalculator(checkpoint_path="evals/OCPCalcs/esen_30m_oam.pt", cpu=False)

    # Import file from CSV
    df = pd.read_csv(gen_file)

    # If index is None, process all structures
    if index is None:
        index = np.arange(len(df))

    res = []
    # Use a generic description or pass it if needed
    desc = (
        f"Processing structures {index.min()}-{index.max()}"
        if index.size > 0
        else "Processing structures"
    )

    # Track iteration count for garbage collection
    iter_count = 0

    for i in tqdm(index.tolist(), desc=desc):
        try:
            r = df.iloc[i].to_dict()  # Work with a copy
            cif_str = r["cif"]

            crystal = cif_str_to_crystal(cif_str)
            if crystal is None or not crystal.valid:
                print(f"Skipping structure {i}: Invalid crystal")
                continue

            # Use pymatgen Structure to check length easily
            structure_pmg = Structure.from_str(cif_str, fmt="cif")
            if len(structure_pmg) == 1:
                print(f"Skipping structure {i}: Single atom structure")
                continue

            # 1. Convert CIF string to ASE Atoms
            cif_buffer = io.StringIO(cif_str)
            atoms = read(cif_buffer, format="cif")

            # 2. Attach the OCP calculator
            atoms.calc = calc

            # 3. Relax both cell and atomic positions using FIRE + FrechetCellFilter
            dyn = FIRE(
                FrechetCellFilter(atoms), logfile=None
            )  # Set logfile to None to suppress output
            dyn.run(fmax=fmax, steps=500)

            # 4. Final energy in eV
            final_energy_eV = atoms.get_potential_energy()

            # 5. Convert relaxed ASE Atoms -> Pymatgen Structure -> CIF
            final_structure_pmg = AseAtomsAdaptor.get_structure(atoms)
            final_relaxed_cif = final_structure_pmg.to(fmt="cif")

            # Add results to the dictionary
            r["relaxed_energy"] = final_energy_eV
            r["relaxed_cif"] = final_relaxed_cif
            # Add other initial info if needed, e.g., initial energy requires separate calc
            # r["initial_energy"] = atoms.get_potential_energy() # Would need calc before relax
            res.append(r)

            # Perform garbage collection every 10 iterations
            iter_count += 1
            if iter_count % 10 == 0:
                # Run Python's garbage collector
                gc.collect()
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"Memory cleaned after processing {iter_count} structures")

        except Exception as e:
            print(f"Error processing structure {i}: {e}")
            continue

    # Final garbage collection before returning
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return the results as a DataFrame
    return pd.DataFrame(res)
