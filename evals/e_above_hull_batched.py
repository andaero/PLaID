"""
This file depends on and heavily modifies code from Meta's crystal-text-llm repository, which is MIT-licensed.
The original license is preserved.
"""

import io
import warnings
import tempfile
import os

# fairchem v1 imports scipy.special.sph_harm which was removed in scipy >=1.15.
# Patch it back using the renamed sph_harm_y (args are reordered: n,m swapped).
import scipy.special as _sp

if not hasattr(_sp, "sph_harm"):
    _sp.sph_harm = lambda m, n, theta, phi: _sp.sph_harm_y(n, m, theta, phi)

from pymatgen.core import Structure
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet

import torch
import torch_sim as ts
from torch_sim.models.fairchem import FairChemModel
from ase.io import read

import tqdm

from pymatgen.core.structure import Structure

from evals.llm_utils import cif_str_to_crystal
from pathlib import Path

CHECKPOINT_PATHS = {
    "eqv2_batched": "evals/OCPCalcs/eqV2_86M_omat_mp_salex.pt",
    "esen_batched": "evals/OCPCalcs/esen_30m_oam.pt",
}

# Pre-estimated max_memory_scaler values (measured on this GPU), scaled to 85%.
MAX_MEMORY_SCALERS = {
    "eqv2_batched": 26002.31,  # ~21591.96
    "esen_batched": 25402.31,  # TODO: estimate and fill in
}


def label_energies_batched(
    filename: Path,
    out_folder: str | None,
    relaxer_type: str,
    num_structures: int | None,
):
    """Label energies using torch-sim InFlightAutoBatcher for GPU-batched relaxation.

    Args:
        filename: Path to the input CSV file with structure data.
        out_folder: Optional subdirectory under evals/results/ for output.
        relaxer_type: The type of relaxer ('eqv2_batched' or 'esen_batched').
        num_structures: Maximum number of structures to process. None = all.
    """
    checkpoint = CHECKPOINT_PATHS.get(relaxer_type)
    if checkpoint is None:
        raise ValueError(
            f"Unsupported relaxer_type: {relaxer_type}. "
            f"Choose from: {list(CHECKPOINT_PATHS.keys())}"
        )

    df_full = pd.read_csv(filename)
    print(f"Reading file: {filename}")
    print(f"Output folder: {out_folder}")

    total_structures = len(df_full)
    if num_structures is not None:
        assert num_structures <= total_structures
        process_count = num_structures
    else:
        process_count = total_structures
    print(f"Total structures in file: {total_structures}")
    print(f"Processing {process_count} structures using {relaxer_type} (torch-sim)")

    # Validate structures and convert to ASE Atoms
    atoms_list = []
    valid_indices = []
    for i in tqdm.tqdm(range(process_count), desc="Validating structures"):
        r = df_full.iloc[i]
        cif_str = r["cif"]

        crystal = cif_str_to_crystal(cif_str, compute_fingerprints=False)
        if crystal is None or not crystal.valid:
            print(f"Skipping structure {i}: Invalid crystal")
            continue

        structure = Structure.from_str(cif_str, fmt="cif")
        if len(structure) == 1:
            print(f"Skipping structure {i}: Single atom structure")
            continue

        try:
            atoms = read(io.StringIO(cif_str), format="cif")
            atoms_list.append(atoms)
            valid_indices.append(i)
        except Exception as e:
            print(f"Skipping structure {i}: Failed to parse CIF: {e}")
            continue

    print(f"Valid structures: {len(atoms_list)}/{process_count}")

    # Pre-allocate results DataFrame with all rows; invalid rows keep NaN.
    all_results_df = df_full.head(process_count).copy()
    all_results_df["relaxed_energy"] = np.nan
    all_results_df["relaxed_cif"] = np.nan

    if not atoms_list:
        print("No valid structures to process.")
    else:
        # Load model
        use_cpu = not torch.cuda.is_available()
        print(f"Loading FairChemModel from {checkpoint} (cpu={use_cpu})")
        model = FairChemModel(
            model=checkpoint,
            cpu=use_cpu,
            compute_stress=True,
        )

        # Run batched relaxation with InFlightAutoBatcher
        max_scaler = MAX_MEMORY_SCALERS.get(relaxer_type)
        if max_scaler is not None:
            print(f"Using pre-estimated max_memory_scaler={max_scaler:.1f}")
            batcher = ts.InFlightAutoBatcher(
                model=model,
                max_memory_scaler=max_scaler,
            )
        else:
            print("No pre-estimated scaler; auto-estimating GPU memory (slow)...")
            batcher = True

        print("Starting batched relaxation: max_steps=500, fmax=0.02")
        optimize_kwargs = dict(
            model=model,
            optimizer=ts.frechet_cell_fire,
            convergence_fn=ts.generate_force_convergence_fn(0.02),
            autobatcher=batcher,
            max_steps=500,
            pbar=True,
        )
        try:
            final_state = ts.optimize(system=atoms_list, **optimize_kwargs)
        except RuntimeError as e:
            if "expanded size" not in str(e):
                raise
            print(f"Batch relaxation failed: {e}")
            print("Pre-screening structures to find problematic ones...")
            good_mask = []
            for idx, atoms in enumerate(atoms_list):
                try:
                    state = ts.initialize_state(
                        atoms, model.device, model.dtype
                    )
                    model(state)
                    good_mask.append(True)
                except RuntimeError:
                    print(
                        f"  Removing structure {valid_indices[idx]}:"
                        " fails eqV2 forward pass"
                    )
                    good_mask.append(False)
            atoms_list = [a for a, ok in zip(atoms_list, good_mask) if ok]
            valid_indices = [v for v, ok in zip(valid_indices, good_mask) if ok]
            print(f"Retrying with {len(atoms_list)} structures")
            final_state = ts.optimize(system=atoms_list, **optimize_kwargs)

        # Extract results and map back to valid indices
        energies = final_state.energy.detach().cpu().numpy()
        relaxed_structures = final_state.to_structures()

        for orig_i, energy, struct in zip(valid_indices, energies, relaxed_structures):
            all_results_df.at[orig_i, "relaxed_energy"] = float(energy)
            all_results_df.at[orig_i, "relaxed_cif"] = struct.to(fmt="cif")

    # Save output
    base_filename = Path(filename).stem
    relaxer_name = relaxer_type.replace("_batched", "")
    if out_folder:
        new_filename_path = (
            Path("evals/results")
            / out_folder
            / f"{base_filename}_{relaxer_name}_relaxed_energy.csv"
        )
    else:
        new_filename_path = (
            Path("evals/results") / f"{base_filename}_{relaxer_name}_relaxed_energy.csv"
        )

    new_filename_path.parent.mkdir(parents=True, exist_ok=True)
    all_results_df.to_csv(new_filename_path, index=False)
    print(f"Wrote results to: {new_filename_path}")
    return str(new_filename_path)


def generate_CSE(structure, m3gnet_energy):
    # Write VASP inputs files as if we were going to do a standard MP run
    # this is mainly necessary to get the right U values / etc
    b = MPRelaxSet(structure)
    with tempfile.TemporaryDirectory() as tmpdirname:
        b.write_input(f"{tmpdirname}/", potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

    # Get the U values and figure out if we should have run a GGA+U calc
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"

    # Make a ComputedStructureEntry without the correction
    cse_d = {
        "structure": clean_structure,
        "energy": m3gnet_energy,
        "correction": 0.0,
        "parameters": param,
    }

    # Apply the MP 2020 correction scheme (anion/+U/etc)
    cse = ComputedStructureEntry.from_dict(cse_d)
    _ = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse,
        clean=True,
    )

    # Return the final CSE (notice that the composition/etc is also clean, not things like Fe3+)!
    return cse


def get_e_above_hull(fn):
    import gzip
    import pickle

    ppd_path = "evals/2023-02-07-ppd-mp.pkl.gz"
    print(f"Loading pre-built PatchedPhaseDiagram from {ppd_path}")
    with gzip.open(ppd_path, "rb") as f:
        ppd_mp = pickle.load(f)

    print("READING FROM RELAXED CSV: " + fn)

    df = pd.read_csv(fn)

    df["e_above_hull"] = np.nan
    for idx, d in tqdm.tqdm(enumerate(df.to_dict(orient="records")), total=len(df)):
        if pd.isna(d.get("relaxed_cif")):
            continue
        try:
            structure = Structure.from_str(d["relaxed_cif"], fmt="cif")
            energy = d["relaxed_energy"]

            cse = generate_CSE(structure, energy)
            e_above_hull = ppd_mp.get_e_above_hull(cse, allow_negative=True)

            df.at[idx, "e_above_hull"] = e_above_hull
        except Exception as e:
            print(e)
            continue

    new_df = df

    print("WRITING TO CSV: " + fn.replace("_relaxed_energy.csv", "_ehull_results.csv"))

    new_df.to_csv(fn.replace("_relaxed_energy.csv", "_ehull_results.csv"), index=False)
    try:
        print(f"DELETING intermediate file: {fn}")
        os.remove(fn)
    except FileNotFoundError:
        print(f"Warning: Could not find file {fn} to delete.")
    except Exception as e:
        print(f"Warning: Could not delete file {fn}. Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="data/llm/relaxations/relaxations.csv"
    )
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument("--relaxer", type=str, default="eqv2_batched")
    parser.add_argument(
        "--num_structures",
        type=int,
        default=None,
        help="Limit number of structures to process",
    )
    parser.add_argument(
        "--bandgap",
        action="store_true",
        help="Save results to evals/results/bandgap/ instead of evals/results/",
    )

    args = parser.parse_args()

    if args.bandgap and args.out_folder is None:
        args.out_folder = "bandgap"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    output_dir = Path("evals/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    relaxed_filename = label_energies_batched(
        filename=Path(args.filename),
        out_folder=args.out_folder,
        relaxer_type=args.relaxer,
        num_structures=args.num_structures,
    )

    warnings.filterwarnings("ignore")
    get_e_above_hull(relaxed_filename)
