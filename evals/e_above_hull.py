"""
This file depends on and heavily modifies code from Meta's crystal-text-llm repository, which is MIT-licensed.
The original license is preserved.
"""

"""
Build a PatchedPhaseDiagram from all MP ComputedStructureEntries for calculating
DFT-ground truth convex hull energies.
"""

import warnings
import tempfile
import os

from pymatgen.core import Structure
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet

import tqdm

from pymatgen.core.structure import Structure

from evals.llm_utils import cif_str_to_crystal
from evals.relaxations import (
    m3gnet_relaxed_energy,
    eqv2_relaxed_energy,
    chgnet_relaxed_energy,
    chgnet_relaxed_energy_batch,
    eqv2_relaxed_energy_batch,
    esen_relaxed_energy_batch,
)

from timeout_decorator import timeout
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from pathlib import Path


# Define the worker function at the top level for CHGNet
def run_chgnet_batch_job(gen_file, steps, index):
    """Worker function for parallel relaxation using chgnet_relaxed_energy_batch."""
    try:
        return chgnet_relaxed_energy_batch(gen_file=gen_file, steps=steps, index=index)
    except Exception as e:
        print(
            f"Error in CHGNet job processing indices {index.min()}-{index.max()}: {e}"
        )
        # Return an empty DataFrame or None to indicate failure
        return pd.DataFrame()


# Define the worker function at the top level for EQV2
def run_eqv2_batch_job(gen_file, fmax, index):
    """Worker function for parallel relaxation using eqv2_relaxed_energy_batch."""
    try:
        return eqv2_relaxed_energy_batch(gen_file=gen_file, fmax=fmax, index=index)
    except Exception as e:
        print(f"Error in EQV2 job processing indices {index.min()}-{index.max()}: {e}")
        # Return an empty DataFrame or None to indicate failure
        return pd.DataFrame()


def run_esen_batch_job(gen_file, fmax, index):
    """Worker function for parallel relaxation using esen_relaxed_energy_batch."""
    try:
        return esen_relaxed_energy_batch(gen_file=gen_file, fmax=fmax, index=index)
    except Exception as e:
        print(f"Error in ESEN job processing indices {index.min()}-{index.max()}: {e}")
        # Return an empty DataFrame or None to indicate failure
        return pd.DataFrame()


@timeout(30)
def call_relaxed_energy(cif_str, relaxer):
    if relaxer == "m3gnet":
        return m3gnet_relaxed_energy(cif_str)
    elif relaxer == "eqv2":
        return eqv2_relaxed_energy(cif_str)
    elif relaxer == "chgnet":
        return chgnet_relaxed_energy(cif_str)


def label_energies(filename, relaxer):
    df = pd.read_csv(filename)
    new_df = []
    for r in tqdm.tqdm(df.to_dict("records")):
        cif_str = r["cif"]

        crystal = cif_str_to_crystal(cif_str)
        if crystal is None or not crystal.valid:
            print("not walid")
            continue

        structure = Structure.from_str(cif_str, fmt="cif")
        if len(structure) == 1:
            print("broken")
            continue

        try:
            e, relaxed_s = call_relaxed_energy(cif_str, relaxer)
            r["relaxed_energy"] = e
            r["relaxed_cif"] = relaxed_s.to(fmt="cif")
        except Exception as e:
            print(e)
            continue

        new_df.append(r)

    new_df = pd.DataFrame(new_df)
    filename = filename.split("/")[-1]
    new_filename = filename.replace(".csv", "") + "_relaxed_energy.csv"
    new_filename = "evals/results/" + new_filename
    new_df.to_csv(new_filename)


def label_energies_batched(
    filename: Path,
    out_folder: str | None,
    relaxer_type: str,  # Added relaxer type ('chgnet' or 'eqv2')
    num_jobs: int,
    num_structures: int | None,
    steps: int = 1500,  # Default for chgnet
    fmax: float = 0.05,  # Default for eqv2
):
    """Label energies for structures in a CSV file using batched relaxation in parallel.

    Args:
        filename (Path): Path to the input CSV file with structure data.
        relaxer_type (str): The type of relaxer to use ('chgnet' or 'eqv2').
        num_jobs (int): Number of parallel jobs to run.
        num_structures (int | None): Maximum number of structures to process from the file.
                                   If None, process all structures.
        steps (int): Maximum number of relaxation steps (used by CHGNet).
        fmax (float): Maximum force convergence criterion (used by EQV2).
    """
    # Load CSV file
    df_full = pd.read_csv(filename)
    print(f"Reading file: {filename}")
    print(f"Output folder: {out_folder}")
    # Determine number of structures to process
    total_structures = len(df_full)
    if num_structures is not None:
        assert num_structures <= total_structures
        process_count = num_structures
    else:
        process_count = total_structures
    print(f"Total structures in file: {total_structures}")
    print(f"Processing {process_count} structures using {relaxer_type}_batched")

    # Split indices into chunks
    index = np.arange(process_count)
    indexes = np.array_split(index, num_jobs)
    indexes = [i for i in indexes if i.size > 0]  # Remove empty chunks

    # Prepare arguments and select worker function based on relaxer_type
    job_args = []
    if relaxer_type == "chgnet_batched":
        worker_func = run_chgnet_batch_job
        for idx_chunk in indexes:
            job_args.append((filename, steps, idx_chunk))
    elif relaxer_type == "eqv2_batched":
        worker_func = run_eqv2_batch_job
        for idx_chunk in indexes:
            job_args.append((filename, fmax, idx_chunk))
    elif relaxer_type == "esen_batched":
        worker_func = run_esen_batch_job
        for idx_chunk in indexes:
            job_args.append((filename, fmax, idx_chunk))
    else:
        raise ValueError(
            f"Unsupported relaxer_type: {relaxer_type}. Choose 'chgnet' or 'eqv2'."
        )

    print(f"Starting relaxation with {len(indexes)} parallel jobs...")

    # Process each chunk in parallel locally
    # Ensure spawn method is set in the main execution block
    with Pool(num_jobs) as pool:
        results_list = pool.starmap(
            worker_func,
            job_args,
        )

    print("Parallel jobs finished.")
    # Combine results
    # Filter out None/empty results in case of errors in worker processes
    valid_results = [df for df in results_list if df is not None and not df.empty]

    if not valid_results:
        print("No results were successfully generated or returned by any job.")
        return

    print(
        f"Successfully combined results from {len(valid_results)}/{len(indexes)} jobs."
    )

    all_results_df = pd.concat(valid_results, ignore_index=True)

    # Save the final combined DataFrame
    base_filename = Path(filename).stem  # Get filename without extension

    if out_folder:
        new_filename_path = (
            Path("evals/results")
            / out_folder
            / f"{base_filename}_{relaxer_type.replace('_batched', '')}_relaxed_energy.csv"
        )
    else:
        # Include relaxer type in the output filename for clarity
        new_filename_path = (
            Path("evals/results")
            / f"{base_filename}_{relaxer_type.replace('_batched', '')}_relaxed_energy.csv"
        )

    # Ensure the parent directory exists
    new_filename_path.parent.mkdir(parents=True, exist_ok=True)

    all_results_df.to_csv(new_filename_path, index=False)
    print(f"Wrote final combined results to: {new_filename_path}")
    # Return the filename for the next step
    return str(
        new_filename_path
    )  # Return as string, as expected by downstream functions


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
    # Loading computed entries as our benchmark --- SUBJECT TO CHANGE IN DIRECTORY
    data_path = "computed-structure-entries.json.gz"

    print(f"Loading MP ComputedStructureEntries from {data_path}")
    df = pd.read_json(data_path)

    # filter to only df entries that contain the substring "GGA" in the column 'index'
    # df = df[df['index'].str.contains("GGA")]
    df = df[df["entry"].apply(lambda x: "GGA" in x["entry_id"])]
    print(len(df))

    mp_computed_entries = [
        ComputedEntry.from_dict(x)
        for x in tqdm.tqdm(df.entry)
        if "GGA" in x["parameters"]["run_type"]
    ]
    mp_computed_entries = [
        entry
        for entry in mp_computed_entries
        if not np.any(["R2SCAN" in a.name for a in entry.energy_adjustments])
    ]

    ppd_mp = PatchedPhaseDiagram(mp_computed_entries, verbose=True)

    print("READING FROM RELAXED CSV: " + fn)

    df = pd.read_csv(fn)

    new_df = []
    for d in tqdm.tqdm(df.to_dict(orient="records")):
        try:
            structure = Structure.from_str(d["relaxed_cif"], fmt="cif")
            energy = d["relaxed_energy"]

            cse = generate_CSE(structure, energy)
            e_above_hull = ppd_mp.get_e_above_hull(cse, allow_negative=True)

            d["e_above_hull"] = e_above_hull
            new_df.append(d)
        except Exception as e:
            print(e)
            continue

    new_df = pd.DataFrame(new_df)

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

    # Ensure spawn start method is set for CUDA compatibility with multiprocessing
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="data/llm/relaxations/relaxations.csv"
    )
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument("--relaxer", type=str, default="m3gnet")
    # Add arguments needed for label_energies_batched if you plan to use it directly
    parser.add_argument("--steps", type=int, default=1500, help="Max relaxation steps")
    parser.add_argument(
        "--fmax", type=float, default=0.05, help="Max force criterion (for EQV2)"
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=10,
        help="Number of parallel jobs for batched relaxers",
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=None,
        help="Limit number of structures to process",
    )

    args = parser.parse_args()

    # suppress tensorflow warnings
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # import tensorflow as tf # type: ignore

    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # relaxed_filename = "evals/results/alex_8b_wyckoff_210k_temp_0.7_relaxed_energy.csv"
    relaxed_filename = None
    # Decide which function to call based on arguments
    if args.relaxer in ["chgnet_batched", "eqv2_batched", "esen_batched"]:
        output_dir = Path("evals/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        relaxed_filename = label_energies_batched(
            filename=Path(args.filename),
            out_folder=args.out_folder,
            relaxer_type=args.relaxer,
            steps=args.steps,
            num_jobs=args.num_jobs,
            num_structures=args.num_structures,
            fmax=args.fmax,
        )
    else:
        # Original non-batched path
        label_energies(args.filename, args.relaxer)
        filename_str = args.filename.split("/")[-1]
        relaxed_filename = filename_str.replace(".csv", "_relaxed_energy.csv")
        relaxed_filename = "evals/results/" + relaxed_filename

    warnings.filterwarnings("ignore")
    get_e_above_hull(relaxed_filename)
