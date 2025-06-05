"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import sys

sys.path.append(
    "/home/compmat/larwang/flowmm/src"
)  # this line gets the flowmm.pandas_ and flowmm.pymatgen_ dependencies working.

import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
import gzip

import pandas as pd
from pymatgen.io.vasp import Vasprun
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry, PhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Poscar
from tqdm import tqdm

from flowmm.pandas_ import filter_prerelaxed, maybe_get_missing_columns
from flowmm.pymatgen_ import COLUMNS_COMPUTATIONS, to_structure

EntriessType = list[list[ComputedStructureEntry]]
EntryDictssType = list[list[dict]]
PATH_PPD_MP = Path(__file__).parents[1] / "mp_02072023/2023-02-07-ppd-mp.pkl.gz"

NAME_FN_FOR_PATH = {
    "original_index": lambda path: int(path.stem.split("_")[-1]),
    "method": lambda path: path.stem.rsplit("_", maxsplit=1)[0],
}

NAME_FN_FOR_TRAJ = {
    "energy_initial": lambda traj: traj[0].get_potential_energy(),
    "energy": lambda traj: traj[-1].get_potential_energy(),
    "forces": lambda traj: traj[-1].get_forces(),
    # "stress": lambda traj: traj[-1].get_stress(),
    # "mag_mom": lambda traj: traj[-1].get_magnetic_moment(),
    "num_sites": lambda traj: traj[-1].get_global_number_of_atoms(),
}


def get_compat_params(dft_path):
    incar_file = os.path.join(dft_path, "INCAR")
    poscar_file = os.path.join(dft_path, "POSCAR")
    incar = Incar.from_file(incar_file)
    poscar = Poscar.from_file(poscar_file)
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"
    return param


def get_energy_correction(ase_atoms, raw_energy, dft_path):
    params = get_compat_params(dft_path)
    struct = AseAtomsAdaptor.get_structure(ase_atoms)

    cse_d = {
        "structure": struct,
        "energy": raw_energy,
        "correction": 0.0,
        "parameters": params,
    }
    cse = ComputedStructureEntry.from_dict(cse_d)
    print(f"before correction: {cse}")
    print(f"before correction energy: {cse.energy}")
    out = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse,
        # verbose=True,
        clean=True,
    )
    print(f"after correction: {cse}")
    print(f"after correction energy: {cse.energy}")
    pde = PDEntry(composition=cse.composition, energy=cse.energy)
    return (cse, cse.energy, pde)


def apply_name_fn_dict(
    obj: any,
    name_fn: dict[str, callable],
) -> dict[str, any]:
    return {prop: fn(obj) for prop, fn in name_fn.items()}


def get_record(file: Path, dft_path: Path) -> dict[str, any]:
    # 1. Everything that comes from the filename stays the same
    record = apply_name_fn_dict(file, NAME_FN_FOR_PATH)

    # 2. Replace Trajectory load with VASP parser
    vasp_dir = dft_path  # dft_path already contains the full path to the crystal directory
    print(f"vasp_dir: {vasp_dir}")
    vasp_data, raw_energy, last_atoms = _parse_vasp(vasp_dir)
    record.update(vasp_data)

    # 3. Compatibility corrections (same as before, but use last_atoms)
    cse, corrected_energy, pde = get_energy_correction(last_atoms, raw_energy, vasp_dir)
    record["corrected_energy"] = corrected_energy
    record["computed_structure_entry"] = cse.as_dict()
    record["phase_diagram_entry"] = pde.as_dict()
    return record


def _parse_vasp(run_dir: Path):
    vr = Vasprun(run_dir / "vasprun.xml")

    # Get initial energy directly from vasprun object
    if not vr.ionic_steps:
        # Handle case with no ionic steps (e.g., static calc or error)
        energy_initial = float("nan")  # Or vr.final_energy if appropriate
        forces = None  # No forces if no ionic steps
    else:
        energy_initial = vr.ionic_steps[0]["e_0_energy"]
        # Get forces from the last ionic step
        forces = vr.ionic_steps[-1].get("forces", None)

    # Get final structure and convert to ASE *only* for compatibility function
    atoms_final = AseAtomsAdaptor.get_atoms(vr.final_structure)
    print("final structure: ", atoms_final)

    # Get final energy directly from vasprun object
    energy_final = vr.final_energy

    data = {
        "energy_initial": energy_initial,
        "energy": energy_final,
        "forces": forces,  # Get forces directly from vasprun object
        "num_sites": len(atoms_final),
    }
    print("data: ", data)
    return data, energy_final, atoms_final


def get_dft_results(
    model: Path,
    # n_jobs: int = 1
) -> pd.DataFrame:
    dft_path = Path("dft") / model
    
    # Get all crystal directories
    crystal_dirs = []
    for crystal_dir in Path(dft_path).iterdir():
        if crystal_dir.is_dir():
            crystal_dirs.append((crystal_dir.name))
    
    # Process all crystals
    print(crystal_dirs)
    
    records = []
    for crystal in tqdm(crystal_dirs, desc="Processing crystals"):
        try:
            # Construct the full path to the crystal directory
            crystal_path = dft_path / crystal
            record = get_record(Path(f"{crystal}"), crystal_path)
            records.append(record)
        except Exception as e:
            print(f"Error processing {crystal}: {str(e)}")
            continue
    
    df = pd.DataFrame.from_records(records)
    if not df.empty:  # Only try to map method if we have records
        df["method"] = df["method"].map(
            {
                "cdvae": "cdvae",
                "diffcsp_mp20": "diffcsp_mp20",
                "diffscp_mp20": "diffcsp_mp20",  # spelling error
            }
        )
        df["e_per_atom_dft"] = df["energy"] / df["num_sites"]
        df["e_per_atom_dft_initial"] = df["energy_initial"] / df["num_sites"]
    return df


def get_patched_phase_diagram_mp(path: Path) -> PatchedPhaseDiagram:
    # Check if the file has a .gz extension
    if path.suffix == ".gz":
        # Open as gzip file
        with gzip.open(path, "rb") as f:
            ppd_mp = pickle.load(f)
    else:
        # Open as regular file
        with open(path, "rb") as f:
            ppd_mp = pickle.load(f)
    return ppd_mp


def get_e_hull_from_phase_diagram(
    phase_diagram: PhaseDiagram | PatchedPhaseDiagram,
    structure: Structure | dict,
) -> float:
    """returns e_hull_per_atom"""
    structure = to_structure(structure)
    try:
        return phase_diagram.get_hull_energy_per_atom(structure.composition)
    except (ValueError, AttributeError, ZeroDivisionError):
        return float("nan")


def get_e_hull_per_atom_from_pymatgen(
    phase_diagram: PhaseDiagram | PatchedPhaseDiagram,
    pde: PDEntry | dict,
) -> tuple[dict, float]:
    """returns e_hull_per_atom"""
    pde = PDEntry.from_dict(pde)
    try:
        out = phase_diagram.get_decomp_and_e_above_hull(pde, allow_negative=True)
    except (ValueError, AttributeError, ZeroDivisionError):
        out = ({}, float("nan"))
    return out


def main(args: Namespace) -> None:
    # load the data to compare to the hull
    print("readying json_in")
    # df = pd.read_json(args.json_in)
    # print("potentially getting missing columns")
    # df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)
    # print("filtering to those which are prerelaxed")
    # if args.maximum_nary is not None:
    #     print(f"and maximum nary={args.maximum_nary}")
    # df = filter_prerelaxed(
    #     df,
    #     args.num_structures,
    #     maximum_nary=args.maximum_nary,
    # )

    print(f"loading the saved mp phase diagram at {PATH_PPD_MP=}")
    ppd_mp = get_patched_phase_diagram_mp(PATH_PPD_MP)
    # e_hulls = [get_e_hull_from_phase_diagram(ppd_mp, s) for s in df["structure"]]
    # out = pd.DataFrame(data={"e_hull_per_atom": e_hulls})
    # out.index = df.index  # this works because we filtered out exceptions above!
    # out["e_above_hull_per_atom_chgnet_gen"] = (df["e_gen"] / df["num_sites"]) - out[
    #     "e_hull_per_atom"
    # ]
    # out["e_above_hull_per_atom_chgnet"] = (df["e_relax"] / df["num_sites"]) - out[
    #     "e_hull_per_atom"
    # ]

    df_dft = get_dft_results(args.root_dft_clean_outputs)
    decomp_and_e = [
        get_e_hull_per_atom_from_pymatgen(ppd_mp, p)
        for p in df_dft["phase_diagram_entry"]
    ]

    decomposition, e_above_hull_dft_per_atom_corrected = zip(*decomp_and_e)
    df_dft["e_above_hull_per_atom_dft_corrected"] = list(
        e_above_hull_dft_per_atom_corrected
    )
    df_dft["decomposition"] = list(decomposition)
    print(df_dft.head())

    if args.method is not None:
        df_dft["method"] = args.method
    # df_dft = df_dft[df_dft["original_index"].isin(df.index)]
    # df_dft = df_dft.set_index("original_index")
    # out["e_above_hull_per_atom_dft"] = df_dft["e_per_atom_dft"] - out["e_hull_per_atom"]
    # out["e_above_hull_per_atom_dft_initial"] = (
    #     df_dft["e_per_atom_dft_initial"] - out["e_hull_per_atom"]
    # )

    # out["e_above_hull_per_atom_dft_corrected"] = df_dft[
    #     "e_above_hull_per_atom_dft_corrected"
    # ]
    # out["decomposition"] = df_dft["decomposition"]

    # write to file
    df_dft.to_json(args.json_out)
    print("wrote file to: ")
    print(f"{args.json_out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("json_in", type=Path, help="prerelaxed dataframe")
    parser.add_argument("json_out", type=Path, help="new dataframe")
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument(
        "--root_dft_clean_outputs",
        type=Path,
        default=None,
        help="root dir which contains a folder called `dft` and `clean_outputs`.",
        required=True,
    )
    parser.add_argument(
        "--maximum_nary",
        type=int,
        default=None,
        help="Any queries to structures with higher nary are avoided.",
    )
    parser.add_argument("--method", type=str, default=None)

    args = parser.parse_args()

    main(args)

