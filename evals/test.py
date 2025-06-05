import torch
import torch_sim as ts
import sevenn.util
from sevenn.calculator import SevenNetCalculator
from torch_sim.models import SevenNetModel
import time

from ase.io import read
import io
import pandas as pd
from pymatgen.core import Structure
from basic_eval import cif_str_to_crystal

# run natively on gpus
device = torch.device("cuda")


def torch_sim_relaxed_energy(cif_strings):
    # Initialize the model correctly from checkpoint instead of using calculator
    cp = sevenn.util.load_checkpoint("7net-mf-ompa")
    backend = "e3nn"
    model_loaded = cp.build_model(backend)
    model_loaded.set_is_batch_data(False)
    model_loaded = model_loaded.to(device)

    # Create the TorchSim model wrapper
    sevenn_model = SevenNetModel(model=model_loaded, modal="mpa", device=device)
    print("loaded sevenn model")

    # Read CIF strings and convert to ASE Atoms
    atoms_list = []
    for cif in cif_strings:
        crystal = cif_str_to_crystal(cif)

        if crystal is None or not crystal.valid:
            print("not walid")
            continue

        structure = Structure.from_str(cif, fmt="cif")
        if len(structure) == 1:
            print("broken")
            continue
        atoms_list.append(structure)

    # Integrate the system using torch_sim
    final_state = ts.integrate(
        system=atoms_list,
        model=sevenn_model,
        n_steps=50,
        timestep=0.002,
        temperature=1000,
        integrator=ts.nvt_langevin,
        autobatcher=True,
    )
    print("completed MD")

    # relax all of the high temperature states
    relaxed_state = ts.optimize(
        system=final_state,
        model=sevenn_model,
        optimizer=ts.frechet_cell_fire,
        max_steps=1500,
        autobatcher=True,
    )

    # Calculate energy per atom for each structure
    energies_per_atom = relaxed_state.energy / relaxed_state.n_atoms

    energies_per_atom = []
    for energy, atoms in zip(relaxed_state.energy, relaxed_state.to_atoms()):
        energies_per_atom.append(energy / len(atoms))

    relaxed_structures = relaxed_state.to_structures()

    return energies_per_atom, relaxed_structures


def time_torch_sim_relaxed_energy(cif_path, num_structures=None):
    # Read all CIF strings from file
    df = pd.read_csv(cif_path)

    # Use all structures if num_structures is not provided
    if num_structures is None:
        cif_strings = df["cif"]
        print(f"Using all {len(cif_strings)} structures from the file")
    else:
        cif_strings = df["cif"].iloc[:num_structures]
        print(f"Using {num_structures} structures from the file")

    # Measure execution time
    start_time = time.time()
    energies_per_atom, relaxed_structures = torch_sim_relaxed_energy(cif_strings)
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    num_processed = len(cif_strings)

    print(f"Energies per atom: {energies_per_atom}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per structure: {elapsed_time / num_processed:.2f} seconds")

    return energies_per_atom, relaxed_structures, elapsed_time


if __name__ == "__main__":
    # Use all structures by default
    time_torch_sim_relaxed_energy(
        "evals/results/llm_8b_wyckoff_temp_0.7_ehull_results.csv",
        num_structures=20,
    )
