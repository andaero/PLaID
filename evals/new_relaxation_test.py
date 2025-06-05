import warnings
import tempfile

from pymatgen.core.structure import Structure
import numpy as np
import pandas as pd

import tqdm
import pandas as pd

#from m3gnet.models import Relaxer
# import matgl
# from matgl.ext.ase import PESCalculator, MolecularDynamics, Relaxer
#from matgl.ext.calculator import M3GNetCalculator

import io
from ase.io import read
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import OCPCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure


from basic_eval import cif_str_to_crystal


def m3gnet_relaxed_energy(cif_str):
  """
  structure = Structure.from_str(cif_str, fmt="cif")
  
  pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
  relaxer = Relaxer(potential=pot)  # This loads the default pre-trained model
  
  relax_results = relaxer.relax(structure)
  final_structure = relax_results['final_structure']
  final_energy_per_atom = float(relax_results['trajectory'].energies[-1])
  return final_energy_per_atom, final_structure
  """
  # 1. Convert CIF string to ASE Atoms
  cif_buffer = io.StringIO(cif_str)
  atoms = read(cif_buffer, format="cif")

  # 2. Attach the OCP calculator
  calc = OCPCalculator(checkpoint_path="OMAT24 eqV2 86M.pt")
  atoms.calc = calc

  # 3. Relax both cell and atomic positions using FIRE + FrechetCellFilter
  dyn = FIRE(FrechetCellFilter(atoms))
  dyn.run(fmax=0.05)

  # 4. Final energy in eV, then convert to per-atom
  final_energy_eV = atoms.get_potential_energy()
  final_energy_per_atom = final_energy_eV / len(atoms)

  # 5. Convert relaxed ASE Atoms -> Pymatgen Structure
  final_structure = AseAtomsAdaptor.get_structure(atoms)

  return final_energy_per_atom, final_structure

from timeout_decorator import timeout

@timeout(30)
def call_m3gnet_relaxed_energy(cif_str):
  return m3gnet_relaxed_energy(cif_str)

def label_energies(filename):
  df = pd.read_csv(filename)
  
  df = df.head(3)

  new_df = []
  for r in tqdm.tqdm(df.to_dict('records')):
      cif_str = r['cif']

      crystal = cif_str_to_crystal(cif_str)
      if crystal is None or not crystal.valid:
          print("NOT WALID")
          continue

      structure = Structure.from_str(cif_str, fmt="cif")
      if len(structure) == 1:
          continue


      # BROKENNNNN
      e, relaxed_s = call_m3gnet_relaxed_energy(cif_str)
      r['m3gnet_relaxed_energy'] = e
      r['m3gnet_relaxed_cif'] = relaxed_s.to(fmt="cif")

      new_df.append(r)

  new_df = pd.DataFrame(new_df)
  new_filename = filename.replace(".csv","") + "_TEST_m3gnet_relaxed_energy.csv"
  new_df.to_csv(new_filename)

label_energies('llm_samples_temp0.7.csv')