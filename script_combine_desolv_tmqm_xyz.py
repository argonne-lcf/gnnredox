
from fhb_helpers import *

import numpy as np

import pandas as pd
import numpy as np
from ase.io import read, write
from ase import Atoms

import os

TMQM_REPO_PATH = './tmQm/tmQM'   # path to the cloned tmQM repo 

desolv_atoms_dict = {}
desolv_atoms = read('./Data/desolvated_all_tmcs.xyz', format='extxyz', index=':')
csd_codes = []
for atoms in desolv_atoms:
    code = atoms.info['csd_code']
    csd_codes.append(code)
    desolv_atoms_dict[code] = atoms

# tmQM files

xyz_files = ['tmQM_X1.xyz', 'tmQM_X2.xyz', 'tmQM_X3.xyz']
# Parse XYZ files and create ASE Atoms objects
tmqm_atoms_dict = {}
found_csd_codes = set()
for xyz_file in xyz_files:
    fpath = os.path.join(TMQM_REPO_PATH, xyz_file)
    atoms_dict, found_csd_codes = parse_tmqm_xyz_file(fpath, csd_codes, found_csd_codes)
    tmqm_atoms_dict.update(atoms_dict)

# Create DataFrame
codes = list(desolv_atoms_dict.keys())
desolvated_atoms_obj = list(desolv_atoms_dict.values())
tmqm_atoms_obj = [tmqm_atoms_dict[code] for code in codes]  # Explicitly match by key

# Create DataFrame
df = pd.DataFrame({
    'csd_code': codes,
    'desolvated_atoms_obj': desolvated_atoms_obj,
    'tmqm_atoms_obj': tmqm_atoms_obj
})

df.to_pickle('./Data/desolvated_tmqm_all_xyz.pkl')