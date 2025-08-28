import multiprocessing as mp
from fhb_helpers import *

from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdchem, rdEHTTools, rdmolops, rdDetermineBonds, AllChem, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize

from collections import defaultdict

import pandas as pd
import numpy as np
from ase.io import read, write
from ase import Atoms

import time
import os
import subprocess


# Set environment variables for xTB
os.environ["PATH"] += os.pathsep + r"C:\Users\fbhuiyan\xtb-6.7.1\lib"
os.environ["XTBPATH"] = r"C:\Users\fbhuiyan\xtb-6.7.1\share\xtb"
os.environ["OMP_NUM_THREADS"] = "4" 


def calculate_max_fe_neighbor_dist_diff(row):
    tmqm_atoms = row['tmqm_atoms_obj']
    desolvated_atoms = row['desolvated_atoms_obj']

    # Ensure Atoms objects are present
    if tmqm_atoms is None or desolvated_atoms is None:
        return np.nan

    try:
        # Identify Fe atom index in tmqm_atoms
        fe_tmqm_indices = [atom.index for atom in tmqm_atoms if atom.symbol == 'Fe']
        if not fe_tmqm_indices:
            return np.nan # Fe not found in TMQM structure
        fe_tmqm_idx = fe_tmqm_indices[0]

        # Identify Fe atom index in desolvated_atoms
        fe_desolv_indices = [atom.index for atom in desolvated_atoms if atom.symbol == 'Fe']
        if not fe_desolv_indices:
            return np.nan # Fe not found in desolvated structure
        fe_desolv_idx = fe_desolv_indices[0]
    except (AttributeError, TypeError):
        # Handles cases where objects might not be ASE Atoms or are not iterable
        return np.nan


    distance_differences = []
    
    # Iterate through all atoms in tmqm_atoms to find Fe neighbors
    for i in range(len(tmqm_atoms)):
        if i == fe_tmqm_idx:
            continue  # Skip Fe atom itself

        try:
            # Calculate distance from Fe to atom i in tmqm_atoms
            dist_tmqm = tmqm_atoms.get_distance(fe_tmqm_idx, i)

            # If atom i is a neighbor in tmqm_atoms (within 6 Angstroms)
            if dist_tmqm <= 6.0:
                # Atom `i` is a TMQM neighbor.
                # We need the distance for the atom at the *same index* `i`
                # in desolvated_atoms, relative to *its* Fe atom (`fe_desolv_idx`).

                # Check if index `i` is valid in desolvated_atoms
                if i >= len(desolvated_atoms):
                    continue # Corresponding atom at index i not found in desolvated structure

                dist_desolv = desolvated_atoms.get_distance(fe_desolv_idx, i)
                
                distance_differences.append(abs(dist_tmqm - dist_desolv))
        except IndexError:
            # This might occur if an index is somehow invalid for get_distance,
            # though fe_idx checks and len checks should prevent most.
            continue 
        except Exception:
            # Catch any other ASE-related or unexpected errors during distance calculation
            continue


    if not distance_differences:
        return 0.0  # No neighbors found within 6 Angstroms, or no valid pairs
    
    return max(distance_differences)




if __name__ == '__main__':

    # Path to the tmQM repository
    TMQM_REPO_PATH = './tmQm/tmQM'

    # Load the data containing redox potential values
    # note, if you are not interested in the redox potential values, you can comment out the following two lines
    # however, you will also need to comment out the `y` in this line below when running xyz2mol conversion:
    # <line ~190> data.append({'csd_code': code, 'mol': mol, 'smiles': smiles, 'y': tmqm_redox_df[target].loc[code]})
    
    tmqm_redox_df = pd.read_csv('./Final_code/Final_data/tmqm_redox_data_full_data.csv')
    target = 'reduction_pot' #'reduction_potential_tpssh_solvent'

    desolvated_df = pd.read_pickle('Data/desolvated_tmqm_all_xyz.pkl')
    
    
    desolvated_df['max_diff'] = desolvated_df.apply(calculate_max_fe_neighbor_dist_diff, axis=1)
    desolvated_df = desolvated_df[desolvated_df['max_diff'] <= 0.7]
    

    target_csd_codes = desolvated_df['csd_code'].tolist()
    target_csd_codes_set = set(target_csd_codes)

    # charge parse from tmQM dataset (charge in tmQM dataset is calculated using DFT. Calculating charges using xTB does not match with tmQM)
    q_file_path = 'tmQm_X.q'
    fpath = os.path.join(TMQM_REPO_PATH, q_file_path)
    charge_dict = parse_tmqm_q_file(fpath, target_csd_codes_set)

    ase_atoms_dict = {}
    BO_dict = {}

    for i, row in desolvated_df.iterrows():
        csd_code = row['csd_code']
        desolvated_atoms = row['desolvated_atoms_obj']
        xyz_file = 'xtb.xyz'
        write(xyz_file, desolvated_atoms, format='extxyz')
        # Run xTB energy calculation
        xtb_result = subprocess.run(["xtb", xyz_file], capture_output=True, text=True, encoding="utf-8") # , "--opt"
        #atomic_charges = parse_xtb_q_values(xtb_result.stdout)
        #num_atoms_from_output = len(atomic_charges)
        bond_order_matrix = parse_xtb_bond_orders(xtb_result.stdout) #, num_atoms=num_atoms_from_output)

        

        ase_atoms_dict[csd_code] = desolvated_atoms
        #charge_dict[csd_code] = atomic_charges
        BO_dict[csd_code] = bond_order_matrix
    data = []

    # Iterate over the CSD codes (keys of the dictionaries)
    for csd_code in target_csd_codes:
        # Create a dictionary for each row
        row = {
            'csd_code': csd_code,
            'atoms_obj': ase_atoms_dict.get(csd_code),
            'q': charge_dict.get(csd_code),
            'bo_arr': BO_dict.get(csd_code)
        }
        # Append the row to the data list
        data.append(row)

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    #print(df)

    #-------------------#

    # ----- run the xyz2mol conversion -------#

    mols = []
    #codes = []
    #smiles_list = []
    data = []
    no_mol_idxs = [] 

    for idx, row in df.iterrows():
        #if row.csd_code not in ['AVETUQ', 'AVUHOQ']:
        #    continue
        start_time = time.perf_counter()  # high-resolution timer
        if idx:
            retry_attempts = 4
            print(f"{idx}/{len(df)} {row['csd_code']}")
            with mp.Pool(processes=1) as pool:
                for attempt in range(retry_attempts):
                    print(f"Attempt {attempt+1}:")
                    time.sleep(0.2)
                    result = pool.apply_async(process_row, (row,))
                    try:
                        # Attempt to get the result with a timeout of 3 seconds
                        code, mol, smiles = result.get(timeout=5)
                        if mol:
                            # for debug purposes
                            '''mol_2d = Chem.Mol(mol)  # Make a copy to keep 3D data intact
                            AllChem.Compute2DCoords(mol_2d)  # Generate 2D coordinates
                            image = Draw.MolToImage(mol_2d, size=(500, 500))
                            image.show()'''
                            desolv_atoms = row['atoms_obj']
                            tmqm_atoms = desolvated_df[desolvated_df['csd_code'] == code]['tmqm_atoms_obj'].values[0]
                            y = float(tmqm_redox_df[target][np.where(tmqm_redox_df['csd_code']==code)[0][0]])
                            mols.append(mol)
                            data.append({'csd_code': code, 'mol': mol, 'smiles': smiles, 'desolv_atoms': desolv_atoms, 'tmqm_atoms': tmqm_atoms,'y': y})

                            
                            elapsed_time = time.perf_counter() - start_time
                            print('Elapsed time:', elapsed_time, 'seconds\n')
                            break
                    except mp.TimeoutError:
                        elapsed_time = time.perf_counter() - start_time
                        print(f"Timeout when processing row {idx} (Elapsed time: {elapsed_time:.2f} seconds)\n")
                        if attempt == retry_attempts - 1:
                            no_mol_idxs.append(idx)
                            print("--- No mol generated ---\n")
                
        #if idx > 2:
        #    break
        
        
    print('Initial total # of tmcs:', len(df))
    print('mols generated:', len(mols))
    print('Could not generate: ',len(set(no_mol_idxs)), sorted(list(set(no_mol_idxs))))
    
    final_df = pd.DataFrame(data)
    print('Collected mol df rows:', len(final_df))
    final_df.to_pickle('./Final_code/Final_data/tmc_frm_xyz2mol_desolvated_all_df.pkl')