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

import time
import os

if __name__ == '__main__':

    # Path to the tmQM repository
    TMQM_REPO_PATH = './tmQm/tmQM'

    # Load the data containing redox potential values
    # note, if you are not interested in the redox potential values, you can comment out the following two lines
    # however, you will also need to comment out the `y` in this line below when running xyz2mol conversion:
    # <line ~146> data.append({'csd_code': code, 'mol': mol, 'smiles': smiles, 'y': avaz_df[target].loc[code]})
    
    #tmqm_redox_df = pd.read_pickle('Data/df_psi_fe_2024_60atoms2_cleaned.pkl') #('df_psi_fe_2024_50atoms_cleaned.pkl')
    #target = 'reduction_potential_tpssh_solvent'

    tmqm_redox_df = pd.read_csv('./Data/tmqm_redox_data_full_data.csv')
    target = 'reduction_pot' #'reduction_potential_tpssh_solvent'
    

    # Get the CSD codes from a file

    '''
    currently, the target CSD codes are read from a text file since we filter tmcs from the tmQM dataset (and not using the whole dataset). the file contains one column and one CSD code per row. 
    Alternatively, you can directly define the CSD codes in a list taken from the tmQM dataset. in that case, redefine the `target_csd_codes` variable as necessary.
    '''
    # get a list of target csd codes
    #target_csd_codes = tmqm_redox_df.index.tolist()
    #target_csd_codes_set = set(target_csd_codes)
    target_csd_codes = tmqm_redox_df['csd_code'].tolist()
    target_csd_codes_set = set(target_csd_codes)

    # xyz parse
    xyz_files = ['tmQM_X1.xyz', 'tmQM_X2.xyz', 'tmQM_X3.xyz']

    # Parse XYZ files and create ASE Atoms objects
    ase_atoms_dict = {}
    found_csd_codes = set()
    for xyz_file in xyz_files:
        fpath = os.path.join(TMQM_REPO_PATH, xyz_file)
        atoms_dict, found_csd_codes = parse_tmqm_xyz_file(fpath, target_csd_codes_set, found_csd_codes)
        ase_atoms_dict.update(atoms_dict)
        
        # Check if all target molecules are covered
        #found_csd_codes = set(ase_atoms_dict.keys())
        #if found_csd_codes == target_csd_codes:
        #    print('Found all target CSD codes in xyz file.')
        #    break

    # Print the result for verification
    #for csd_code, atoms in ase_atoms_dict.items():
        #print(f"CSD Code: {csd_code}", atoms.get_chemical_symbols())
        #print(atoms)


    # charge parse
    q_file_path = 'tmQm_X.q'
    fpath = os.path.join(TMQM_REPO_PATH, q_file_path)
    charge_dict = parse_tmqm_q_file(fpath, target_csd_codes_set)

    # Print the result for verification
    #for csd_code, charges in charge_dict.items():
    #    print(f"CSD_code: {csd_code}, Charges: {charges}")


    # BO parse
    BO_dict = {}
    bo_files = ['tmQM_X1.BO', 'tmQM_X2.BO', 'tmQM_X3.BO']
    found_csd_codes = set()
    for bo_file in bo_files:
        fpath = os.path.join(TMQM_REPO_PATH, bo_file)
        bond_dict, found_csd_codes = parse_tmqm_bond_order_file(fpath, target_csd_codes_set, found_csd_codes)
        BO_dict.update(bond_dict)


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
    data = []
    no_mol_idxs = [] 

    for idx, row in df.iterrows():
        #if row.csd_code not in ['ATUPEN', 'URIWUQ']:
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
                            atoms = row['atoms_obj']
                            mols.append(mol)
                            y = float(tmqm_redox_df[target][np.where(tmqm_redox_df['csd_code']==code)[0][0]])
                            data.append({'csd_code': code, 'mol': mol, 'smiles': smiles, 'atoms': atoms, 'y': y})

                            
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
    print('collected mol df rows:', len(final_df))
    final_df.to_pickle('./Data/tmc_frm_xyz2mol_tmqm_all_df.pkl')