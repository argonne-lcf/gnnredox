from fhb_xyz2mol_helpers import *
import io
from itertools import combinations
from ase.io import write
from ase import Atoms
from rdkit import Chem
import numpy as np
import time
import concurrent.futures
import multiprocessing as mp

def rdkit_mol_to_ase_atoms(rdkit_mol: Chem.Mol) -> Atoms:
    """
    Converts an RDKit Mol object with an existing conformer to an ASE Atoms object.

    Args:
        rdkit_mol: RDKit Mol object. It is assumed that this molecule
                   has at least one conformer.

    Returns:
        An ASE Atoms object.
    """
    if not isinstance(rdkit_mol, Chem.Mol):
        raise TypeError("Input must be an RDKit Mol object.")
    
    if rdkit_mol.GetNumConformers() == 0:
        raise ValueError("RDKit Mol object must have at least one conformer.")

    # Get atomic symbols
    symbols = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]

    # Get positions from the first available conformer
    # RDKit conformer coordinates are typically in Angstroms,
    # which is the default unit for ASE.
    conformer = rdkit_mol.GetConformer(0)  # Gets the first conformer
    positions = conformer.GetPositions()   # This returns a NumPy array (N_atoms, 3)

    # Create ASE Atoms object
    ase_atoms = Atoms(symbols=symbols, positions=positions)
    
    return ase_atoms

# Target csd codes
def read_csd_codes(file_path):
    """Read CSD codes from a text file."""
    with open(file_path, 'r') as f:
        csd_codes = [line.strip() for line in f.readlines()]
    return csd_codes

# XYZ reader

def parse_tmqm_xyz_file(xyz_file, target_csd_codes, found_csd_codes):
    """Parse an XYZ file and extract ASE Atoms objects for target CSD codes."""
    ase_atoms_dict = {}
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        try:
            num_atoms = int(lines[i].strip())
            if i + num_atoms + 2 > len(lines):  # Check if we have enough lines left
                break
            
            csd_info_line = lines[i+1].strip()
            csd_code = None
            for item in csd_info_line.split('|'):
                if 'CSD_code =' in item:
                    csd_code = item.strip().split('CSD_code =')[1].strip()
                    break
            
            if csd_code and csd_code in target_csd_codes:
                symbols = []
                positions = []
                
                for j in range(i+2, i+num_atoms+2):
                    line = lines[j].strip()
                    symbol, x, y, z = line.split()[:4]
                    symbols.append(symbol)
                    positions.append([float(x), float(y), float(z)])
                
                atoms = Atoms(symbols=symbols, positions=positions)
                ase_atoms_dict[csd_code] = atoms
                found_csd_codes.add(csd_code)
            
            if found_csd_codes == target_csd_codes:
                break
            
            i += num_atoms + 2 + 1 # Move to the next molecule block
        except (ValueError, IndexError) as e:
            print(f"Error parsing line {i}: {e}")
            break
    return ase_atoms_dict, found_csd_codes


# charge file parser

def parse_tmqm_q_file(q_file, target_csd_codes):
    """Parse a .q file and extract charges for each molecule."""
    charge_dict = {}
    found_csd_codes = set()
    
    with open(q_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if "CSD_code =" in line:
            # Extract CSD code
            csd_code = line.split("CSD_code =", 1)[1].split('|', 1)[0].strip()
            charges = []
            
            # Read the next lines until we find "Total charge"
            j = i + 1
            while j < len(lines) and not lines[j].startswith("Total charge"):
                atom_line = lines[j].strip()
                if atom_line:  # Ensure it's not an empty line
                    try:
                        _, charge_str = atom_line.split(maxsplit=1)
                        charges.append(float(charge_str))
                    except ValueError as e:
                        print(f"Error parsing line {j}: {e}")
                j += 1
            
            # Store the charges in the dictionary
            if csd_code and charges and csd_code in target_csd_codes:
                charge_dict[csd_code] = np.array(charges)
                found_csd_codes.add(csd_code)

            if found_csd_codes == target_csd_codes:
                break
            
            # Move i to the end of this block
            i = j
        
        else:
            i += 1
    
    return charge_dict


# Bond order parser

def parse_tmqm_bond_order_file(file_path, target_csd_codes, found_csd_codes):
    BO_dict = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    current_csd_code = None
    atom_data = []
    
    for line in lines:
        if line.strip():
            if "CSD_code" in line:
                # Process the previous molecule's data before moving to the next one
                if current_csd_code is not None and len(atom_data) > 0:
                    n_atoms = max(int(data[0]) for data in atom_data)
                    # print(current_csd_code)
                    bond_order_matrix = np.zeros((n_atoms, n_atoms))
                    
                    for data in atom_data:
                        atom_idx = int(data[0]) - 1
                        total_BO = float(data[2])
                        
                        neighbors = []
                        i = 3
                        while i < len(data):
                            neigh_type = data[i]
                            neigh_idx = int(data[i + 1]) - 1
                            neigh_BO = float(data[i + 2])
                            neighbors.append((neigh_idx, neigh_BO))
                            bond_order_matrix[atom_idx, neigh_idx] = neigh_BO
                            bond_order_matrix[neigh_idx, atom_idx] = neigh_BO
                            i += 3
                        
                    if current_csd_code in target_csd_codes:
                        BO_dict[current_csd_code] = bond_order_matrix
                        found_csd_codes.add(current_csd_code)
                    
                    if found_csd_codes == target_csd_codes:
                        break
                
                # Reset for the new molecule
                current_csd_code = line.split("CSD_code =", 1)[1].split('|', 1)[0].strip()
                atom_data = []
            else:
                # Collect data lines for the current molecule
                atom_data.append(line.split())
    
    # Process the last molecule in the file
    if current_csd_code is not None and len(atom_data) > 0:
        n_atoms = max(int(data[0]) for data in atom_data)
        bond_order_matrix = np.zeros((n_atoms, n_atoms))
        
        for data in atom_data:
            atom_idx = int(data[0]) - 1
            total_BO = float(data[2])
            
            neighbors = []
            i = 3
            while i < len(data):
                neigh_type = data[i]
                neigh_idx = int(data[i + 1]) - 1
                neigh_BO = float(data[i + 2])
                neighbors.append((neigh_idx, neigh_BO))
                bond_order_matrix[atom_idx, neigh_idx] = neigh_BO
                bond_order_matrix[neigh_idx, atom_idx] = neigh_BO
                i += 3
        
        if current_csd_code in target_csd_codes:
            BO_dict[current_csd_code] = bond_order_matrix
    
    return BO_dict, found_csd_codes



def parse_xtb_ptb_q_values(xtb_output_content: str) -> np.ndarray | None:
    """
    Parses the xtb --ptb output content to extract atomic partial charges (q).

    Args:
        xtb_output_content (str): The full string content of the xtb output file.

    Returns:
        np.ndarray: A NumPy array of partial charges, or None if the section is not found.
    """
    charges = []
    in_charge_section = False
    try:
        lines = xtb_output_content.splitlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if "* Atomic partial charges (q)" in line:
                in_charge_section = True
                in_charge_line = line_idx
                continue

            if in_charge_section:
                # skip the first three lines
                if line_idx <= in_charge_line + 3:
                    continue
                if line.startswith("----") or line.startswith("total:"): # End of charge section
                    in_charge_section = False
                    break
                if line.strip(): # Ensure not an empty line
                    parts = line.split()
                    if len(parts) >= 3 and parts[0].isdigit(): # Check if it's an atom line
                        charges.append(float(parts[2])) # Charge is the 3rd element
        
        if charges:
            return np.array(charges)
        else:
            print("Warning: Atomic partial charges section not found or empty in xtb output.")
            return None
    except Exception as e:
        print(f"Error parsing partial charges from xtb output: {e}")
        return None
    

def parse_xtb_ptb_bond_orders(xtb_output_content: str, num_atoms: int = None) -> np.ndarray | None:
    """
    Parses the xtb --ptb output content to extract Wiberg bond orders (WBO)
    and construct a bond order matrix.

    Args:
        xtb_output_content (str): The full string content of the xtb output file.
        num_atoms (int, optional): The total number of atoms in the molecule.
                                   If None, it will try to infer from the WBO section.
                                   Providing it is more robust.

    Returns:
        np.ndarray: A square NumPy array representing the bond order matrix (symmetric),
                    or None if the section is not found or parsing fails.
    """
    bond_orders_data = []
    in_wbo_section = False
    max_atom_idx_found = 0

    try:
        lines = xtb_output_content.splitlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if "Wiberg/Mayer (AO) data" in line:
                in_wbo_section = True
                in_wbo_line = line_idx
                max_atom_idx_found = -1
                continue

            if in_wbo_section:
                # skip the first five lines
                if line_idx <= in_wbo_line + 5:
                    continue
                if line.startswith(" -----") or line.startswith("Topologies differ"): # End of WBO section
                    in_wbo_section = False
                    break
                if line.strip():
                    parts = line.split()
                    # A valid data line starts with an atom index (integer)
                    if '--' in parts:
                        # go to next line and check if it is a continued line
                        for next_line in lines[line_idx + 1:]:
                            next_line = next_line.strip()
                            if next_line.startswith("----") or next_line.startswith("total:"):
                                break
                            elif '--' in next_line.split():
                                break
                            else:
                                # This is a continued line, append to the current parts
                                parts.extend(next_line.split())
                        atom_index = int(parts[0])
                        max_atom_idx_found = max(max_atom_idx_found, atom_index)
                        bonded_atoms_info = []
                        i = 5 # Start index for neighbor data
                        while i < len(parts):
                            # Check if parts[i] is a digit before converting
                            if parts[i].isdigit():
                                bonded_atom_idx_str = parts[i]
                                wbo_str = parts[i+2]
                                bonded_atoms_info.append({
                                    'neighbor_idx': int(bonded_atom_idx_str),
                                    'wbo': float(wbo_str)
                                })
                            
                            i += 3 # Move to the next potential neighbor triplet
                        bond_orders_data.append({'atom_idx': atom_index, 'bonds': bonded_atoms_info})
                    else:
                        # This line does not start with an atom index, skip it
                        continue
        
        if not bond_orders_data:
            print("Warning: Wiberg bond order section not found or empty in xtb output.")
            return None

        # Determine number of atoms if not provided
        if num_atoms is None:
            if max_atom_idx_found > 0:
                num_atoms = max_atom_idx_found
            else:
                print("Error: Could not determine number of atoms for bond order matrix.")
                return None
        
        bond_order_matrix = np.zeros((num_atoms, num_atoms), dtype=float)
        for atom_entry in bond_orders_data:
            idx1 = atom_entry['atom_idx'] - 1 # Convert to 0-based index
            for bond_info in atom_entry['bonds']:
                idx2 = bond_info['neighbor_idx'] - 1 # Convert to 0-based index
                wbo = bond_info['wbo']
                # Ensure indices are within bounds
                if 0 <= idx1 < num_atoms and 0 <= idx2 < num_atoms:
                    bond_order_matrix[idx1, idx2] = wbo
                    bond_order_matrix[idx2, idx1] = wbo # Matrix is symmetric
                else:
                    print(f"Warning: Atom index out of bounds ({idx1+1} or {idx2+1}) for num_atoms={num_atoms}. Skipping bond.")


        return bond_order_matrix

    except Exception as e:
        print(f"Error parsing Wiberg bond orders from xtb output: {e}")
        import traceback
        traceback.print_exc()
        return None


def single_spaced_string(s: str) -> str:
    """
    Converts a string to single spaced format by removing extra spaces.
    """
    return ' '.join(s.split())

def parse_xtb_q_values(xtb_output_content: str) -> np.ndarray | None:
    """
    Parses the xtb --ptb output content to extract atomic partial charges (q).

    Args:
        xtb_output_content (str): The full string content of the xtb output file.

    Returns:
        np.ndarray: A NumPy array of partial charges, or None if the section is not found.
    """
    charges = []
    in_charge_section = False
    try:
        lines = xtb_output_content.splitlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if "# Z covCN q C6AA" in single_spaced_string(line):
                in_charge_section = True
                in_charge_line = line_idx
                continue

            if in_charge_section:
                # skip the first zero lines
                if line_idx <= in_charge_line + 0:
                    continue
                if not line: #line.startswith("----") or line.startswith("total:"): # End of charge section
                    in_charge_section = False
                    break
                if line.strip(): # Ensure not an empty line
                    parts = line.split()
                    if len(parts) >= 3 and parts[0].isdigit(): # Check if it's an atom line
                        charges.append(float(parts[4])) # Charge is the 5th element
        
        if charges:
            return np.array(charges)
        else:
            print("Warning: Atomic partial charges section not found or empty in xtb output.")
            return None
    except Exception as e:
        print(f"Error parsing partial charges from xtb output: {e}")
        return None
    

def parse_xtb_bond_orders(xtb_output_content: str, num_atoms: int = None) -> np.ndarray | None:
    """
    Parses the xtb --ptb output content to extract Wiberg bond orders (WBO)
    and construct a bond order matrix.

    Args:
        xtb_output_content (str): The full string content of the xtb output file.
        num_atoms (int, optional): The total number of atoms in the molecule.
                                   If None, it will try to infer from the WBO section.
                                   Providing it is more robust.

    Returns:
        np.ndarray: A square NumPy array representing the bond order matrix (symmetric),
                    or None if the section is not found or parsing fails.
    """
    bond_orders_data = []
    in_wbo_section = False
    max_atom_idx_found = 0

    try:
        lines = xtb_output_content.splitlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if "Wiberg/Mayer (AO) data" in line:
                in_wbo_section = True
                in_wbo_line = line_idx
                max_atom_idx_found = -1
                continue

            if in_wbo_section:
                # skip the first five lines
                if line_idx <= in_wbo_line + 5:
                    continue
                if line.startswith(" -----") or line.startswith("Topologies differ"): # End of WBO section
                    in_wbo_section = False
                    break
                if line.strip():
                    parts = line.split()
                    if '--' in parts:
                        # go to next line and check if it is a continued line
                        for next_line in lines[line_idx + 1:]:
                            next_line = next_line.strip()
                            if next_line.startswith("----") or next_line.startswith("total:"):
                                break
                            elif '--' in next_line.split():
                                break
                            else:
                                # This is a continued line, append to the current parts
                                parts.extend(next_line.split())
                        atom_index = int(parts[0])
                        max_atom_idx_found = max(max_atom_idx_found, atom_index)    
                        bonded_atoms_info = []
                        i = 5 # Start index for neighbor data
                        while i < len(parts):
                            # Check if parts[i] is a digit before converting
                            if parts[i].isdigit():
                                bonded_atom_idx_str = parts[i]
                                wbo_str = parts[i+2]
                                bonded_atoms_info.append({
                                    'neighbor_idx': int(bonded_atom_idx_str),
                                    'wbo': float(wbo_str)
                                })
                            
                            i += 3 # Move to the next potential neighbor triplet
                        bond_orders_data.append({'atom_idx': atom_index, 'bonds': bonded_atoms_info})
                    else: # continued line
                        continue
                        
        
        if not bond_orders_data:
            print("Warning: Wiberg bond order section not found or empty in xtb output.")
            return None

        # Determine number of atoms if not provided
        if num_atoms is None:
            if max_atom_idx_found > 0:
                num_atoms = max_atom_idx_found
            else:
                print("Error: Could not determine number of atoms for bond order matrix.")
                return None
        
        bond_order_matrix = np.zeros((num_atoms, num_atoms), dtype=float)
        for atom_entry in bond_orders_data:
            idx1 = atom_entry['atom_idx'] - 1 # Convert to 0-based index
            for bond_info in atom_entry['bonds']:
                idx2 = bond_info['neighbor_idx'] - 1 # Convert to 0-based index
                wbo = bond_info['wbo']
                # Ensure indices are within bounds
                if 0 <= idx1 < num_atoms and 0 <= idx2 < num_atoms:
                    bond_order_matrix[idx1, idx2] = wbo
                    bond_order_matrix[idx2, idx1] = wbo # Matrix is symmetric
                else:
                    print(f"Warning: Atom index out of bounds ({idx1+1} or {idx2+1}) for num_atoms={num_atoms}. Skipping bond.")


        return bond_order_matrix

    except Exception as e:
        print(f"Error parsing Wiberg bond orders from xtb output: {e}")
        import traceback
        traceback.print_exc()
        return None


def ase_atoms_to_xyz_block(atoms):
    """
    Convert an ASE Atoms object into an XYZ-format block using ASE's write function.

    """

    # Create an in-memory text buffer.
    buffer = io.StringIO()
    
    # Write the atoms in XYZ format to the buffer.
    write(buffer, atoms, format='xyz')
    
    # Get the string from the buffer.
    xyz_block = buffer.getvalue()
    
    return xyz_block



def get_mol_with_bonds(mol, charge):

    from rdkit.Chem import rdDetermineBonds

    #print(charge)
    try:
        rdDetermineBonds.DetermineBonds(mol, useHueckel=True, charge=charge)
        #print('success')
        return mol
    except:
        print(f'No mol with bond found for mol for charge = {charge}')
        return None

def get_mol_with_bonds_async(mol, charge, timeout=60):

    from rdkit.Chem import rdDetermineBonds

    with mp.Pool(processes=1) as pool:  # Create a separate process for this call
        result = pool.apply_async(rdDetermineBonds.DetermineBonds, (mol,), kwds={'useHueckel': True, 'charge': charge})  # Run asynchronously
        
        try:
            return result.get(timeout=timeout)  # Get result within timeout
        except mp.TimeoutError:
            print("Timeout: Skipping fragment")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        

def get_tot_charge(mols_list):
    tot_charge = 0
    for mol in mols_list:
        for atom in mol.GetAtoms():
            tot_charge += atom.GetFormalCharge()
            #frag_charge += atom.GetFormalCharge()
    return tot_charge



def get_lig_mol(mol, charge, coordinating_atoms):
    """A sanitizable mol object is created for the ligand, taking into account
    the checks defined in lig_checks.

    We try different charge settings and settings where carbenes are
    allowed/not allowed in case no perfect solution (no partial charges
    on other than the coordinating atoms) can be found. Finally best
    found solution based on criteria in lig_checks is returned.
    """

    import logging
    logger = logging.getLogger(__name__)
    
    orig_charge = charge

    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    AC = Chem.rdmolops.GetAdjacencyMatrix(mol)
    #print(AC)
    
    lig_mol = AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False) #get_mol_with_bonds(mol, charge) #AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False) #get_mol_with_bonds(mol, charge)
    
    if not lig_mol and charge >= 0:
        print('----- Initial estimated charge did not work. modifying charge by: -2  -----')
        charge += -2
        lig_mol = AC2mol(
            mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False
        )
        if not lig_mol:
            return None, charge
    if not lig_mol and charge < 0:
        print('----- Initial estimated charge did not work. modifying charge by: +2  -----')
        charge += 2
        lig_mol = AC2mol(
            mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False
        )
        if not lig_mol:
            print('----- Still no solutions found. Again modifying charge by: -4  -----')
            charge += -4
            lig_mol = AC2mol(
                mol,
                AC,
                atoms,
                charge,
                allow_charged_fragments=True,
                use_atom_maps=False,
            )
            if not lig_mol:
                return None, charge
            
    '''if not lig_mol:
        while True:
            if charge <= 0:
                break
            charge += -1
            lig_mol = AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False) #get_mol_with_bonds(mol, charge) #AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False) #get_mol_with_bonds(mol, charge)
            if lig_mol:
                break
        
        while True:
            if charge >= 0:
                break
            charge += 1
            lig_mol = AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False) #get_mol_with_bonds(mol, charge) #AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False) #get_mol_with_bonds(mol, charge)
            if lig_mol:
                break

        if not lig_mol:
            if orig_charge < 0:
                for charge in [orig_charge-1,orig_charge-2,orig_charge-3,orig_charge-4]:
                    print(charge)
                    lig_mol = AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False) #get_mol_with_bonds(mol, charge) #AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False) #get_mol_with_bonds(mol, charge)
                    if lig_mol:
                        break
            if not lig_mol:
                print('Could not generate ligand mol')
                return None, charge'''


    possible_res_mols = lig_checks(lig_mol, coordinating_atoms)
    best_res_mol, lowest_pos, lowest_neg, highest_aromatic = possible_res_mols[0]
    for res_mol, N_pos_atoms, N_neg_atoms, N_aromatic in possible_res_mols:
        if N_aromatic > highest_aromatic:
            best_res_mol, lowest_pos, lowest_neg, highest_aromatic = (
                res_mol,
                N_pos_atoms,
                N_neg_atoms,
                N_aromatic,
            )
        if (
            N_aromatic == highest_aromatic
            and N_pos_atoms + N_neg_atoms < lowest_pos + lowest_neg
        ):
            best_res_mol, lowest_pos, lowest_neg = res_mol, N_pos_atoms, N_neg_atoms
    if lowest_pos + lowest_neg == 0:
        return best_res_mol, charge

    lig_mol_no_carbene = AC2mol(
        mol,
        AC,
        atoms,
        charge,
        allow_charged_fragments=True,
        use_atom_maps=False,
        allow_carbenes=False,
    )
    allow_carbenes = True

    if lig_mol_no_carbene:
        res_mols_no_carbenes = lig_checks(lig_mol_no_carbene, coordinating_atoms)
        for res_mol, N_pos_atoms, N_neg_atoms, N_aromatic in res_mols_no_carbenes:
            if (
                N_aromatic > highest_aromatic
                and N_pos_atoms + N_neg_atoms <= lowest_pos + lowest_neg
            ):
                best_res_mol, lowest_pos, lowest_neg, highest_aromatic = (
                    res_mol,
                    N_pos_atoms,
                    N_neg_atoms,
                    N_aromatic,
                )
            if (
                N_aromatic == highest_aromatic
                and N_pos_atoms + N_neg_atoms < lowest_pos + lowest_neg
            ):
                best_res_mol, lowest_pos, lowest_neg = res_mol, N_pos_atoms, N_neg_atoms
                allow_carbenes = False

    if lowest_pos + lowest_neg == 0:
        logger.debug("found opt solution without carbenes")
        return best_res_mol, charge

    if lowest_pos - lowest_neg + charge < 0:
        new_charge = charge + 2
    else:
        new_charge = charge - 2  # if 0 maybe I should try both

    new_lig_mol = AC2mol(
        mol,
        AC,
        atoms,
        new_charge,
        allow_charged_fragments=True,
        use_atom_maps=False,
        allow_carbenes=allow_carbenes,
    )
    if not new_lig_mol:
        return best_res_mol, charge
    new_possible_res_mols = lig_checks(new_lig_mol, coordinating_atoms)
    for res_mol, N_pos_atoms, N_neg_atoms, N_aromatic in new_possible_res_mols:
        if N_aromatic > highest_aromatic:
            best_res_mol, lowest_pos, lowest_neg, highest_aromatic = (
                res_mol,
                N_pos_atoms,
                N_neg_atoms,
                N_aromatic,
            )
            charge = new_charge
        if (
            N_aromatic == highest_aromatic
            and N_pos_atoms + N_neg_atoms < lowest_pos + lowest_neg
        ):
            best_res_mol, lowest_pos, lowest_neg = res_mol, N_pos_atoms, N_neg_atoms
            charge = new_charge

    return best_res_mol, charge


def get_fragments(row, bo_cutoff = 0.4, tm_lig_bo_cutoff = 0.25):
    
    import networkx as nx
    from ase import Atoms

    halogens = {'F', 'Cl', 'Br', 'I'}
    
    #csd_code = row['csd_code']
    tm_atoms = row['atoms_obj']   # ASE Atoms object of the molecule.
    bo_arr = row['bo_arr']     # Bond order array for the molecule.
    tot_charge = round(float(row['q'].sum()))

    n_atoms = len(tm_atoms)
    
    #Identify metal atoms (Fe in this case) by their symbol.
    fe_idx = [i for i, atom in enumerate(tm_atoms) if atom.symbol == 'Fe']
    if len(fe_idx) > 1:
        print('Error: more than one TM')
        return None
    else:
        fe_idx = fe_idx[0]
    
    #Determine non-metal indices (all indices not in fe_indices).
    nonmetal_indices = [i for i in range(n_atoms) if i != fe_idx]

    # Build graph using only nonmetal atoms.
    G = nx.Graph()
    G.add_nodes_from(nonmetal_indices)
    
    # Consider a bond to be present if the bond order is > cutoff
    for i in nonmetal_indices:
        for j in nonmetal_indices:
            if i < j and bo_arr[i, j] > 0.4:
                G.add_edge(i, j)
    
    #Create new ASE Atoms object for Fe atom/s.
    Fe_positions = tm_atoms.get_positions()[np.array([fe_idx])]
    Fe_numbers = tm_atoms.get_atomic_numbers()[np.array([fe_idx])]
    Fe_atoms = Atoms(numbers=Fe_numbers, positions=Fe_positions,
                        cell=tm_atoms.get_cell(), pbc=tm_atoms.get_pbc())
    Fe_mol = Chem.MolFromXYZBlock(ase_atoms_to_xyz_block(Fe_atoms))  
    rwMol = Chem.RWMol(Fe_mol)
    atom = rwMol.GetAtomWithIdx(0)
    atom.SetIntProp("origIdx", fe_idx)
    atom.SetIntProp("fe_connected", 0)
    Fe_mol = rwMol.GetMol()


    #Find connected components; these are the fragments.
    fragments = list(nx.connected_components(G))
    lig_mols = []
    for frag in fragments:
        # Order indices for clarity.
        frag_indices = sorted(frag)

        frag_positions = tm_atoms.get_positions()[np.array(frag_indices)]
        frag_numbers = tm_atoms.get_atomic_numbers()[np.array(frag_indices)]
        frag_atoms = Atoms(numbers=frag_numbers, positions=frag_positions, cell=tm_atoms.get_cell(), pbc=tm_atoms.get_pbc())
        lig_mol = Chem.MolFromXYZBlock(ase_atoms_to_xyz_block(frag_atoms))
        
        rwMol = Chem.RWMol(lig_mol)

        # check for halogens
        if len(rwMol.GetAtoms()) == 1:
            #print('checking for halogen')
            atom = rwMol.GetAtomWithIdx(0)
            if atom.GetSymbol() in halogens:
                atom.SetFormalCharge(-1)
                
        # Disable automatic H addition
        for atom in rwMol.GetAtoms():
            atom.SetIntProp("origIdx", frag_indices[atom.GetIdx()])
            atom.SetIntProp("fe_connected", 0)
            atom.SetNoImplicit(True)
        # Consider a bond to be present if the bond order is > bo_cutoff
        for atom1 in rwMol.GetAtoms():
            i = frag_indices[atom1.GetIdx()]
            for atom2 in rwMol.GetAtoms():
                j = frag_indices[atom2.GetIdx()]
                if i < j and bo_arr[i, j] > bo_cutoff:
                    rwMol.AddBond(atom1.GetIdx(), atom2.GetIdx(), Chem.BondType.SINGLE)
        
        # Fe connections
        for atom in rwMol.GetAtoms():
            atom_idx = frag_indices[atom.GetIdx()]
            if bo_arr[atom_idx, fe_idx] > tm_lig_bo_cutoff:
                atom.SetIntProp("fe_connected", 1)
        
        lig_mol = rwMol.GetMol()

        lig_mols.append(lig_mol)
    
    return Fe_mol, lig_mols, tot_charge


def get_tmc_mol(df_row, with_stereo=False):
    """Get TMC mol object from given xyz file.

    Args:
        xyz_file (str) : Path to TMC xyz file
        overall_charge (int): Overall charge of TMC
        with_stereo (bool): Whether to percieve stereochemistry from the 3D data

    Returns:
        tmc_mol (rdkit.Chem.rdchem.Mol): TMC mol object
    """
    
    from functools import reduce
    import logging
    
    logger = logging.getLogger(__name__)

    #halogens = {'F', 'Cl', 'Br', 'I'}
    
    Fe_mol, frag_mols, tot_charge = get_fragments(df_row)

    # Run calcualtions in parallel for all fragments
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        prop_ligand_charges = list(executor.map(get_proposed_ligand_charge, frag_mols))
    print('calculated all charges:',prop_ligand_charges)

    total_lig_charge = 0
    tm_idx = None
    lig_list = []
    for i, frag in enumerate(frag_mols):
        #time.sleep(0.5) # Seconds
        lig_charge = prop_ligand_charges[i] #get_proposed_ligand_charge(frag)
        #print('debug: lig charge final', lig_charge)
        m = Chem.Mol(frag)

        lig_coordinating_atoms = [
            atom.GetIdx()
            for atom in m.GetAtoms()
            if int(atom.GetProp('fe_connected')) == 1
        ]

        lig_mol, lig_charge = get_lig_mol(m, lig_charge, lig_coordinating_atoms)

        
        if not lig_mol:
            return None

        total_lig_charge += lig_charge
        lig_list.append(lig_mol)

    lig_list.append(Fe_mol)

    combined = reduce(Chem.CombineMols, lig_list)
    tm_ox = tot_charge - total_lig_charge
    print(f'Charge info: Total charge = {tot_charge}, Total ligand charge = {total_lig_charge}, TM charge = {tm_ox}')

    for a in combined.GetAtoms():
        if a.GetAtomicNum() in TRANSITION_METALS_NUM:
            a.SetFormalCharge(tm_ox)


    emol = Chem.RWMol(combined)
    coordinating_atoms_idx = [atom.GetIdx() 
                              for atom in emol.GetAtoms() 
                              if int(atom.GetProp('fe_connected')) == 1]
    tm_idx = [
        a.GetIdx() for a in emol.GetAtoms() if a.GetSymbol() in TRANSITION_METALS
    ][0]

    #print(coordinating_atoms_idx, tm_idx)

    dMat = Chem.Get3DDistanceMatrix(emol)
    cut_atoms = []
    for i, j in combinations(coordinating_atoms_idx, 2):
        bond = emol.GetBondBetweenAtoms(int(i), int(j))
        if bond and abs(dMat[i, tm_idx] - dMat[j, tm_idx]) >= 0.4:
            logger.debug(
                "Haptic bond pattern with too great distance:",
                dMat[i, tm_idx],
                dMat[j, tm_idx],
            )
            if dMat[i, tm_idx] > dMat[j, tm_idx] and i in coordinating_atoms_idx:
                coordinating_atoms_idx.remove(i)
                cut_atoms.append(i)
            if dMat[j, tm_idx] > dMat[i, tm_idx] and j in coordinating_atoms_idx:
                coordinating_atoms_idx.remove(j)
                cut_atoms.append(j)
    for j in cut_atoms:
        for i in coordinating_atoms_idx:
            bond = emol.GetBondBetweenAtoms(int(i), int(j))
            if (
                bond
                and dMat[i, tm_idx] - dMat[j, tm_idx] >= -0.1
                and i in coordinating_atoms_idx
            ):
                coordinating_atoms_idx.remove(i)

    for i in coordinating_atoms_idx:
        if emol.GetBondBetweenAtoms(i, tm_idx):
            continue
        emol.AddBond(i, tm_idx, Chem.BondType.DATIVE)

    smiles = Chem.MolToSmiles(emol.GetMol())
    # Fix specific cases
    smiles = fix_equivalent_Os_smiles(smiles)
    smiles = fix_NO2_smiles(smiles)
    smiles = fix_nitroso_smiles(smiles)

    tmc_mol = emol.GetMol()
    tmc_mol = fix_equivalent_Os(tmc_mol)
    tmc_mol = fix_NO2(tmc_mol)
    tmc_mol = fix_nitroso(tmc_mol)

    #tmc_mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(tmc_mol)
    if with_stereo:
        chiral_stereo_check(tmc_mol)


    return tmc_mol, smiles


def process_row(row):
    """Function to process a single row"""
    try:
        index = row.name
        code = row['csd_code']
        mol, smiles = get_tmc_mol(row)
        return (code, mol, smiles)
    except Exception as e:
        print(f"Error processing {row['csd_code']}: {e}")
        return (code, None, None)
    

def process_row_conc(row):
    """Function to process a single row"""
    try:
        index = row.name
        code = row['csd_code']
        print(index,code)
        mol, smiles = get_tmc_mol(row)
        return (code, mol, smiles)
    except Exception as e:
        print(f"Error processing {row['csd_code']}: {e}")
        return (code, None, None)
