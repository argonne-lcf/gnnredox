from fhb_helpers import *

import copy
import itertools
import sys
import numpy as np

from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdchem, rdEHTTools, rdmolops, rdDetermineBonds, AllChem, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize

from collections import defaultdict
import networkx as nx


# XYZ2mol variables

TRANSITION_METALS = ["Sc","Ti","V","Cr","Mn","Fe","Co","La","Ni","Cu","Zn",
                     "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","Lu",
                     "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
]

TRANSITION_METALS_NUM = [21,22,23,24,25,26,27,57,28,29,30,39,40,41,
                         42,43,44,45,46,47,48,71,72,73,74,75,76,77,78,79,80,
]


ALLOWED_OXIDATION_STATES = {
    "Sc": [3],
    "Ti": [3, 4],
    "V": [2, 3, 4, 5],
    "Cr": [2, 3, 4, 6],
    "Mn": [2, 3, 4, 6, 7],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2],
    "Cu": [1, 2],
    "Zn": [2],
    "Y": [3],
    "Zr": [4],
    "Nb": [3, 4, 5],
    "Mo": [2, 3, 4, 5, 6],
    "Tc": [2, 3, 4, 5, 6, 7],
    "Ru": [2, 3, 4, 5, 6, 7, 8],
    "Rh": [1, 3],
    "Pd": [2, 4],
    "Ag": [1],
    "Cd": [2],
    "La": [3],
    "Hf": [4],
    "Ta": [3, 4, 5],
    "W": [2, 3, 4, 5, 6],
    "Re": [2, 3, 4, 5, 6, 7],
    "Os": [3, 4, 5, 6, 7, 8],
    "Ir": [1, 3],
    "Pt": [2, 4],
    "Au": [1, 3],
    "Hg": [1, 2],
}


global atomic_valence_electrons

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[13] = 3
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[18] = 8
atomic_valence_electrons[32] = 4
atomic_valence_electrons[33] = 5  # As
atomic_valence_electrons[35] = 7
atomic_valence_electrons[34] = 6
atomic_valence_electrons[53] = 7

# TMs
atomic_valence_electrons[21] = 3  # Sc
atomic_valence_electrons[22] = 4  # Ti
atomic_valence_electrons[23] = 5  # V
atomic_valence_electrons[24] = 6  # Cr
atomic_valence_electrons[25] = 7  # Mn
atomic_valence_electrons[26] = 8  # Fe
atomic_valence_electrons[27] = 9  # Co
atomic_valence_electrons[28] = 10  # Ni
atomic_valence_electrons[29] = 11  # Cu
atomic_valence_electrons[30] = 12  # Zn

atomic_valence_electrons[39] = 3  # Y
atomic_valence_electrons[40] = 4  # Zr
atomic_valence_electrons[41] = 5  # Nb
atomic_valence_electrons[42] = 6  # Mo
atomic_valence_electrons[43] = 7  # Tc
atomic_valence_electrons[44] = 8  # Ru
atomic_valence_electrons[45] = 9  # Rh
atomic_valence_electrons[46] = 10  # Pd
atomic_valence_electrons[47] = 11  # Ag
atomic_valence_electrons[48] = 12  # Cd

atomic_valence_electrons[57] = 3  # La
atomic_valence_electrons[72] = 4  # Hf
atomic_valence_electrons[73] = 5  # Ta
atomic_valence_electrons[74] = 6  # W
atomic_valence_electrons[75] = 7  # Re
atomic_valence_electrons[76] = 8  # Os
atomic_valence_electrons[77] = 9  # Ir
atomic_valence_electrons[78] = 10  # Pt
atomic_valence_electrons[79] = 11  # Au
atomic_valence_electrons[80] = 12  # Hg



global atomic_valence

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3, 4]
atomic_valence[6] = [4, 2]
atomic_valence[7] = [3, 4]
atomic_valence[8] = [2, 1, 3]  # [2,1,3]
atomic_valence[9] = [1]
atomic_valence[13] = [3, 4]
atomic_valence[14] = [4]
atomic_valence[15] = [3, 5]  # [5,4,3]
atomic_valence[16] = [2, 4, 6]  # [6,3,2]
atomic_valence[17] = [1]
atomic_valence[18] = [0]
atomic_valence[32] = [4]
atomic_valence[33] = [5, 3]
atomic_valence[35] = [1]
atomic_valence[34] = [2]
atomic_valence[52] = [2]
atomic_valence[53] = [1]

atomic_valence[21] = [20]
atomic_valence[22] = [20]
atomic_valence[23] = [20]
atomic_valence[24] = [20]
atomic_valence[25] = [20]
atomic_valence[26] = [20]
atomic_valence[27] = [20]
atomic_valence[28] = [20]
atomic_valence[29] = [20]
atomic_valence[30] = [20]

atomic_valence[39] = [20]
atomic_valence[40] = [20]
atomic_valence[41] = [20]
atomic_valence[42] = [20]
atomic_valence[43] = [20]
atomic_valence[44] = [20]
atomic_valence[45] = [20]
atomic_valence[46] = [20]
atomic_valence[47] = [20]
atomic_valence[48] = [20]


atomic_valence[57] = [20]
atomic_valence[72] = [20]
atomic_valence[73] = [20]
atomic_valence[74] = [20]
atomic_valence[75] = [20]
atomic_valence[76] = [20]
atomic_valence[77] = [20]
atomic_valence[78] = [20]
atomic_valence[79] = [20]
atomic_valence[80] = [20]





#================ XYZ2mol Ligand checks ==================#


def lig_checks(lig_mol, coordinating_atoms):
    """Sending proposed ligand mol object through series of checks.

    - neighbouring coordinating atoms must be connected by pi-bond, aromatic bond (haptic), conjugated system
    - If I have two neighbouring identical charges -> fail, I would rather change the charge and make a bond
     -> suggest new charge adding/subtracting electrons based on these neighbouring charges
    - count partial charges: partial charges that are not negative on ligand coordinating atoms count against this ligand
      -> loop through resonance forms to see if any live up to this, then choose that one.
      -> partial positive charge on coordinating atom is big red flag
      -> If "bad" partial charges still exists suggest a new charge: add/subtract electrons based on the values of the partial charges
    """
    res_mols = rdchem.ResonanceMolSupplier(lig_mol)
    if len(res_mols) == 0:
        res_mols = rdchem.ResonanceMolSupplier(
            lig_mol, flags=Chem.ALLOW_INCOMPLETE_OCTETS
        )
    # Check for neighbouring coordinating atoms:
    possible_lig_mols = []

    for res_mol in res_mols:
        positive_atoms = []
        negative_atoms = []
        N_aromatic = 0
        for a in res_mol.GetAtoms():
            if a.GetIsAromatic():
                N_aromatic += 1
            if a.GetFormalCharge() > 0:
                positive_atoms.append(a.GetIdx())
            if a.GetFormalCharge() < 0 and a.GetIdx() not in coordinating_atoms:
                negative_atoms.append(a.GetIdx())

        possible_lig_mols.append(
            (res_mol, len(positive_atoms), len(negative_atoms), N_aromatic)
        )
    return possible_lig_mols



def fix_NO2(mol):
    """Localizes nitro groups that have been assigned a charge of -2 (neutral
    Nitrogen bound to two negatively charged Oxygen atoms).

    These groups are changed to reflect the correct neutral
    configuration of a nitro group. The oxidation state on the
    transition metal is changed accordingly.
    """
    #m = Chem.MolFromSmiles(smiles)
    emol = Chem.RWMol(mol)
    patt = Chem.MolFromSmarts(
        "[#8-]-[#7+0]-[#8-].[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]"
    )
    matches = emol.GetSubstructMatches(patt)
    for a1, a2, a3, a4 in matches:
        if not emol.GetBondBetweenAtoms(a1, a4) and not emol.GetBondBetweenAtoms(
            a3, a4
        ):
            tm = emol.GetAtomWithIdx(a4)
            o1 = emol.GetAtomWithIdx(a1)
            n = emol.GetAtomWithIdx(a2)
            tm_charge = tm.GetFormalCharge()
            new_charge = tm_charge - 2
            tm.SetFormalCharge(new_charge)
            n.SetFormalCharge(+1)
            o1.SetFormalCharge(0)
            emol.RemoveBond(a1, a2)
            emol.AddBond(a1, a2, rdchem.BondType.DOUBLE)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol #Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

def fix_NO2_smiles(smiles):
    """Localizes nitro groups that have been assigned a charge of -2 (neutral
    Nitrogen bound to two negatively charged Oxygen atoms).

    These groups are changed to reflect the correct neutral
    configuration of a nitro group. The oxidation state on the
    transition metal is changed accordingly.
    """
    m = Chem.MolFromSmiles(smiles)
    emol = Chem.RWMol(m)
    patt = Chem.MolFromSmarts(
        "[#8-]-[#7+0]-[#8-].[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]"
    )
    matches = emol.GetSubstructMatches(patt)
    for a1, a2, a3, a4 in matches:
        if not emol.GetBondBetweenAtoms(a1, a4) and not emol.GetBondBetweenAtoms(
            a3, a4
        ):
            tm = emol.GetAtomWithIdx(a4)
            o1 = emol.GetAtomWithIdx(a1)
            n = emol.GetAtomWithIdx(a2)
            tm_charge = tm.GetFormalCharge()
            new_charge = tm_charge - 2
            tm.SetFormalCharge(new_charge)
            n.SetFormalCharge(+1)
            o1.SetFormalCharge(0)
            emol.RemoveBond(a1, a2)
            emol.AddBond(a1, a2, rdchem.BondType.DOUBLE)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))



def fix_nitroso(mol):
    """
    Attempts to correct a specific representation of a nitroso-like group
    where Nitrogen is incorrectly assigned charge -2 and Oxygen -1 ([N-2]-[O-]),
    changing it to [N-1]=O (formal charge -1 on N, double bond to neutral O)
    and adjusting the formal charge of a nearby transition metal accordingly.

    Args:
        mol: An RDKit molecule object.

    Returns:
        An RDKit molecule object (potentially modified). Returns the original
        molecule if no corrections are made or if input is invalid.
    """

    emol = Chem.RWMol(mol)


    # SMARTS Pattern: Find [N-2]-[O-1] and the TM
    patt_smarts = "[#7-2]-[#8-1].[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]"
    patt = Chem.MolFromSmarts(patt_smarts)
    matches = emol.GetSubstructMatches(patt)

    modified = False

    for n_idx, o_idx, tm_idx in matches:
        # Optional heuristic check can remain here if needed

        nitroso_N = emol.GetAtomWithIdx(n_idx)
        nitroso_O = emol.GetAtomWithIdx(o_idx)
        tm = emol.GetAtomWithIdx(tm_idx)

        # Original total charge of N(-2)-O(-1) fragment = -3
        # Target total charge of N(-1)=O(0) fragment = -1
        # Change in fragment charge = (-1) - (-3) = +2
        # To compensate, metal charge must change by -2
        original_tm_charge = tm.GetFormalCharge()
        new_tm_charge = original_tm_charge - 2

        tm.SetFormalCharge(new_tm_charge)
        nitroso_N.SetFormalCharge(-1) # <<< Target N is -1
        nitroso_O.SetFormalCharge(0)  # Target O is neutral
        emol.RemoveBond(n_idx, o_idx)
        emol.AddBond(n_idx, o_idx, rdchem.BondType.DOUBLE)
        modified = True
        print(f"Corrected [N-2]-[O-] near TM {tm_idx} to [N-1]=O. Adjusted TM charge from {original_tm_charge} to {new_tm_charge}.")


    # SMARTS Pattern: Find [N-1]-[O-1] and the TM
    patt_smarts = "[#7-1]-[#8-1].[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]"
    patt = Chem.MolFromSmarts(patt_smarts)
    matches = emol.GetSubstructMatches(patt)

    for n_idx, o_idx, tm_idx in matches:
        # Optional heuristic check can remain here if needed

        nitroso_N = emol.GetAtomWithIdx(n_idx)
        nitroso_O = emol.GetAtomWithIdx(o_idx)
        tm = emol.GetAtomWithIdx(tm_idx)

        # Original total charge of N(-1)-O(-1) fragment = -2
        # Target total charge of N(0)=O(0) fragment = 0
        # Change in fragment charge = (0) - (-2) = +2
        # To compensate, metal charge must change by -2
        original_tm_charge = tm.GetFormalCharge()
        new_tm_charge = original_tm_charge - 2

        tm.SetFormalCharge(new_tm_charge)
        nitroso_N.SetFormalCharge(0) # <<< Target N is -1
        nitroso_O.SetFormalCharge(0)  # Target O is neutral
        emol.RemoveBond(n_idx, o_idx)
        emol.AddBond(n_idx, o_idx, rdchem.BondType.DOUBLE)
        modified = True
        print(f"Corrected [N-1]-[O-] near TM {tm_idx} to N=O. Adjusted TM charge from {original_tm_charge} to {new_tm_charge}.")

    if not modified:
        return mol

    corrected_mol = emol.GetMol()
    Chem.SanitizeMol(corrected_mol)

    return corrected_mol

def fix_nitroso_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    corrected_mol = fix_nitroso(m)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(corrected_mol)))



def fix_equivalent_Os(mol):
    """Localizes and fixes where a neutral atom is coordinating to the metal
    but connected ro a negatively charged atom through resonane.

    The charge is moved to the coordinating atom and charges fixed
    accordingly.
    """
    #m = Chem.MolFromSmiles(smiles)
    emol = Chem.RWMol(mol)

    patt = Chem.MolFromSmarts("[#6-,#7-,#8-,#15-,#16-]-[*]=[#6,#7,#8,#15,#16]")

    matches = emol.GetSubstructMatches(patt)
    used_atom_ids_1 = []
    used_atom_ids_3 = []
    for atom in emol.GetAtoms():
        if atom.GetAtomicNum() in TRANSITION_METALS_NUM:
            neighbor_idxs = [a.GetIdx() for a in atom.GetNeighbors()]
            for a1, a2, a3 in matches:
                if (
                    a3 in neighbor_idxs
                    and a1 not in neighbor_idxs
                    and a1 not in used_atom_ids_1
                    and a3 not in used_atom_ids_3
                ):
                    used_atom_ids_1.append(a1)
                    used_atom_ids_3.append(a3)

                    emol.RemoveBond(a1, a2)
                    emol.AddBond(a1, a2, Chem.rdchem.BondType.DOUBLE)
                    emol.RemoveBond(a2, a3)
                    emol.AddBond(a2, a3, Chem.rdchem.BondType.SINGLE)
                    emol.GetAtomWithIdx(a1).SetFormalCharge(0)
                    emol.GetAtomWithIdx(a3).SetFormalCharge(-1)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol #Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

def fix_equivalent_Os_smiles(smiles):
    """Localizes and fixes where a neutral atom is coordinating to the metal
    but connected ro a negatively charged atom through resonane.

    The charge is moved to the coordinating atom and charges fixed
    accordingly.
    """
    m = Chem.MolFromSmiles(smiles)
    emol = Chem.RWMol(m)

    patt = Chem.MolFromSmarts("[#6-,#7-,#8-,#15-,#16-]-[*]=[#6,#7,#8,#15,#16]")

    matches = emol.GetSubstructMatches(patt)
    used_atom_ids_1 = []
    used_atom_ids_3 = []
    for atom in emol.GetAtoms():
        if atom.GetAtomicNum() in TRANSITION_METALS_NUM:
            neighbor_idxs = [a.GetIdx() for a in atom.GetNeighbors()]
            for a1, a2, a3 in matches:
                if (
                    a3 in neighbor_idxs
                    and a1 not in neighbor_idxs
                    and a1 not in used_atom_ids_1
                    and a3 not in used_atom_ids_3
                ):
                    used_atom_ids_1.append(a1)
                    used_atom_ids_3.append(a3)

                    emol.RemoveBond(a1, a2)
                    emol.AddBond(a1, a2, Chem.rdchem.BondType.DOUBLE)
                    emol.RemoveBond(a2, a3)
                    emol.AddBond(a2, a3, Chem.rdchem.BondType.SINGLE)
                    emol.GetAtomWithIdx(a1).SetFormalCharge(0)
                    emol.GetAtomWithIdx(a3).SetFormalCharge(-1)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))


def chiral_stereo_check(mol):
    """Find and embed chiral information into the model based on the
    coordinates.

    args:
        mol - rdkit molecule, with embeded conformer
    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    return



#=========================== Fixing Bond Order ==================================#

def AC2BO(
    AC, atoms, charge, allow_charged_fragments=True, use_graph=True, allow_carbenes=True
):
    """Implemenation of algorithm shown in Figure 2.

    UA: unsaturated atoms

    DU: degree of unsaturation (u matrix in Figure)

    best_BO: Bcurr in Figure
    """

    global atomic_valence
    global atomic_valence_electrons

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    AC_valence = list(AC.sum(axis=1))

    for i, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        # valence can't be smaller than number of neighbourgs
        possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
        if atomicNum == 6 and valence == 1:
            possible_valence.remove(2)
        if atomicNum == 6 and not allow_carbenes and valence == 2:
            possible_valence.remove(2)
        if atomicNum == 6 and valence == 2:
            possible_valence.append(3)
        if atomicNum == 16 and valence == 1:
            possible_valence = [1, 2]

        if not possible_valence:
            print(
                "Valence of atom",
                i,
                "is",
                valence,
                "which bigger than allowed max",
                max(atomic_valence[atomicNum]),
                ". Stopping",
            )
            sys.exit()
        valences_list_of_lists.append(possible_valence)

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = itertools.product(*valences_list_of_lists)

    best_BO = AC.copy()

    O_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 8
    ]
    N_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 7
    ]
    C_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 6
    ]
    P_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 15
    ]
    S_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 16
    ]

    O_sums = []
    for v_list in itertools.product(*O_valences):
        O_sums.append(v_list)
        # if sum(v_list) not in O_sums:
        #    O_sums.append(v_list))

    N_sums = []
    for v_list in itertools.product(*N_valences):
        N_sums.append(v_list)
        # if sum(v_list) not in N_sums:
        #    N_sums.append(sum(v_list))

    C_sums = []
    for v_list in itertools.product(*C_valences):
        C_sums.append(v_list)
        # if sum(v_list) not in C_sums:
        #    C_sums.append(sum(v_list))

    P_sums = []
    for v_list in itertools.product(*P_valences):
        P_sums.append(v_list)

    S_sums = []
    for v_list in itertools.product(*S_valences):
        S_sums.append(v_list)

    order_dict = dict()
    for i, v_list in enumerate(
        itertools.product(*[O_sums, N_sums, C_sums, P_sums, S_sums])
    ):
        order_dict[v_list] = i

    valence_order_list = []
    for valence_list in valences_list:
        C_sum = []
        N_sum = []
        O_sum = []
        P_sum = []
        S_sum = []
        for v, atomicNum in zip(valence_list, atoms):
            if atomicNum == 6:
                C_sum.append(v)
            if atomicNum == 7:
                N_sum.append(v)
            if atomicNum == 8:
                O_sum.append(v)
            if atomicNum == 15:
                P_sum.append(v)
            if atomicNum == 16:
                S_sum.append(v)

        order_idx = order_dict[
            (tuple(O_sum), tuple(N_sum), tuple(C_sum), tuple(P_sum), tuple(S_sum))
        ]
        valence_order_list.append(order_idx)

    sorted_valences_list = [
        y
        for x, y in sorted(
            zip(valence_order_list, list(itertools.product(*valences_list_of_lists)))
        )
    ]

    for valences in sorted_valences_list:  # valences_list:
        UA, DU_from_AC = get_UA(valences, AC_valence)
        check_len = len(UA) == 0
        if check_len:
            check_bo = BO_is_OK(
                AC,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, DU_from_AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )
            charge_OK = charge_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )

            if status:
                return BO, atomic_valence_electrons
            elif (
                BO.sum() >= best_BO.sum()
                and valences_not_too_large(BO, valences)
                and charge_OK
            ):
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons


def get_UA(maxValence_list, valence_list):
    """"""
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """"""
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, DU, use_graph=use_graph)[0]
    return BO


def valences_not_too_large(BO, valences):
    """"""
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def charge_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
    allow_carbenes=True,
):
    # total charge
    Q = 0
    # charge fragment list
    q_list = []

    if allow_charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if (
                    not allow_carbenes
                    and number_of_single_bonds_to_C == 2
                    and BO_valences[i] == 2
                ):
                    print("found illegal carbene")
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1
            if q != 0:
                q_list.append(q)
    return charge == Q


def BO_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
    allow_carbenes=True,
):
    """Sanity of bond-orders.

    args:
        BO -
        AC -
        charge -
        DU -


    optional
        allow_charges_fragments -


    returns:
        boolean - true of molecule is OK, false if not
    """

    if not valences_not_too_large(BO, valences):
        return False

    check_sum = (BO - AC).sum() == sum(DU)
    check_charge = charge_is_OK(
        BO,
        AC,
        charge,
        DU,
        atomic_valence_electrons,
        atoms,
        valences,
        allow_charged_fragments,
        allow_carbenes=True,
    )

    if check_charge and check_sum:
        return True

    return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    """"""
    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 6 and BO_valence == 2:
        charge = 0
    elif atom == 13:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    elif atom == 16 and BO_valence == 4:  # testing for sulphur
        charge = 0
    elif atom == 16 and BO_valence == 5:
        charge = 1

    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def BO2mol(
    mol,
    BO_matrix,
    atoms,
    atomic_valence_electrons,
    mol_charge,
    allow_charged_fragments=True,
    use_atom_maps=True,
):
    """Based on code written by Paolo Toscani.

    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.

    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule

    optional:
        allow_charged_fragments - bool - allow charged fragments

    returns
        mol - updated rdkit molecule with bond connectivity
    """

    length_bo = len(BO_matrix)
    length_atoms = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if length_bo != length_atoms:
        raise RuntimeError(
            "sizes of adjMat ({0:d}) and Atoms {1:d} differ".format(
                length_bo, length_atoms
            )
        )

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    for i in range(length_bo):
        for j in range(i + 1, length_bo):
            bo = int(round(BO_matrix[i, j]))
            if bo == 0:
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.RemoveBond(i, j)  # added this for TMC procedure
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            BO_matrix,
            mol_charge,
            use_atom_maps=use_atom_maps,
        )
    else:
        mol = set_atomic_radicals(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            use_atom_maps=use_atom_maps,
        )

    Chem.SanitizeMol(mol)

    return mol


def set_atomic_charges(
    mol,
    atoms,
    atomic_valence_electrons,
    BO_valences,
    BO_matrix,
    mol_charge,
    use_atom_maps=True,
):
    """"""
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if BO_valences[i] == 2:
                # q += 1
                a.SetNumRadicalElectrons(2)
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if abs(charge) > 0:
            a.SetFormalCharge(int(charge))

    # mol = clean_charges(mol)

    return mol


def set_atomic_radicals(
    mol, atoms, atomic_valence_electrons, BO_valences, use_atom_maps=True
):
    """The number of radical electrons = absolute atomic charge."""
    atomic_valence[8] = [2, 1]
    atomic_valence[7] = [3, 2]
    atomic_valence[6] = [4, 2]

    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        if abs(charge) > 0:
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    """"""
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1 :]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, DU, use_graph=True):
    """"""
    N_UA = 10000
    matching_ids = dict()
    matching_ids2 = dict()
    for i, du in zip(UA, DU):
        if du > 1:
            matching_ids[i] = N_UA
            matching_ids2[N_UA] = i
            N_UA += 1

    bonds = get_bonds(UA, AC)
    for i, j in bonds:
        if i in matching_ids:
            bonds.append(tuple(sorted([matching_ids[i], j])))

        elif j in matching_ids:
            bonds.append(tuple(sorted([i, matching_ids[j]])))

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        UA_pair = UA_pairs[0]

        remove_pairs = []
        add_pairs = []
        for i, j in UA_pair:
            if i in matching_ids2 and j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], matching_ids2[j]]))
                # UA_pair.remove(tuple([i,j]))
                # UA_pair.append(tuple([matching_ids2[i], matching_ids2[j]]))
            elif i in matching_ids2:
                # UA_pair.remove(tuple([i,j]))
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], j]))
                # UA_pair.append(tuple([matching_ids2[i],j]))
            elif j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([i, matching_ids2[j]]))

                # UA_pair.remove(tuple([i,j]))
                # UA_pair.append(tuple([i,matching_ids2[j]]))
        for p1, p2 in zip(remove_pairs, add_pairs):
            UA_pair.remove(p1)
            UA_pair.append(p2)
        return [UA_pair]

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs



def AC2mol(
    mol,
    AC,
    atoms,
    charge,
    allow_charged_fragments=True,
    use_graph=True,
    use_atom_maps=True,
    allow_carbenes=True,
):
    """"""

    # convert AC matrix to bond order (BO) matrix
    BO, atomic_valence_electrons = AC2BO(
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
        allow_carbenes=allow_carbenes,
    )
    # add BO connectivity and charge info to mol object
    mol = BO2mol(
        mol,
        BO,
        atoms,
        atomic_valence_electrons,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_atom_maps=use_atom_maps,
    )

    # print(Chem.GetFormalCharge(mol), charge)
    # If charge is not correct don't return mol
    if Chem.GetFormalCharge(mol) != charge:
        return None

    # BO2mol returns an arbitrary resonance form. Let's make the rest

    # mols = rdchem.ResonanceMolSupplier(mol)
    # mols = [mol for mol in mols]
    # print(mols)

    return mol

#============================================================#


def get_proposed_ligand_charge(ligand_mol, cutoff=-10):
    
    from rdkit.Chem import rdEHTTools
    
    valence_electrons = 0
    for a in ligand_mol.GetAtoms():
        valence_electrons += atomic_valence_electrons[a.GetAtomicNum()]

    passed, result = rdEHTTools.RunMol(ligand_mol)
    N_occ_orbs = sum(1 for i in result.GetOrbitalEnergies() if i < cutoff)
    charge = valence_electrons - 2 * N_occ_orbs

    #print(passed, charge)
    
    percieved_homo = result.GetOrbitalEnergies()[N_occ_orbs - 1]
    if N_occ_orbs == len(result.GetOrbitalEnergies()):
        percieved_lumo = np.nan
    else:
        percieved_lumo = result.GetOrbitalEnergies()[N_occ_orbs]
    while charge >= 1 and percieved_lumo < -9:
        N_occ_orbs += 1
        charge += -2
        #logger.debug("added two more electrons:", charge, percieved_lumo)
        percieved_lumo = result.GetOrbitalEnergies()[N_occ_orbs]
    while charge < -1 and percieved_homo > -10.2:
        N_occ_orbs -= 1
        charge += 2
        #logger.debug("removed two electrons:", charge, percieved_homo)
        percieved_homo = result.GetOrbitalEnergies()[N_occ_orbs - 1]

    return charge



