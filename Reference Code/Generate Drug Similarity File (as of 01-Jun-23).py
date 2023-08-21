# # import sys
# import numpy as np
# import pandas as pd
# from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit import DataStructs
# # from rdkit.Chem import rdFingerprintGenerator
#
# # show full results
# # np.set_printoptions(threshold=sys.maxsize)
#
def tanimoto_calc(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)
    return s

# SMILE1 = 'COc1ncc(-c2nc3C(=O)N(C(c3n2C(C)C)c2ccc(Cl)cc2)c2cn(C)c(=O)n(C)c2=O)c(OC)n1'
# SMILE2 = 'COc1ncc(-c2nc3C(=O)N(C(c3n2C(C)C)c2ccc(Cl)cc2)c2cc(Cl)cn(C)c2=O)c(OC)n1'
# print(tanimoto_calc(SMILE1, SMILE2))

smiles = {}
with open("smiles2.csv") as file:
    line = file.readline()
    while line:
        key, value = line.split('\t')
        smiles[key] = value
        line = file.readline()

# Load target gene names
# similarity_scores = pd.DataFrame(data = np.zeros((len(smiles), len(smiles))),
#                   index = smiles.keys(),
#                   columns = smiles.keys())
# print(similarity_scores)

# print(10008863, 123452926, tanimoto_calc(smiles['10008863'], smiles['123452926']))
# print(10008863, 91928583, tanimoto_calc(smiles["10008863"], smiles["122192875"]))

for PubChem_CID1 in smiles:
    for PubChem_CID2 in smiles:
        print(PubChem_CID1, PubChem_CID2, tanimoto_calc(smiles[PubChem_CID1], smiles[PubChem_CID2]))


    #     similarity_scores.at[PubChem_CID1, PubChem_CID2] = tanimoto_calc(smiles[PubChem_CID1], smiles[PubChem_CID2])
#         print(smiles[PubChem_CID1])
# for i in range(1, len(smiles)):
#     similarity = DataStructs.BulkTanimotoSimilarity(smiles[i], smiles[:i])
#     similarity_scores[i, :i] = similarity
#     similarity_scores[:i, i] = similarity
#
# print(similarity_scores)


import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
# from rdkit.Chem import AllChem

# show full results
np.set_printoptions(threshold=sys.maxsize)

# Reading the input CSV file.

ligands_df = pd.read_csv("../data/smiles.csv", sep='\t')
print(ligands_df)

# Creating molecules and storing in an array
molecules = []

"""Let's fetch the smiles from the input file and store in molecules array
        We have used '_' because we don't want any other column.
        If you want to fetch index and any other column, then replace '_' with 
            index and write column names after a ','.
"""

for _, smiles in ligands_df[["SMILES"]].itertuples():
    molecules.append((Chem.MolFromSmiles(smiles)))
# molecules[:15]
# print(ligands_df[["SMILES"]])
# Creating fingerprints for all molecules
print(molecules)

rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

# fgrps = [rdkit_gen.GetFingerprint(mol) for mol in molecules]

for mol in molecules:
    print(rdkit_gen.GetFingerprint(mol))



# Calculating number of fingerprints
# nfgrps = len(fgrps)
# print("Number of fingerprints:", nfgrps)

# for PubChem_CID1 in molecules:
#     for PubChem_CID2 in molecules:
#         print(PubChem_CID1, PubChem_CID2, tanimoto_calc(molecules[PubChem_CID1], molecules[PubChem_CID2]))

