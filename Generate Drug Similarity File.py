# Reference Code:
# https://github.com/MunibaFaiza/tanimoto_similarities

# Further Reading:
# https://projects.volkamerlab.org/teachopencadd/talktorials/T004_compound_similarity.html

import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

# show full results
np.set_printoptions(threshold=sys.maxsize)

# Reading the input CSV file.
smiles_data = pd.read_csv("data/GPCR/Smiles/smiles.csv", sep=',', header=0)

# Creating molecules and storing in an array
molecules = []

for _, smiles in smiles_data[["Ligand SMILES"]].itertuples():
    molecules.append((Chem.MolFromSmiles(smiles)))

# Creating fingerprints for all molecules
rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)
fgrps = [rdkit_gen.GetFingerprint(mol) for mol in molecules]

# Calculating number of fingerprints
nfgrps = len(fgrps)

# Defining a function to calculate similarities among the molecules
def pairwise_similarity(nfgrps):
    similarities = np.zeros((nfgrps, nfgrps))
    similarities.fill(1)

    for i in range(1, nfgrps):
        similarity = DataStructs.BulkTanimotoSimilarity(fgrps[i], fgrps[:i])
        similarities[i, :i] = similarity
        similarities[:i, i] = similarity

    return similarities

# Calculating similarities of molecules
df = pd.DataFrame(data=pairwise_similarity(nfgrps),
                index=smiles_data["PubChem CID"],
                columns=smiles_data["PubChem CID"])
print(df)

df.to_csv('data/GPCR/drug_similarities.csv', sep=',')