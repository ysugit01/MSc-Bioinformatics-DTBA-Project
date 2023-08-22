# Read this article:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4364066/
#
# The following article explaining the KronRLS algorithm is available within the
# supplementary data section of the above article:
# C:\Users\Koji\PycharmProjects\MSc Project (Yuri)\supp_bbu010_Supplementary_Methods.pdf

# Reference Code:
# https://github.com/aatapa/RLScore
#
# "KronRLS" Tutorial:
# http://staff.cs.utu.fi/~aatapa/software/RLScore/tutorial_kronecker.html#tutorial-1-kronrls
# https://github.com/aatapa/RLScore/blob/master/docs/tutorial_kronecker.rst
#
# Cindex measure tutorial:
# http://staff.cs.utu.fi/~aatapa/software/RLScore/tutorial_measure.html
#
# Load "davis data"
# https://github.com/aatapa/RLScore/blob/master/docs/src/davis_data.py
#
# KronRLS: Generalise to new drugs that were not observed in the training set:
# https://github.com/aatapa/RLScore/blob/master/docs/src/kron_rls2.py
#
# KronRLS: Generalise to new targets that were not observed in the training set:
# https://github.com/aatapa/RLScore/blob/master/docs/src/kron_rls3.py
#
# KronRLS: Generalise to new (u, v) pairs that neither have been observed in the training set:
# https://github.com/aatapa/RLScore/blob/master/docs/src/kron_rls4.py
#
# Cindex Tutorial
# http://staff.cs.utu.fi/~aatapa/software/RLScore/tutorial_measure.html

# Troubleshooting:
# =======================
# Reason for error when reading data file in module that has been imported:
# https://stackoverflow.com/questions/61289041/python-import-module-from-directory-error-reading-file

import numpy as np
import pandas as pd
from rlscore.learner import KronRLS
from rlscore.measure import cindex
from rlscore.measure import sqerror

target_gene_names = pd.read_csv("../data/target_gene_names.txt", header=None, index_col=0)
drug_pubchemIDs = pd.read_csv("../data/drug_PubChem_CIDs.txt", header=None, index_col=0)

f1, f2 = open("../data/target_gene_names.txt"), open("../data/drug_PubChem_CIDs.txt")
id1, id2 = [], []

# Load target gene names
for i in f1:
    k = i.strip("\n")
    id1.append(k)
# Load drug PubChem IDs
for i in f2:
    k = i.strip("\n")
    id2.append(int(k))

# Load drug-drug similarity scores
sim_drugs = pd.read_csv("../data/drug-drug_similarities_2D.txt", sep=" ", header=None, names=id2)
sim_drugs.index = id2
print("Drug similarities matrix: ", sim_drugs.shape)
# print(sim_drugs.head())

# Load normalised target-target similarity scores
sim_targets = pd.read_csv("../data/target-target_similarities_WS.txt", sep=" ", header=None, names=id1)
# sim_targets = pd.read_csv("../data/target-target_similarities_WS_normalized.txt", sep=" ", header=None, names=id1)
sim_targets.index = id1
print("Target similarities matrix: ", sim_targets.shape)
# print(sim_targets.head())

# Load drug-target interaction affinity scores
bindings = pd.read_csv("../data/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt", sep=" ", header=None, names=id1)
bindings.index = id2
print("Bindings matrix: ", bindings.shape)
# print(bindings.head())
print("")

# Sort all data
# drug_pubchemIDs.sort_index(inplace=True)
# target_gene_names.sort_index(inplace=True)
# bindings.sort_index(inplace=True)
# sim_targets.sort_index(inplace=True)
# sim_drugs.sort_index(inplace=True)

# Select 40 random drugs and 300 random targets
drug_ind = list(range(bindings.shape[0]))
target_ind = list(range(bindings.shape[1]))
np.random.seed(1)
np.random.shuffle(drug_ind)
np.random.shuffle(target_ind)
select_drug_ind = drug_ind[:40]
remain_drug_ind = drug_ind[40:]
select_target_ind = target_ind[:300]
remain_target_ind = target_ind[300:]

# Scenario A - Do not split (compute the in-sample leave-one-out cross-validation predictions)
scenario_A_drugs    = sim_drugs
scenario_A_targets  = sim_targets
scenario_A_bindings = bindings.values.ravel(order='F')

# Scenario B - Split according to drugs
scenario_B_bindings_train = bindings.iloc[select_drug_ind].values.ravel(order='F')
scenario_B_bindings_test  = bindings.iloc[remain_drug_ind].values.ravel(order='F')
scenario_B_drugs_train = sim_drugs.iloc[select_drug_ind]
scenario_B_drugs_test  = sim_drugs.iloc[remain_drug_ind]
scenario_B_targets_train = sim_targets
scenario_B_targets_test  = sim_targets

# Scenario C - Split according to targets
scenario_C_bindings_train = bindings.iloc[:, select_target_ind].values.ravel(order='F')
scenario_C_bindings_test  = bindings.iloc[:, remain_target_ind].values.ravel(order='F')
scenario_C_drugs_train = sim_drugs
scenario_C_drugs_test  = sim_drugs
scenario_C_targets_train = sim_targets.iloc[select_target_ind]
scenario_C_targets_test  = sim_targets.iloc[remain_target_ind]

# Scenario D - Split so that drug-target pairs do not overlap between training and test set
scenario_D_bindings_train = bindings.iloc[select_drug_ind, select_target_ind].values.ravel(order='F')
scenario_D_bindings_test  = bindings.iloc[remain_drug_ind, remain_target_ind].values.ravel(order='F')
scenario_D_drugs_train = sim_drugs.iloc[select_drug_ind]
scenario_D_drugs_test  = sim_drugs.iloc[remain_drug_ind]
scenario_D_targets_train = sim_targets.iloc[select_target_ind]
scenario_D_targets_test  = sim_targets.iloc[remain_target_ind]


print("Scenario A - Leave-one-out cross-validation predictions:")
learner = KronRLS(X1=scenario_A_drugs, X2=scenario_A_targets, Y=scenario_A_bindings)
best_regparam = None
best_cindex = 0.0
log_regparams = range(15, 35)
for log_regparam in log_regparams:
    learner.solve(2. ** log_regparam)
    # Computes the in-sample leave-one-out cross-validation prediction
    # http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/kron_rls.html
    P = learner.in_sample_loo()
    perf = cindex(scenario_A_bindings, P)
    if perf > best_cindex:
        best_regparam = log_regparam
        best_cindex = perf
    # print("regparam 2**%d, cindex %f" % (log_regparam, perf))
print("best regparam 2**%d with cindex %f" % (best_regparam, best_cindex))
print("")

print("Scenario B - Split_drugs:")
learner = KronRLS(X1=scenario_B_drugs_train, X2=scenario_B_targets_train, Y=scenario_B_bindings_train)
best_regparam = None
best_cindex = 0.0
log_regparams = range(15, 35)
for log_regparam in log_regparams:
    learner.solve(2. ** log_regparam)
    P = learner.predict(scenario_B_drugs_test, scenario_B_targets_test)
    perf = cindex(scenario_B_bindings_test, P)
    if perf>best_cindex:
        best_regparam=log_regparam
        best_cindex=perf
    # print("regparam 2**%d, cindex %f" % (log_regparam, perf))
print("best regparam 2**%d with cindex %f" % (best_regparam, best_cindex))
print("")

print("Scenario C - Split_targets:")
learner = KronRLS(X1=scenario_C_drugs_train, X2=scenario_C_targets_train, Y=scenario_C_bindings_train)
best_regparam = None
best_cindex = 0.0
log_regparams = range(15, 35)
for log_regparam in log_regparams:
    learner.solve(2. ** log_regparam)
    P = learner.predict(scenario_C_drugs_test, scenario_C_targets_test)
    perf = cindex(scenario_C_bindings_test, P)
    if perf>best_cindex:
        best_regparam=log_regparam
        best_cindex=perf
    # print("regparam 2**%d, cindex %f" % (log_regparam, perf))
print("best regparam 2**%d with cindex %f" % (best_regparam, best_cindex))
print("")

print("Scenario D - Split_pairs:")
learner = KronRLS(X1=scenario_D_drugs_train, X2=scenario_D_targets_train, Y=scenario_D_bindings_train)
best_regparam = None
best_cindex = 0.0
log_regparams = range(15, 35)
for log_regparam in log_regparams:
    learner.solve(2. ** log_regparam)
    P = learner.predict(scenario_D_drugs_test, scenario_D_targets_test)
    perf = cindex(scenario_D_bindings_test, P)
    if perf>best_cindex:
        best_regparam=log_regparam
        best_cindex=perf
    # print("regparam 2**%d, cindex %f" % (log_regparam, perf))
print("best regparam 2**%d with cindex %f" % (best_regparam, best_cindex))
print("")



# # Calculate RMSE for Test data
# # Then calculate accuracy, precision, recall, f1-score for X_test
# # Then  confusion_matrix (from sklearn)
# y_true=np.where(Y_test>7,1,0)
# y_pred=np.where(model.predict(X_test)>7,1,0)
#
# tn, fp, fn, tp =confusion_matrix(y_true,y_pred).ravel()
# print(tn, fp, fn, tp)
#
# test_rmse = mean_squared_error(Y_test,model.predict(X_test))**0.5
# test_acc =(tp+tn)/(tp+tn+fn+fp)
# test_precision = tp/(tp+fp)
# test_recall = tp/(tp+fn)
# test_f1 = (2*test_precision*test_recall)/(test_precision+test_recall)
#
# print("Test RMSE: %.3f" % test_rmse)
# print("Test Accuracy: %.3f" % test_acc)
# print("Test Precision: %.3f" % test_precision)
# print("Test Recall: %.3f" % test_recall)
# print("Test F1-Score: %.3f" % test_f1)
# print("")
