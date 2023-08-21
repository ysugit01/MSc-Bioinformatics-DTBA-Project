# Read this article:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4364066/
#
# The following article explaining the KronRLS algorithm is available within the
# supplementary data section of the above article:
# C:\Users\Koji\PycharmProjects\MSc Project (Yuri)\supp_bbu010_Supplementary_Methods.pdf

# Github Reference Code:
# https://github.com/aatapa/RLScore
#
# "KronRLS" Tutorial:
# http://staff.cs.utu.fi/~aatapa/software/RLScore/tutorial_kernels.html
# http://staff.cs.utu.fi/~aatapa/software/RLScore/tutorial_kronecker.html
#
# Cindex Tutorial
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
# Pearson Correlation:
# https://realpython.com/numpy-scipy-pandas-correlation-python/

# Troubleshooting:
# =======================
# Reason for error when reading data file in module that has been imported:
# https://stackoverflow.com/questions/61289041/python-import-module-from-directory-error-reading-file

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
# from rlscore.learner import KronRLS
from rlscore.learner import TwoStepRLS
from rlscore.measure import cindex
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from rlscore.utilities.cross_validation import random_folds
from lifelines.utils import concordance_index


# Evaluate performance of each model (i.e. concordance index, confusion_matrix, MSE, RMSE, pearson correlation,
# accuracy, precision, recall, f1-score)
def eval_model(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Concordance Index Tutorial:
    # https://medium.com/analytics-vidhya/concordance-index-72298c11eac7
    test_cindex = concordance_index(y_true, y_pred)
    test_mse = mean_squared_error(y_true, y_pred)
    # Pearson Correlation Tutorial:
    # https://realpython.com/numpy-scipy-pandas-correlation-python/
    test_pearson, ignore_var = pearsonr(y_true, y_pred)
    # R2 Score Tutorial:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    test_r2 = r2_score(y_true, y_pred)
    test_acc = (tp + tn) / (tp + tn + fn + fp)
    test_precision = tp / (tp + fp)
    test_recall = tp / (tp + fn)
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall)

    print("Concordance Index: ", test_cindex)
    print("True Negative:", tn, ", False Positive:", fp, ", False Negative:", fn, ", True Positive:", tp)
    # print("Test MSE: %.3f" % test_mse)
    print("Test RMSE: %.3f" % test_mse ** 0.5)
    print("Test Pearson Correlation: %.3f" % test_pearson)
    print("Test R2 Score: %.3f" % test_r2)
    print("Test Accuracy: %.3f" % test_acc)
    print("Test Precision: %.3f" % test_precision)
    print("Test Recall: %.3f" % test_recall)
    print("Test F1-Score: %.3f" % test_f1)
    print("")


target_gene_names = pd.read_csv("data/target_gene_names.txt", header=None, index_col=0)
drug_pubchemIDs = pd.read_csv("data/drug_PubChem_CIDs.txt", header=None, index_col=0)

f1, f2 = open("data/target_gene_names.txt"), open("data/drug_PubChem_CIDs.txt")
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
sim_drugs = pd.read_csv("data/drug-drug_similarities_2D.txt", sep=" ", header=None, names=id2)
sim_drugs.index = id2
print("Drug similarities matrix: ", sim_drugs.shape)
# print(sim_drugs.head())

# Load normalised target-target similarity scores
# sim_targets = pd.read_csv("data/target-target_similarities_WS.txt", sep=" ", header=None, names=id1)
sim_targets = pd.read_csv("data/target-target_similarities_WS_normalized.txt", sep=" ", header=None, names=id1)
sim_targets.index = id1
print("Target similarities matrix: ", sim_targets.shape)
# print(sim_targets.head())

# Load drug-target interaction affinity scores
bindings = pd.read_csv("data/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt", sep=" ", header=None, names=id1)
bindings.index = id2
print("Bindings matrix: ", bindings.shape)

# transform binding values
# sim_targets = sim_targets/100
bindings = -np.log10(bindings/(10**9))
# print("bindings.head()", bindings.head())
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
select_drug_ind   = drug_ind[:40]
remain_drug_ind   = drug_ind[40:]
select_target_ind = target_ind[:300]
remain_target_ind = target_ind[300:]

# Scenario A - Data is not split (in-sample leave-one-out cross-validation predictions computed)
scenario_A_drugs    = sim_drugs
scenario_A_targets  = sim_targets
scenario_A_bindings = bindings.values.ravel(order='F')

# Scenario B - Split according to drugs
scenario_B_bindings_train = bindings.iloc[select_drug_ind].values.ravel(order='F')
scenario_B_bindings_test  = bindings.iloc[remain_drug_ind].values.ravel(order='F')
scenario_B_drugs_train    = sim_drugs.iloc[select_drug_ind]
scenario_B_drugs_test     = sim_drugs.iloc[remain_drug_ind]
scenario_B_targets_train  = sim_targets
scenario_B_targets_test   = sim_targets

# Scenario C - Split according to targets
scenario_C_bindings_train = bindings.iloc[:, select_target_ind].values.ravel(order='F')
scenario_C_bindings_test  = bindings.iloc[:, remain_target_ind].values.ravel(order='F')
scenario_C_drugs_train    = sim_drugs
scenario_C_drugs_test     = sim_drugs
scenario_C_targets_train  = sim_targets.iloc[select_target_ind]
scenario_C_targets_test   = sim_targets.iloc[remain_target_ind]

# Scenario D - Split so that drug-target pairs do not overlap between training and test set
scenario_D_bindings_train = bindings.iloc[select_drug_ind, select_target_ind].values.ravel(order='F')
scenario_D_bindings_test  = bindings.iloc[remain_drug_ind, remain_target_ind].values.ravel(order='F')
scenario_D_drugs_train    = sim_drugs.iloc[select_drug_ind]
scenario_D_drugs_test     = sim_drugs.iloc[remain_drug_ind]
scenario_D_targets_train  = sim_targets.iloc[select_target_ind]
scenario_D_targets_test   = sim_targets.iloc[remain_target_ind]

# print("Scenario A - Leave-one-out cross-validation predictions:")
# learner = TwoStepRLS(X1=scenario_A_drugs, X2=scenario_A_targets, Y=scenario_A_bindings, regparam1=1.0, regparam2=1.0)
# best_regparam = None
# best_cindex = 0.0
# log_regparams1 = range(-20, 15)
# log_regparams2 = range(-20, 15)
# for log_regparam1 in log_regparams1:
#     for log_regparam2 in log_regparams2:
#         learner.solve(2. ** log_regparam1, 2. ** log_regparam2)
#         # Computes the in-sample leave-one-out cross-validation prediction
#         # http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/kron_rls.html
#         P = learner.in_sample_loo()
#         perf = cindex(scenario_A_bindings, P)
#         if perf > best_cindex:
#             best_regparam = log_regparam
#             best_cindex = perf
#             best_predict = P
#     # print("regparam 2**%d, cindex %f" % (log_regparam, perf))
# print("best regparam 2**%d with cindex %f" % (best_regparam, best_cindex))
# y_true=np.where(scenario_A_bindings>7, 1, 0)
# y_pred=np.where(best_predict>7,1,0)
# eval_model(y_true, y_pred)


print("Scenario B - Split_drugs:")
learner = TwoStepRLS(X1=scenario_B_drugs_train, X2=scenario_B_targets_train, Y=scenario_B_bindings_train, regparam1=1.0, regparam2=1.0)
m = scenario_B_drugs_train.shape[0]
best_regparam = None
best_predict = None
best_cindex = 0.0
log_regparams1 = range(-20, 15)
log_regparams2 = range(-20, 15)
folds = random_folds(m, 5, seed=1)
print("scenario_B_bindings_train: ", scenario_B_bindings_train.shape)
print("scenario_B_bindings_test: ", scenario_B_bindings_test.shape)
print("scenario_B_drugs_train: ", scenario_B_drugs_train.shape)
print("scenario_B_drugs_test: ", scenario_B_drugs_test.shape)
print("scenario_B_targets_train: ", scenario_B_targets_train.shape)
print("scenario_B_targets_test: ", scenario_B_targets_test.shape)
for log_regparam1 in log_regparams1:
    for log_regparam2 in log_regparams2:
        learner.solve(2. ** log_regparam1, 2. ** log_regparam2)
        # Computes the in-sample leave-one-out cross-validation prediction
        # http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/kron_rls.html
        P = learner.x2_kfold_cv(folds)
        perf = cindex(scenario_B_bindings_train, P)
        if perf > best_cindex:
            best_regparam1, best_regparam2 = log_regparam1, log_regparam2
            best_cindex = perf
            best_predict = P
        # print("regparam1 2**%d, regparam2 2**%d, cindex %f" % (log_regparam1, log_regparam2, perf))
# print("best regparam1 2**%d, regparam2 2**%d with cindex %f" % (best_regparam1, best_regparam2, best_cindex))
P = learner.predict(scenario_B_drugs_test, scenario_B_targets_test)
y_true=np.where(scenario_B_bindings_test>7,1,0)
y_pred=np.where(P>7,1,0)
print("y_true: ", y_true.shape)
print("y_pred: ", y_pred.shape)
eval_model(y_true, y_pred)

# print("scenario_C_bindings_train: ", scenario_C_bindings_train.shape)
# print("scenario_C_drugs_train: ", scenario_C_drugs_train.shape)
# print("scenario_C_targets_train: ", scenario_C_targets_train.shape)
# print("scenario_D_bindings_train: ", scenario_D_bindings_train.shape)
# print("scenario_D_drugs_train: ", scenario_D_drugs_train.shape)
# print("scenario_D_targets_train: ", scenario_D_targets_train.shape)


print("Scenario C - Split_targets:")
learner = TwoStepRLS(X1=scenario_C_drugs_train, X2=scenario_C_targets_train, Y=scenario_C_bindings_train, regparam1=1.0, regparam2=1.0)
n = scenario_C_targets_train.shape[0]
best_regparam = None
best_cindex = 0.0
log_regparams1 = range(-20, 15)
log_regparams2 = range(-20, 15)
folds = random_folds(n, 5, seed=1)
for log_regparam1 in log_regparams1:
    for log_regparam2 in log_regparams2:
        learner.solve(2. ** log_regparam1, 2. ** log_regparam2)
        # Computes the in-sample leave-one-out cross-validation prediction
        # http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/kron_rls.html
        P = learner.predict(scenario_C_drugs_test, scenario_C_targets_test)
        perf = cindex(scenario_C_bindings_test, P)
        P = learner.x2_kfold_cv(folds)
        if perf > best_cindex:
            best_regparam1, best_regparam2 = log_regparam1, log_regparam2
            best_cindex = perf
            best_predict = P
        # print("regparam1 2**%d, regparam2 2**%d, cindex %f" % (log_regparam1, log_regparam2, perf))
# print("best regparam1 2**%d, regparam2 2**%d with cindex %f" % (best_regparam1, best_regparam2, best_cindex))
y_true=np.where(scenario_C_bindings_test>7,1,0)
y_pred=np.where(best_predict>7,1,0)
eval_model(y_true, y_pred)


print("Scenario D - Split_pairs:")
learner = TwoStepRLS(X1=scenario_D_drugs_train, X2=scenario_D_targets_train, Y=scenario_D_bindings_train, regparam1=1.0, regparam2=1.0)
m = scenario_D_drugs_train.shape[0]
n = scenario_D_targets_train.shape[0]
best_regparam = None
best_cindex = 0.0
log_regparams1 = range(-20, 15)
log_regparams2 = range(-20, 15)
drug_folds = random_folds(m, 5, seed=1)
target_folds = random_folds(n, 5, seed=1)
for log_regparam1 in log_regparams1:
    for log_regparam2 in log_regparams2:
        learner.solve(2. ** log_regparam1, 2. ** log_regparam2)
        # Computes the in-sample leave-one-out cross-validation prediction
        # http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/kron_rls.html
        P = learner.predict(scenario_D_drugs_test, scenario_D_targets_test)
        perf = cindex(scenario_D_bindings_test, P)
        P = learner.out_of_sample_kfold_cv(drug_folds, target_folds)
        if perf > best_cindex:
            best_regparam1, best_regparam2 = log_regparam1, log_regparam2
            best_cindex = perf
            best_predict = P
    # print("regparam1 2**%d, regparam2 2**%d, cindex %f" % (log_regparam1, log_regparam2, perf))
# print("best regparam1 2**%d, regparam2 2**%d with cindex %f" % (best_regparam1, best_regparam2, best_cindex))
y_true=np.where(scenario_D_bindings_test>7,1,0)
y_pred=np.where(best_predict>7,1,0)
eval_model(y_true, y_pred)
