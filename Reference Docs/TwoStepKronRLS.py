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
# TwoStepRLS Function Definitions:
# http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/two_step_rls.html
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
# Pearson Correlation Tutorial:
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
from lifelines.utils import concordance_index
# from rlscore.utilities.cross_validation import random_folds

# pd.set_option('display.max_columns', None)

# Load the raw data (i.e. drud-drug, target-target, and drug-target similarity scores)
def load_data():
    # Load normalised target-target similarity scores and set the index for rows and columns with the gene names
    target_gene_names = pd.read_csv("../data/target_gene_names.txt", header=None, index_col=0)
    sim_targets = pd.read_csv("../data/target-target_similarities_WS.txt", sep=" ", header=None, names=target_gene_names.index)
    # sim_targets = pd.read_csv("data/target-target_similarities_WS_normalized.txt", sep=" ", header=None, names=target_gene_names.index)
    sim_targets.index = target_gene_names.index

    # Load drug-drug similarity scores and set the index for rows and columns with the drug names
    drug_pubchemIDs = pd.read_csv("../data/drug_PubChem_CIDs.txt", header=None, index_col=0)
    sim_drugs = pd.read_csv("../data/drug-drug_similarities_2D.txt", sep=" ", header=None, names=drug_pubchemIDs.index)
    sim_drugs.index = drug_pubchemIDs.index

    # Load drug-target interaction affinity scores and set the row index with  drug names and column index with gene names
    bindings = pd.read_csv("../data/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt", sep=" ", header=None, names=target_gene_names.index)
    bindings.index = drug_pubchemIDs.index

    print("Target similarities matrix: ", sim_targets.shape)
    print("Drug similarities matrix: ", sim_drugs.shape)
    print("Bindings matrix: ", bindings.shape)
    print("")

    # Apply Min-Max Normalisation:
    # https://www.geeksforgeeks.org/how-to-scale-pandas-dataframe-columns/
    sim_targets = (sim_targets - sim_targets.min()) / (sim_targets.max() - sim_targets.min())

    # Apply pKd affinity score for bindings:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5395521/#CR4
    bindings = -np.log10(bindings/(10**9))

    # Sort the data
    drug_pubchemIDs.sort_index(inplace=True)
    target_gene_names.sort_index(inplace=True)
    bindings.sort_index(inplace=True)
    bindings = bindings.reindex(sorted(bindings.columns), axis=1)
    sim_targets.sort_index(inplace=True)
    sim_targets = sim_targets.reindex(sorted(sim_targets.columns), axis=1)
    sim_drugs.sort_index(inplace=True)
    sim_drugs = sim_drugs.reindex(sorted(sim_drugs.columns), axis=1)

    return target_gene_names, drug_pubchemIDs, sim_drugs, sim_targets, bindings


# Scenario B - Split according to drugs
def split_B(sim_drugs, sim_targets, bindings):
    # Select 40 random drugs and 300 random targets
    drug_ind = list(range(bindings.shape[0]))
    target_ind = list(range(bindings.shape[1]))
    np.random.seed(123)
    np.random.shuffle(drug_ind)
    np.random.shuffle(target_ind)
    select_drug_ind = drug_ind[:40]
    remain_drug_ind = drug_ind[40:]
    # select_target_ind = target_ind[:300]
    # remain_target_ind = target_ind[300:]

    bindings_train = bindings.iloc[select_drug_ind].values.ravel(order='F')
    bindings_test  = bindings.iloc[remain_drug_ind].values.ravel(order='F')
    drugs_train    = sim_drugs.iloc[select_drug_ind]
    drugs_test     = sim_drugs.iloc[remain_drug_ind]
    targets_train  = sim_targets
    targets_test   = sim_targets

    return drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test


# Scenario C - Split according to targets
def split_C(sim_drugs, sim_targets, bindings):
    # Select 40 random drugs and 300 random targets
    drug_ind = list(range(bindings.shape[0]))
    target_ind = list(range(bindings.shape[1]))
    np.random.seed(123)
    np.random.shuffle(drug_ind)
    np.random.shuffle(target_ind)
    # select_drug_ind = drug_ind[:40]
    # remain_drug_ind = drug_ind[40:]
    select_target_ind = target_ind[:300]
    remain_target_ind = target_ind[300:]

    bindings_train = bindings.iloc[:, select_target_ind].values.ravel(order='F')
    bindings_test  = bindings.iloc[:, remain_target_ind].values.ravel(order='F')
    drugs_train    = sim_drugs
    drugs_test     = sim_drugs
    targets_train  = sim_targets.iloc[select_target_ind]
    targets_test   = sim_targets.iloc[remain_target_ind]

    return drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test


# Scenario D - Split so that drug-target pairs do not overlap between training and test set
def split_D(sim_drugs, sim_targets, bindings):
    # Select 40 random drugs and 300 random targets
    drug_ind = list(range(bindings.shape[0]))
    target_ind = list(range(bindings.shape[1]))
    np.random.seed(123)
    np.random.shuffle(drug_ind)
    np.random.shuffle(target_ind)
    select_drug_ind = drug_ind[:40]
    remain_drug_ind = drug_ind[40:]
    select_target_ind = target_ind[:300]
    remain_target_ind = target_ind[300:]

    bindings_train = bindings.iloc[select_drug_ind, select_target_ind].values.ravel(order='F')
    bindings_test  = bindings.iloc[remain_drug_ind, remain_target_ind].values.ravel(order='F')
    drugs_train    = sim_drugs.iloc[select_drug_ind]
    drugs_test     = sim_drugs.iloc[remain_drug_ind]
    targets_train  = sim_targets.iloc[select_target_ind]
    targets_test   = sim_targets.iloc[remain_target_ind]

    return drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test


# # Scenario A - Data is not split (in-sample leave-one-out cross-validation predictions computed)
def train_model_A(sim_drugs, sim_targets, bindings):
    print("Setting 1 (S1): Leave One Drug-Target Pair Out Cross-Validation:")
    print("======================")
    a_drugs    = sim_drugs.values
    a_targets  = sim_targets.values
    a_bindings = bindings.values.ravel(order='F')
    # n = a_drugs.shape[0]
    # m = a_targets.shape[0]

    # Create random split to 5 folds for the drugs
    # folds = random_folds(n*m, 5, seed=123)
    # folds = [np.unravel_index(fold, (n, m)) for fold in folds]

    # learner = TwoStepRLS(X1=a_drugs, X2=a_targets, Y=a_bindings, regparam1=1.0, regparam2=1.0)
    learner = TwoStepRLS(X1=sim_drugs.values, X2=sim_targets.values, Y=bindings.values.ravel(order='F'), regparam1=1.0, regparam2=1.0)
    best_regparam_1 = None
    best_regparam_2 = None
    best_predict = None
    best_cindex = 0.0
    log_regparams1 = range(-10, -5)
    log_regparams2 = range(-4, 0)
    for log_regparam1 in log_regparams1:
        for log_regparam2 in log_regparams2:
            learner.solve(2. ** log_regparam1, 2. ** log_regparam2)
            P = learner.in_sample_loo()
            # P = learner.in_sample_kfoldcv(folds)
            # perf = cindex(a_bindings, P)
            perf = cindex(bindings.values.ravel(order='F'), P)
            if perf > best_cindex:
                best_regparam_1 = log_regparam1
                best_regparam_2 = log_regparam2
                best_cindex = perf
                best_predict = P
            # print("regparam 2**%d 2**%d, cindex %f" % (log_regparam1, log_regparam2, perf))
    # print("best regparam1: 2**%d, regparam2: 2**%d, with cindex: %f" % (best_regparam_1, best_regparam_2, best_cindex))

    # y_true = np.where(a_bindings>7, 1, 0)
    y_true = bindings.values.ravel(order='F')
    # y_pred = np.where(best_predict > 7, 1, 0)
    y_pred = best_predict
    return y_true, y_pred


def train_model_B_C_D(sim_drugs, sim_targets, bindings, scenario):
    if scenario=="B":
        print("Setting 2 (S2): Leave One Drug Out Cross-Validation):")
        print("======================")
        drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test = split_B(sim_drugs, sim_targets, bindings)
    elif scenario=="C":
        print("Setting 3 (S3): Leave One Target Out Cross-Validation):")
        print("======================")
        drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test = split_C(sim_drugs, sim_targets, bindings)
    # elif scenario == "D":
    #     print("Scenario D - Split Drug-Target Pairs")
    #     drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test = split_D(sim_drugs, sim_targets, bindings)
    else:
        return None, None

    drugs_train = drugs_train.values
    drugs_test = drugs_test.values
    targets_train = targets_train.values
    targets_test = targets_test.values
    # bindings_train = bindings_train.values
    # bindings_test = bindings_test.values

    # n = drugs_train.shape[0]
    # learner = TwoStepRLS(X1=drugs_train, X2=targets_train, Y=bindings_train, regparam1=1.0, regparam2=1.0)
    learner = TwoStepRLS(X1=sim_drugs.values, X2=sim_targets.values, Y=bindings.values.ravel(order='F'), regparam1=1.0, regparam2=1.0)
    best_regparam_1 = None
    best_regparam_2 = None
    # best_predict = None
    best_cindex = 0.0
    log_regparams1 = range(-20, -15)
    log_regparams2 = range(-10, -5)
    # folds = random_folds(n, 5, seed=123)
    for log_regparam1 in log_regparams1:
        for log_regparam2 in log_regparams2:
            P = None
            learner.solve(2. ** log_regparam1, 2. ** log_regparam2)
            # P = learner.predict(drugs_test, targets_test)
            # perf = cindex(bindings_test, P)
            # print("regparam 2**%d 2**%d, test set cindex %f" % (log_regparam1, log_regparam2, perf))
            # P = learner.in_sample_loo()
            if scenario == "B":
                P = learner.leave_x1_out()
            elif scenario == "C":
                P = learner.leave_x2_out()
            # P = learner.x2_kfold_cv(folds)
            # perf = cindex(bindings_train, P)
            perf = cindex(bindings.values.ravel(order='F'), P)
            # print("regparam 2**%d 2**%d, K-fold cindex %f" % (log_regparam1, log_regparam2, perf))
            if perf > best_cindex:
                best_regparam_1 = log_regparam1
                best_regparam_2 = log_regparam2
                best_cindex = perf
                best_predict = P
    # print("best regparam1: 2**%d, regparam2: 2**%d, with cindex: %f" % (best_regparam_1, best_regparam_2, best_cindex))
    # learner.solve(2. ** best_regparam_1, 2. ** best_regparam_2)
    # P = learner.predict(drugs_test, targets_test)
    # y_true = np.where(bindings_test > 7, 1, 0)
    y_true = bindings.values.ravel(order='F')
    # y_pred = np.where(best_predict > 7, 1, 0)
    y_pred = best_predict
    return y_true, y_pred


# Evaluate performance of each model (i.e. concordance index, confusion_matrix, MSE, RMSE, pearson correlation,
# accuracy, precision, recall, f1-score)
def eval_model(y_true, y_pred):
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
    print("Concordance Index: ", test_cindex)
    # print("Test MSE: %.3f" % test_mse)
    print("RMSE: %.3f" % test_mse ** 0.5)
    print("Pearson Correlation: %.3f" % test_pearson)
    print("R2 Score: %.3f" % test_r2)

    threshold = 6
    y_true_bin = np.where(y_true > threshold, 1, 0)
    y_pred_bin = np.where(y_pred > threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    test_acc = (tp + tn) / (tp + tn + fn + fp)
    test_precision = tp / (tp + fp)
    test_recall = tp / (tp + fn)
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall)
    print("True Negative:", tn, ", False Positive:", fp, ", False Negative:", fn, ", True Positive:", tp)
    print("Accuracy: %.3f" % test_acc)
    print("Precision: %.3f" % test_precision)
    print("Recall: %.3f" % test_recall)
    print("F1-Score: %.3f" % test_f1)
    print("")


def main():
    # Load drugs, targets, and bindings data
    target_gene_names, drug_pubchemIDs, sim_drugs, sim_targets, bindings = load_data()

    y_true, y_pred = train_model_A(sim_drugs, sim_targets, bindings)
    eval_model(y_true, y_pred)

    y_true, y_pred = train_model_B_C_D(sim_drugs, sim_targets, bindings, "B")
    eval_model(y_true, y_pred)

    y_true, y_pred = train_model_B_C_D(sim_drugs, sim_targets, bindings, "C")
    eval_model(y_true, y_pred)

    # y_true, y_pred = train_model_B_C_D(sim_drugs, sim_targets, bindings, "D")
    # eval_model(y_true, y_pred)

if __name__=="__main__":
    main()
