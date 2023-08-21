filepath = 'data/GPCR/746x12/'
conf_matrx_threshold = 6.5

# Read this article:
# https://www.frontiersin.org/articles/10.3389/fchem.2019.00782/full
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
# Pearson Correlation Tutorial:
# https://realpython.com/numpy-scipy-pandas-correlation-python/
#
# KronRLS Function Definitions:
# http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/kron_rls.html

# Troubleshooting:
# =======================
# Reason for error when reading data file in module that has been imported:
# https://stackoverflow.com/questions/61289041/python-import-module-from-directory-error-reading-file

import numpy as np
import pandas as pd
from rlscore.learner import TwoStepRLS
from rlscore.measure import cindex
from rlscore.utilities.cross_validation import random_folds
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import seaborn as sn
# from rlscore.utilities.cross_validation import random_folds


# Load the raw data (i.e. drug-drug, target-target, and drug-target similarity scores)
def load_data():
    # Load drug-drug similarity scores and set the index for rows and columns with the drug names
    sim_drugs = pd.read_csv(filepath + "drug_similarities.csv", sep=",", header=0, index_col=0)

    # Load normalised target-target similarity scores and set the index for rows and columns with the gene names
    sim_targets = pd.read_csv(filepath + "target_similarities_norm.csv", sep=",", header=0, index_col=0)

    # Load drug-target interaction affinity scores and set the row index with  drug names and column index with gene names
    bindings = pd.read_csv(filepath + "drug_target_matrix.csv", sep=",", header=0, index_col=0)

    print("Drug similarities matrix: ", sim_drugs.shape)
    print("Target similarities matrix: ", sim_targets.shape)
    print("Bindings matrix: ", bindings.shape)
    print("")

    # Apply pKd affinity score for bindings:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5395521/#CR4
    bindings = bindings.fillna(10000000)
    bindings = -np.log10(bindings/(10**9))

    # Sort the data
    sim_drugs.sort_index(inplace=True)
    sim_drugs = sim_drugs.reindex(sorted(sim_drugs.columns), axis=1)
    sim_targets.sort_index(inplace=True)
    sim_targets = sim_targets.reindex(sorted(sim_targets.columns), axis=1)
    bindings.sort_index(inplace=True)
    bindings = bindings.reindex(sorted(bindings.columns), axis=1)

    return sim_drugs, sim_targets, bindings


# http://staff.cs.utu.fi/~aatapa/software/RLScore/tutorial_kronecker.html#tutorial-2-twosteprls-cross-validation-with-bipartite-network
def train_model(learner, drugs_train, targets_train, bindings_train, scenario):
    bindings_train_flat = bindings_train.values.ravel(order='F')
    n_folds = 5
    m = drugs_train.shape[0]
    n = targets_train.shape[0]
    best_regparam = None
    best_predict = None
    best_cindex = 0.0
    log_regparams1 = range(-20, 15)
    log_regparams2 = range(-20, 15)

    if scenario == 'A':
        folds = random_folds(n*m, n_folds, seed=1)
        # Map the indices back to (drug_indices, target_indices)
        folds = [np.unravel_index(fold, (n, m)) for fold in folds]
    elif scenario == 'B':
        drug_folds = random_folds(m, n_folds, seed=1)
    elif scenario == 'C':
        target_folds = random_folds(n, n_folds, seed=1)
    elif scenario == 'D':
        drug_folds = random_folds(n, n_folds, seed=1)
        target_folds = random_folds(m, n_folds, seed=1)
    else:
        print("Invalid scenario selected!")
        exit()

    for log_regparam1 in log_regparams1:
        for log_regparam2 in log_regparams2:
            learner.solve(2. ** log_regparam1, 2. ** log_regparam2)
            # Computes the in-sample leave-one-out cross-validation prediction
            # http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/kron_rls.html
            if scenario == 'A':
                # P = learner.in_sample_loo()
                P = learner.in_sample_kfoldcv(folds)
            elif scenario == 'B':
                # P = learner.leave_x1_out()
                P = learner.x1_kfold_cv(drug_folds)
            elif scenario == 'C':
                # P = learner.leave_x2_out()
                P = learner.x2_kfold_cv(target_folds)
            else:
                # P = learner.out_of_sample_loo()
                P = learner.out_of_sample_kfold_cv(drug_folds, target_folds)
            perf = cindex(bindings_train_flat.to_numpy(), P)
            if perf > best_cindex:
                best_regparam1, best_regparam2 = log_regparam1, log_regparam2
                best_cindex = perf
                best_predict = P
    print("best regparam1 2**%d, regparam2 2**%d with cindex %f" % (best_regparam1, best_regparam2, best_cindex))
    print("")

    y_true = bindings_train_flat
    y_pred = best_predict

    labels = []
    for target in bindings_train.columns:
        for drug in bindings_train.index:
            labels.append(str(drug) + "|"+ target)

    test_data = pd.DataFrame(y_true, columns=['y_true'], index=labels)
    test_data['y_pred'] = y_pred
    test_data.to_csv(filepath + 'KronRLS/KronRLS_' + scenario + '_test.csv', sep=',')

    return y_true, y_pred


# https://www.geeksforgeeks.org/seaborn-heatmap-a-comprehensive-guide/
def heatmap(bindings):
    sn.heatmap(data=bindings)
    plt.show()

# Evaluate performance of each model (i.e. concordance index, confusion_matrix, MSE, RMSE, pearson correlation,
# accuracy, precision, recall, f1-score)
def eval_model(y_true, y_pred, axis, chart_id):
    y_true_bin = np.where(y_true > conf_matrx_threshold, 1, 0)
    y_pred_bin = np.where(y_pred > conf_matrx_threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    print("True Negative:", tn, ", False Positive:", fp, ", False Negative:", fn, ", True Positive:", tp)

    test_acc = (tp + tn) / (tp + tn + fn + fp)
    print("Test Accuracy: %.3f" % test_acc)

    test_precision = tp / (tp + fp)
    print("Test Precision: %.3f" % test_precision)

    test_recall = tp / (tp + fn)
    print("Test Recall: %.3f" % test_recall)

    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall)
    print("Test F1-Score: %.3f" % test_f1)

    roc_auc = roc_auc_score(y_true_bin, y_pred_bin)
    print("ROC AUC: %.3f" % roc_auc)

    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_pred_bin)
    auprc = auc(recall, precision)
    print("AUPRC: %.3f" % auprc)

    # Concordance Index Tutorial:
    # https://medium.com/analytics-vidhya/concordance-index-72298c11eac7
    test_cindex = concordance_index(y_true, y_pred)
    print("Concordance Index: ", test_cindex)

    test_mse = mean_squared_error(y_true, y_pred)
    # print("Test MSE: %.3f" % test_mse)
    print("Test RMSE: %.3f" % test_mse ** 0.5)

    test_spearman, ignore_var = spearmanr(y_true, y_pred)
    print("Test Spearman Correlation: %.3f" % test_spearman)

    print("")

    if chart_id == 0:
        axis[chart_id // 2, chart_id % 2].set_title("A")
    elif chart_id == 1:
        axis[chart_id // 2, chart_id % 2].set_title("B")
    elif chart_id == 2:
        axis[chart_id // 2, chart_id % 2].set_title("C")
    elif chart_id == 3:
        axis[chart_id // 2, chart_id % 2].set_title("D")
    axis[chart_id // 2, chart_id % 2].set_xlim(left=2, right=10)
    axis[chart_id // 2, chart_id % 2].set_ylim(bottom=0, top=10)
    axis[chart_id // 2, chart_id % 2].set_xlabel("Actual Values")
    axis[chart_id // 2, chart_id % 2].set_ylabel("Predicted Values")
    axis[chart_id // 2, chart_id % 2].scatter(x=y_true, y=y_pred)
    a, b = np.polyfit(y_true, y_pred, 1)
    axis[chart_id // 2, chart_id % 2].plot(y_true, a * y_true + b, color="red")


def main():
    # pd.set_option('display.max_columns', None)

    # Load drugs, targets, and bindings data
    sim_drugs, sim_targets, bindings = load_data()

    # specifying the plot size
    plt.figure(figsize=(10, 5))
    # Specify the binarization threshold
    plt.axvline(x=conf_matrx_threshold, color='r', label='Binarization Threshold')
    # Create histogram with 10 buckets
    plt.hist(bindings.values.ravel(), bins=10)
    plt.title("Distribution of Values in the Dataset")
    plt.xlabel("pIC50 (nM)")
    plt.ylabel("Frequency")
    plt.show()

    print("Confusion Matrix Threshold:", conf_matrx_threshold)
    print("")

    print("TwoStepRLS (K-Fold):")
    print("")

    # specifying the plot size
    figure, axis = plt.subplots(2, 2, figsize=(10, 6))

    print("Scenario A - Leave-one-out cross-validation predictions:")
    print("======================")
    # Leave-pair-out cross-validation. On each round of CV, one (drug,target) pair is left out of the training set as test pair.
    learner = TwoStepRLS(X1=sim_drugs.to_numpy(), X2=sim_targets.to_numpy(), Y=bindings.to_numpy(), regparam1=1.0, regparam2=1.0)
    y_true, y_pred = train_model(learner, sim_drugs, sim_targets, bindings, 'A')
    eval_model(y_true, y_pred, axis, 0)


    print("Scenario B - Split_drugs:")
    print("======================")
    # Leave-drug-out cross-validation. On each CV round, a single holdout drug is left out, and all (drug, target) pairs this drug
    # belongs to used as the test fold.
    learner = TwoStepRLS(X1=sim_drugs.to_numpy(), X2=sim_targets.to_numpy(), Y=bindings.to_numpy(), regparam1=1.0, regparam2=1.0)
    y_true, y_pred = train_model(learner, sim_drugs, sim_targets, bindings, 'B')
    eval_model(y_true, y_pred, axis, 1)


    print("Scenario C - Split_targets:")
    print("======================")
    # Leave-target-out cross-validation. On each CV round, a single holdout target is left out, and all (drug, target) pairs this
    # target belongs to used as the test fold.
    learner = TwoStepRLS(X1=sim_drugs.to_numpy(), X2=sim_targets.to_numpy(), Y=bindings.to_numpy(), regparam1=1.0, regparam2=1.0)
    y_true, y_pred = train_model(learner, sim_drugs, sim_targets, bindings, 'C')
    eval_model(y_true, y_pred, axis, 2)


    print("Scenario D - Split_pairs:")
    print("======================")
    # Out-of-sample leave-pair-out. On each CV round, a single (drug, target) pair is used as test pair (similar to setting A).
    # However, all pairs where either the drug or the target appears, are left out of the training set
    learner = TwoStepRLS(X1=sim_drugs.to_numpy(), X2=sim_targets.to_numpy(), Y=bindings.to_numpy(), regparam1=1.0, regparam2=1.0)
    y_true, y_pred = train_model(learner, sim_drugs, sim_targets, bindings, 'D')
    eval_model(y_true, y_pred, axis, 3)

    figure.tight_layout()
    plt.show()

    # Create heatmap of the bindings data
    plt.figure(figsize=(10, 5))
    sn.set()
    plt.title("Heatmap of Drug-Target pIC50 Values (nM)")
    heatmap(bindings)
    plt.show()


if __name__=="__main__":
    main()
