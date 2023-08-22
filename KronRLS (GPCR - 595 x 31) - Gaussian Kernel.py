filepath = 'data/GPCR/595x31/'
train_pct = 0.7
conf_matrx_threshold = 6.5

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
from rlscore.learner import KronRLS
from rlscore.kernel import GaussianKernel
from rlscore.measure import cindex
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index
# from scipy.stats import pearsonr
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


# Scenario B - Split according to drugs
def split_B(sim_drugs, sim_targets, bindings):
    drug_split = int(round(sim_drugs.shape[0] * train_pct, 0))
    # target_split = int(round(sim_targets.shape[0] * train_pct, 0))

    # Select 40 random drugs and 300 random targets
    drug_ind = list(range(bindings.shape[0]))
    target_ind = list(range(bindings.shape[1]))
    np.random.seed(1)
    np.random.shuffle(drug_ind)
    np.random.shuffle(target_ind)
    select_drug_ind = drug_ind[:drug_split]
    remain_drug_ind = drug_ind[drug_split:]
    # select_target_ind = target_ind[:target_split]
    # remain_target_ind = target_ind[target_split:]

    bindings_train = bindings.iloc[select_drug_ind]
    bindings_test = bindings.iloc[remain_drug_ind]
    drugs_train = sim_drugs.iloc[select_drug_ind]
    drugs_test = sim_drugs.iloc[remain_drug_ind]
    targets_train = sim_targets
    targets_test = sim_targets

    return drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test


# Scenario C - Split according to targets
def split_C(sim_drugs, sim_targets, bindings):
    # drug_split = int(round(sim_drugs.shape[0] * train_pct, 0))
    target_split = int(round(sim_targets.shape[0] * train_pct, 0))

    # Select 40 random drugs and 300 random targets
    drug_ind = list(range(bindings.shape[0]))
    target_ind = list(range(bindings.shape[1]))
    np.random.seed(1)
    np.random.shuffle(drug_ind)
    np.random.shuffle(target_ind)

    # select_drug_ind = drug_ind[:drug_split]
    # remain_drug_ind = drug_ind[drug_split:]
    select_target_ind = target_ind[:target_split]
    remain_target_ind = target_ind[target_split:]

    bindings_train = bindings.iloc[:, select_target_ind]
    bindings_test = bindings.iloc[:, remain_target_ind]
    drugs_train = sim_drugs
    drugs_test = sim_drugs
    targets_train = sim_targets.iloc[select_target_ind]
    targets_test = sim_targets.iloc[remain_target_ind]

    return drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test


# Scenario D - Split so that drug-target pairs do not overlap between training and test set
def split_D(sim_drugs, sim_targets, bindings):
    drug_split = int(round(sim_drugs.shape[0] * train_pct, 0))
    target_split = int(round(sim_targets.shape[0] * train_pct, 0))

    # Select 40 random drugs and 300 random targets
    drug_ind = list(range(bindings.shape[0]))
    target_ind = list(range(bindings.shape[1]))
    np.random.seed(1)
    np.random.shuffle(drug_ind)
    np.random.shuffle(target_ind)
    select_drug_ind = drug_ind[:drug_split]
    remain_drug_ind = drug_ind[drug_split:]
    select_target_ind = target_ind[:target_split]
    remain_target_ind = target_ind[target_split:]

    bindings_train = bindings.iloc[select_drug_ind, select_target_ind]
    bindings_test = bindings.iloc[remain_drug_ind, remain_target_ind]
    drugs_train = sim_drugs.iloc[select_drug_ind]
    drugs_test = sim_drugs.iloc[remain_drug_ind]
    targets_train = sim_targets.iloc[select_target_ind]
    targets_test = sim_targets.iloc[remain_target_ind]

    return drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test


def train_model(learner, drugs_test, targets_test, bindings_test, scenario):
    bindings_test_flat = bindings_test.values.ravel(order='F')
    # best_regparam = None
    best_predict = None
    best_cindex = 0.0
    log_regparams = range(-20, 15)
    for log_regparam in log_regparams:
        learner.solve(2. ** log_regparam)
        if scenario == 'A':
            P = learner.in_sample_loo()
        else:
            P = learner.predict(drugs_test, targets_test)
        perf = cindex(bindings_test_flat, P)
        if perf>best_cindex:
            best_regparam=log_regparam
            best_cindex=perf
            best_predict = P
        # print("regparam 2**%d, cindex %f" % (log_regparam, perf))
    print("best regparam 2**%d with cindex %f" % (best_regparam, best_cindex))

    y_true = bindings_test_flat
    y_pred = best_predict

    labels = []
    for target in bindings_test.columns:
        for drug in bindings_test.index:
            labels.append(str(drug) + "|"+ target)

    test_data = pd.DataFrame(y_true, columns=['y_true'], index=labels)
    test_data['y_pred'] = y_pred
    test_data.to_csv(filepath + 'KronRLS/KronRLS_Gaussian_' + scenario + '_test.csv', sep=',')

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
    test_acc = (tp + tn) / (tp + tn + fn + fp)
    test_precision = tp / (tp + fp)
    test_recall = tp / (tp + fn)
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall)
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin)
    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_pred_bin)
    auprc = auc(recall, precision)

    print("True Negative:", tn, ", False Positive:", fp, ", False Negative:", fn, ", True Positive:", tp)
    print("Test Accuracy: %.3f" % test_acc)
    print("Test Precision: %.3f" % test_precision)
    print("Test Recall: %.3f" % test_recall)
    print("Test F1-Score: %.3f" % test_f1)
    print("ROC AUC: %.3f" % roc_auc)
    print("AUPRC: %.3f" % auprc)

    # Concordance Index Tutorial:
    # https://medium.com/analytics-vidhya/concordance-index-72298c11eac7
    test_cindex = concordance_index(y_true, y_pred)
    test_mse = mean_squared_error(y_true, y_pred)
    # Pearson Correlation Tutorial:
    # https://realpython.com/numpy-scipy-pandas-correlation-python/
    # test_pearson, ignore_var = pearsonr(y_true, y_pred)
    test_spearman, ignore_var = spearmanr(y_true, y_pred)
    # R2 Score Tutorial:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    test_r2 = r2_score(y_true, y_pred)

    print("Concordance Index: ", test_cindex)
    # print("Test MSE: %.3f" % test_mse)
    print("Test RMSE: %.3f" % test_mse ** 0.5)
    # print("Test Pearson Correlation: %.3f" % test_pearson)
    print("Test Spearman Correlation: %.3f" % test_spearman)
    print("Test R2 Score: %.3f" % test_r2)
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

    drug_split = int(round(sim_drugs.shape[0] * train_pct, 0))
    target_split = int(round(sim_targets.shape[0] * train_pct, 0))

    print("Training / test split used:")
    print("Drugs = " + str(drug_split) + ":" + str(sim_drugs.shape[0] - drug_split))
    print("Targets = " + str(target_split) + ":" , str(sim_targets.shape[0] - target_split))
    print("")

    print("Confusion Matrix Threshold:", conf_matrx_threshold)
    print("")

    print("Running Using Gaussian Kernels:")
    print("")

    # specifying the plot size
    figure, axis = plt.subplots(2, 2, figsize=(10, 6))

    print("Scenario A - Leave-one-out cross-validation predictions:")
    print("======================")
    # Scenario A - Data is not split (in-sample leave-one-out cross-validation predictions computed)
    kernel1 = GaussianKernel(sim_drugs, gamma=10**-1)
    kernel2 = GaussianKernel(sim_targets, gamma=10**-2)
    K1_train_test = kernel1.getKM(sim_drugs)
    K2_train_test = kernel2.getKM(sim_targets)
    learner = KronRLS(K1=K1_train_test, K2=K2_train_test, Y=bindings)
    y_true, y_pred = train_model(learner, K1_train_test, K2_train_test, bindings, 'A')
    eval_model(y_true, y_pred, axis, 0)


    print("Scenario B - Split_drugs:")
    print("======================")
    drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test_B = split_B(sim_drugs, sim_targets, bindings)
    kernel1 = GaussianKernel(drugs_train, gamma=10**-1)
    kernel2 = GaussianKernel(targets_train, gamma=10**-2)
    K1_train = kernel1.getKM(drugs_train)
    K1_test = kernel1.getKM(drugs_test)
    K2_train = kernel2.getKM(targets_train)
    K2_test = kernel2.getKM(targets_test)
    learner = KronRLS(K1=K1_train, K2=K2_train, Y=bindings_train)
    y_true, y_pred = train_model(learner, K1_test, K2_test, bindings_test_B, 'B')
    eval_model(y_true, y_pred, axis, 1)


    print("Scenario C - Split_targets:")
    print("======================")
    drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test_C = split_C(sim_drugs, sim_targets, bindings)
    kernel1 = GaussianKernel(drugs_train, gamma=10**-1)
    kernel2 = GaussianKernel(targets_train, gamma=10**-2)
    K1_train = kernel1.getKM(drugs_train)
    K1_test = kernel1.getKM(drugs_test)
    K2_train = kernel2.getKM(targets_train)
    K2_test = kernel2.getKM(targets_test)
    learner = KronRLS(K1=K1_train, K2=K2_train, Y=bindings_train)
    y_true, y_pred = train_model(learner, K1_test, K2_test, bindings_test_C, 'C')
    eval_model(y_true, y_pred, axis, 2)


    print("Scenario D - Split_pairs:")
    print("======================")
    drugs_train, drugs_test, targets_train, targets_test, bindings_train, bindings_test_D = split_D(sim_drugs, sim_targets, bindings)
    kernel1 = GaussianKernel(drugs_train, gamma=10**-1)
    kernel2 = GaussianKernel(targets_train, gamma=10**-2)
    K1_train = kernel1.getKM(drugs_train)
    K1_test = kernel1.getKM(drugs_test)
    K2_train = kernel2.getKM(targets_train)
    K2_test = kernel2.getKM(targets_test)
    learner = KronRLS(K1=K1_train, K2=K2_train, Y=bindings_train)
    y_true, y_pred = train_model(learner, K1_test, K2_test, bindings_test_D, 'D')
    eval_model(y_true, y_pred, axis, 3)

    plt.show()

    figure.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sn.set()

    plt.title("Scenario A")
    heatmap(bindings)
    plt.show()

    plt.title("Scenario B")
    heatmap(bindings_test_B)
    plt.show()

    plt.title("Scenario C")
    heatmap(bindings_test_C)
    plt.show()

    plt.title("Scenario D")
    heatmap(bindings_test_D)
    plt.show()


if __name__=="__main__":
    main()
