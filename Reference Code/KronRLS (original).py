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


# Troubleshooting:
# =======================
# Reason for error when reading data file in module that has been imported:
# https://stackoverflow.com/questions/61289041/python-import-module-from-directory-error-reading-file

import numpy as np
from rlscore.learner import KronRLS
from rlscore.measure import cindex

def load_data():
    Y = np.loadtxt("data/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt")
    XD = np.loadtxt("data/drug-drug_similarities_2D.txt")
    XT = np.loadtxt("data/target-target_similarities_WS.txt")
    # SimBoost file uses normalised target-target similarity values:
    # XT = np.loadtxt("data/target-target_similarities_WS_normalized.txt")
    return XD, XT, Y

# Split according to drugs (Setting B)
def split_drugs():
    np.random.seed(1)
    XD, XT, Y = load_data()
    drug_ind = list(range(Y.shape[0]))
    # target_ind = list(range(Y.shape[1]))
    np.random.shuffle(drug_ind)
    train_drug_ind = drug_ind[:40]
    test_drug_ind = drug_ind[40:]
    Y_train = Y[train_drug_ind]
    Y_test = Y[test_drug_ind]
    Y_train = Y_train.ravel(order='F')
    Y_test = Y_test.ravel(order='F')
    XD_train = XD[train_drug_ind]
    XT_train = XT
    XD_test = XD[test_drug_ind]
    XT_test = XT
    return XD_train, XT_train, Y_train, XD_test, XT_test, Y_test

# Split according to targets (Setting C)
def split_targets():
    np.random.seed(1)
    XD, XT, Y = load_data()
    # drug_ind = list(range(Y.shape[0]))
    target_ind = list(range(Y.shape[1]))
    np.random.shuffle(target_ind)
    train_target_ind = target_ind[:300]
    test_target_ind = target_ind[300:]
    Y_train = Y[:, train_target_ind]
    Y_test = Y[:, test_target_ind]
    Y_train = Y_train.ravel(order='F')
    Y_test = Y_test.ravel(order='F')
    XD_train = XD
    XT_train = XT[train_target_ind]
    XD_test = XD
    XT_test = XT[test_target_ind]
    return XD_train, XT_train, Y_train, XD_test, XT_test, Y_test

# Split so that d,t pairs do not overlap between training and test set (Setting D)
def split_pairs():
    np.random.seed(1)
    XD, XT, Y = load_data()
    drug_ind = list(range(Y.shape[0]))
    target_ind = list(range(Y.shape[1]))
    np.random.shuffle(drug_ind)
    np.random.shuffle(target_ind)
    train_drug_ind = drug_ind[:40]
    test_drug_ind = drug_ind[40:]
    train_target_ind = target_ind[:300]
    test_target_ind = target_ind[300:]
    Y_train = Y[np.ix_(train_drug_ind, train_target_ind)]
    Y_test = Y[np.ix_(test_drug_ind, test_target_ind)]
    # Ravel flattens the matrix into a row with ordering based on columns
    Y_train = Y_train.ravel(order='F')
    Y_test = Y_test.ravel(order='F')
    XD_train = XD[train_drug_ind]
    XT_train = XT[train_target_ind]
    XD_test = XD[test_drug_ind]
    XT_test = XT[test_target_ind]
    return XD_train, XT_train, Y_train, XD_test, XT_test, Y_test


def main():
    XD, XT, Y = load_data()
    print("Y dimensions %d %d" %Y.shape)
    print("XD dimensions %d %d" %XD.shape)
    print("XT dimensions %d %d" %XT.shape)
    print("drug-target pairs: %d" %(Y.shape[0]*Y.shape[1]))
    print("")

    # https://github.com/aatapa/RLScore/blob/master/docs/src/kron_rls1.py
    print("Scenario A - Imputing missing values inside the Y-matrix:")
    Y = Y.ravel(order='F')

    learner = KronRLS(X1=XD, X2=XT, Y=Y)
    log_regparams = range(15, 35)
    for log_regparam in log_regparams:
        learner.solve(2. ** log_regparam)
        # http://staff.cs.utu.fi/~aatapa/software/RLScore/modules/two_step_rls.html
        # Computes the in-sample leave-one-out cross-validation prediction
        P = learner.in_sample_loo()
        perf = cindex(Y, P)
        print("regparam 2**%d, cindex %f" % (log_regparam, perf))
    print("")


    # https://github.com/aatapa/RLScore/blob/master/docs/src/kron_rls2.py
    print("Scenario B - Split_drugs:")
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = split_drugs()
    learner = KronRLS(X1=X1_train, X2=X2_train, Y=Y_train)
    log_regparams = range(15, 35)
    for log_regparam in log_regparams:
        learner.solve(2. ** log_regparam)
        P = learner.predict(X1_test, X2_test)
        perf = cindex(Y_test, P)
        print("regparam 2**%d, cindex %f" % (log_regparam, perf))
    print("")

    #https://github.com/aatapa/RLScore/blob/master/docs/src/kron_rls3.py
    print("Scenario C - Split_targets:")
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = split_targets()
    learner = KronRLS(X1=X1_train, X2=X2_train, Y=Y_train)
    log_regparams = range(15, 35)
    for log_regparam in log_regparams:
        learner.solve(2. ** log_regparam)
        P = learner.predict(X1_test, X2_test)
        perf = cindex(Y_test, P)
        print("regparam 2**%d, cindex %f" % (log_regparam, perf))
    print("")

    #https://github.com/aatapa/RLScore/blob/master/docs/src/kron_rls4.py
    print("Scenario D - Split_pairs:")
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = split_pairs()
    learner = KronRLS(X1=X1_train, X2=X2_train, Y=Y_train)
    log_regparams = range(15, 35)
    for log_regparam in log_regparams:
        learner.solve(2. ** log_regparam)
        P = learner.predict(X1_test, X2_test)
        perf = cindex(Y_test, P)
        print("regparam 2**%d, cindex %f" % (log_regparam, perf))
    print("")

if __name__=="__main__":
    main()

