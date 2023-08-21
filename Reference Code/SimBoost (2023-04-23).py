# Reference Code:
# https://github.com/mahtaz/Simboost-ML_project-

# Extract from following article explains what SimBoost is:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5395521/#CR4
#
# SimBoost, constructs features for each drug, each target and each drug–target pair.
# These features represent the properties of drugs, targets and drug–target pairs, respectively.
# SimBoost associates a feature vector with each pair of one drug and one target.
# From pairs with observed binding affinities, it trains a gradient boosting machine model
# to learn the nonlinear relationships between the features and the binding affinities.
# Once the model is trained, SimBoost can make predictions of the binding affinities for
# unobserved drug–target pairs, based on their known features.

import numpy as np
import pandas as pd
import igraph
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

# Load the raw data (i.e. drud-drug, target-target, and drug-target similarity scores)
def load_data():
    # Load normalised target-target similarity scores and set the index for rows and columns with the gene names
    target_gene_names = pd.read_csv("../data/Davis/target_gene_names.txt", header=None, index_col=0)
    # sim_targets = pd.read_csv("data/target-target_similarities_WS.txt", sep=" ", header=None, names=target_gene_names.index)
    sim_targets = pd.read_csv("../data/Davis/target-target_similarities_WS_normalized.txt", sep=" ", header=None, names=target_gene_names.index)
    sim_targets.index = target_gene_names.index

    # Load drug-drug similarity scores and set the index for rows and columns with the drug names
    drug_pubchemIDs = pd.read_csv("../data/Davis/drug_PubChem_CIDs.txt", header=None, index_col=0)
    sim_drugs = pd.read_csv("../data/Davis/drug-drug_similarities_2D.txt", sep=" ", header=None, names=drug_pubchemIDs.index)
    sim_drugs.index = drug_pubchemIDs.index

    # Load drug-target interaction affinity scores and set the row index with  drug names and column index with gene names
    bindings = pd.read_csv("../data/Davis/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt", sep=" ", header=None, names=target_gene_names.index)
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


def matrix_factorization(target_gene_names, drug_pubchemIDs, sim_drugs, sim_targets, bindings):
    # Build drug_target_binding dataframe (i.e. 'Drug', 'Target', 'Binding_Val')
    l = []
    for i in target_gene_names.index:
        for j in drug_pubchemIDs.index:
            k = bindings.loc[j, i]
            l2 = [j, i, k]
            l.append(l2)
    drug_target_binding = pd.DataFrame(l, columns=['Drug', 'Target', 'Binding_Val'])

    # Add average drug similarity score (d_avg-sim) as feature to 'drug_pubchemIDs' DataFrame
    for i in sim_drugs:
        drug_pubchemIDs.loc[i, 'd_avg-sim'] = np.mean(sim_drugs.loc[i, :])

    # Add average target similarity score (t_avg-sim) as feature to 'target_gene_names' DataFrame
    for i in sim_targets:
        target_gene_names.loc[i, 't_avg-sim'] = np.mean(sim_targets.loc[i, :])

    # Add average binding value (t_avg-binding) as feature to 'target_gene_names' DataFrame
    target_bindingvals = {}
    for i in range(len(drug_target_binding)):
        x = drug_target_binding.iloc[i, :]  # drug=x[0],target=x[1],binding_val=[2]
        if x[1] not in target_bindingvals:
            target_bindingvals[x[1]] = [x[2]]
        else:
            target_bindingvals[x[1]].append(x[2])
    for i in target_bindingvals:
        target_gene_names.loc[i, 't_avg-binding'] = np.mean(target_bindingvals[i])

    # Add average binding value (d_avg-binding) as feature to 'drug_pubchemIDs' DataFrame
    drug_bindingvals = {}
    for i in range(len(drug_target_binding)):
        x = drug_target_binding.iloc[i, :]
        if x[0] not in drug_bindingvals:
            drug_bindingvals[x[0]] = [x[2]]
        else:
            drug_bindingvals[x[0]].append(x[2])
    for i in drug_bindingvals:
        drug_pubchemIDs.loc[i, 'd_avg-binding'] = np.mean(drug_bindingvals[i])

    # Generate drug_graph with a node for each drug
    drug_graph = igraph.Graph()
    drug_graph.add_vertices(len(sim_drugs))

    # Generate target_graph with a node for each target
    target_graph = igraph.Graph()
    target_graph.add_vertices(len(sim_targets))

    # Create edges between nodes where the similarity score is over the threshold
    drug_sim_threshold = 0.6
    target_sim_threshold = 0.6

    for i, drug_1 in enumerate(sim_drugs):
        for j, drug_2 in enumerate(sim_drugs):
            if (sim_drugs.loc[drug_1, drug_2] > drug_sim_threshold) and (drug_1 != drug_2):
                drug_graph.add_edges([(i, j)])

    for i, tar_1 in enumerate(sim_targets):
        for j, tar_2 in enumerate(sim_targets):
            if (sim_targets.loc[tar_1, tar_2] > target_sim_threshold) and (tar_1 != tar_2):
                target_graph.add_edges([(i, j)])

    # Number of neighbors and it's PageRank score:
    for vertex in target_graph.vs:
        target_gene_names.loc[target_gene_names.index[vertex.index], 't_n_neighbors'] = target_graph.neighborhood_size(vertex, mindist=1)
        target_gene_names.loc[target_gene_names.index[vertex.index], 't_page_rank'] = target_graph.pagerank(vertex)

    for vertex in drug_graph.vs:
        drug_pubchemIDs.loc[drug_pubchemIDs.index[vertex.index], 'd_n_neighbors'] = drug_graph.neighborhood_size(vertex, mindist=1)
        drug_pubchemIDs.loc[drug_pubchemIDs.index[vertex.index], 'd_page_rank'] = drug_graph.pagerank(vertex)

    # Non-negative Matrix Factorization
    # Define latent_dim variable, after that use drug_target_binding to build bindings matrix
    # your matrix will have some na values fill them with 5 (Which is lowest binding score).
    latent_dim=3
    train_binding_matrix = pd.DataFrame(columns=target_gene_names.index, index=drug_pubchemIDs.index)

    for i in range(len(drug_target_binding)):
        x = drug_target_binding.iloc[i, :]
        train_binding_matrix.loc[x[0], x[1]] = x[2]

    train_binding_matrix = train_binding_matrix.fillna(5)

    # Use sklearn NMF and factor 'train_binding_matrix' to P and Q.
    # add columns of P and Q as features to 'drug_pubchemIDs' and 'target_gene_names'

    # Set random number seed to ensure result is reproducible
    np.random.seed(1)
    model = NMF(n_components=latent_dim, init='random', random_state=0, max_iter=1000)
    P = model.fit_transform(train_binding_matrix)
    Q = model.components_

    for idx, x in enumerate(drug_pubchemIDs.index):
        drug_pubchemIDs.loc[x, "d_features1"] = P[idx][0]
        drug_pubchemIDs.loc[x, "d_features2"] = P[idx][1]
        drug_pubchemIDs.loc[x, "d_features3"] = P[idx][2]

    for idx, x in enumerate(target_gene_names.index):
        target_gene_names.loc[x, "t_features1"] = Q.T[idx][0]
        target_gene_names.loc[x, "t_features2"] = Q.T[idx][1]
        target_gene_names.loc[x, "t_features3"] = Q.T[idx][2]

    # Replace drug pubchemID and Target gene name with extracted features.
    # and build X (drug and target features) and Y (binding_values) for each dataset
    rows_list = []
    for i in target_gene_names.index:
        for j in drug_pubchemIDs.index:
            d_row = drug_pubchemIDs.loc[j, :].tolist()
            t_row = target_gene_names.loc[i, :].tolist()
            rows_list.append(t_row + d_row)

    X = pd.DataFrame(rows_list, columns=['d_avg-sim', 'd_avg-binding', 'd_n_neighbors', 'd_page_rank', 'd_features1',
                                         'd_features2', 'd_features3', 't_avg-sim', 't_avg-binding', 't_n_neighbors',
                                         't_page_rank', 't_features1', 't_features2', 't_features3'])
    Y = drug_target_binding.loc[:, "Binding_Val"]

    return X, Y


def train_model(X, Y):
    # Split data: Training = 70%, Test = 30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    model = xgboost.XGBRegressor()
    param_grid = {
        'learning_rate': np.arange(0.0005, 0.3, 0.0005),
        'n_estimators': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 200],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20],
        'colsample_bytree': np.arange(0.1, 1.0, 0.01),
        'subsample': np.arange(0.01, 1.0, 0.01)}

    # Apply k-fold = 5 cross-validation on training data
    # https://www.projectpro.io/recipes/find-optimal-parameters-using-randomizedsearchcv-for-regression
    grid_search = RandomizedSearchCV(model, param_grid, random_state=0, cv=5, n_iter=10)
    grid_result = grid_search.fit(X_train, Y_train)

    # print("Best parameters are:")
    # print(grid_result.best_params_)
    # print("")


    # Tune hyperparameters values and train model using .fit with Train data you can use l1 and l2 regularization
    # terms too after training calculate Root Mean Square Error (RMSE) for validation data
    learning_rate = grid_result.best_params_['learning_rate']
    n_estimators = grid_result.best_params_['n_estimators']
    max_depth = grid_result.best_params_['max_depth']
    colsample_bytree = grid_result.best_params_['colsample_bytree']
    subsample = grid_result.best_params_['subsample']

    model = xgboost.XGBRegressor(objective='reg:squarederror',
                                 learning_rate=learning_rate,
                                 colsample_bytree=colsample_bytree,
                                 max_depth=max_depth,
                                 subsample=subsample,
                                 n_estimators=n_estimators,
                                 eval_metric='rmse')

    model.fit(X_train, Y_train)
    xgboost.plot_importance(model)

    y_true = np.where(Y_test > 7, 1, 0)
    y_pred = np.where(model.predict(X_test) > 7, 1, 0)

    return y_true, y_pred


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


def main():
    # Show all columns:
    # https://www.roelpeters.be/how-to-solve-pandas-not-showing-all-columns-head/
    # pd.set_option('display.max_columns', None)

    # Load the raw data
    target_gene_names, drug_pubchemIDs, sim_drugs, sim_targets, bindings = load_data()

    # Prepare the data and apply matrix factorisation
    X, Y = matrix_factorization(target_gene_names, drug_pubchemIDs, sim_drugs, sim_targets, bindings)

    # Train model using k-fold cross-validation
    y_true, y_pred = train_model(X, Y)

    # Evaluate how well the model is performing
    eval_model(y_true, y_pred)


if __name__=="__main__":
    main()
