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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Perceptron
from scipy.stats import randint

# Show all columns:
# https://www.roelpeters.be/how-to-solve-pandas-not-showing-all-columns-head/
# pd.set_option('display.max_columns', None)

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

# Load normalised target-target similarity scores
sim_targets = pd.read_csv("data/target-target_similarities_WS_normalized.txt", sep=" ", header=None, names=id1)
sim_targets.index = id1
print("Target similarities matrix: ", sim_targets.shape)
# print(sim_targets.head())

# Load drug-drug similarity scores
sim_drugs = pd.read_csv("data/drug-drug_similarities_2D.txt", sep=" ", header=None, names=id2)
sim_drugs.index = id2
print("Drug similarities matrix: ", sim_drugs.shape)
# print(sim_drugs)

# Load drug-target interaction affinity scores
bindings = pd.read_csv("data/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt", sep=" ", header=None, names=id1)
bindings.index = id2
print("Bindings matrix: ", bindings.shape)
# print(bindings)
print("")


# Sort drug PubChem CIDs index and target gene names index
drug_pubchemIDs.sort_index(inplace=True)
target_gene_names.sort_index(inplace=True)

# sort bindings (sim_drugs : sim_targets) based on their index
bindings.sort_index(inplace=True)
sim_targets.sort_index(inplace=True)
sim_drugs.sort_index(inplace=True)

# transform binding values
sim_targets = sim_targets/100
transformed_bindings = -np.log10(bindings/(10**9))
print(transformed_bindings.head())
print("")


# Build drug_target_binding dataframe (i.e. 'Drug', 'Target', 'Binding_Val')
drug_target_binding = None
l=[]
for i in id1:
    for j in id2:
        k=transformed_bindings.loc[j,i]
        l2=[j,i,k]
        l.append(l2)
drug_target_binding=pd.DataFrame(l,columns=['Drug', 'Target', 'Binding_Val'])
print("Top drug-target bindings:")
print(drug_target_binding.head())
print("")


# split drug_target_binding dataframe to 3 sets (train_data, val_data, test_data)
train_data = None
val_data = None
test_data = None
train_data, x_remain = train_test_split(drug_target_binding, test_size=0.3)
print("Training Data Shape: ", train_data.head())
val_data, test_data = train_test_split(x_remain, test_size=0.3)
print("Train data shape", train_data.shape)
print("Validation data shape", val_data.shape)
print("Test data shape", test_data.shape)
print("")

# Average similarity in sim_targets matrix for each target and average binding value (in train_data).
# Adding these two feature as 'avg-sim', 'avg-binding' to target_gene_names DataFrame
for i in sim_targets:
    target_gene_names.loc[i,'t_avg-sim']=np.mean(sim_targets.loc[i,:])

target_bindingvals={}
for i in range(len(train_data)):
    x=train_data.iloc[i,:] #drug=x[0],target=x[1],binding_val=[2]

    if x[1] not in target_bindingvals:
        target_bindingvals[x[1]]=[x[2]]
    else:
        target_bindingvals[x[1]].append(x[2])
for i in target_bindingvals:
    target_gene_names.loc[i,'t_avg-binding']=np.mean(target_bindingvals[i])
print("Target Gene Names:")
print(target_gene_names.head())
print("")


# Average similarity in sim_drugs matrix for each drug and average binding value (in train_data).
# Adding these two feature as 'avg-sim', 'avg-binding' to drug_pubchemIDs DataFrame
for i in sim_drugs:
    drug_pubchemIDs.loc[i,'d_avg-sim']=np.mean(sim_drugs.loc[i,:])

drug_bindingvals={}
for i in range(len(train_data)):
    x=train_data.iloc[i,:] #drug=x[0],target=x[1],binding_val=[2]
    if x[0] not in drug_bindingvals:
        drug_bindingvals[x[0]]=[x[2]]
    else:
        drug_bindingvals[x[0]].append(x[2])
for i in drug_bindingvals:
    drug_pubchemIDs.loc[i,'d_avg-binding']=np.mean(drug_bindingvals[i])

print("Drug PubChem IDs:")
print(drug_pubchemIDs.head())
print("")


# 2.2 Drug/Target Similarity Networks
print(target_gene_names.loc[:,"t_avg-sim"].describe())
print("")

print(drug_pubchemIDs.loc[:,"d_avg-sim"].describe())
print("")

sim_drugs=sim_drugs.reindex(sorted(sim_drugs.columns),axis=1) #sorting columns
print("Drugs similarity:")
print(sim_drugs.head())
print("")

drug_sim_threshold = 0.6
adj=np.where(sim_drugs!=1 ,sim_drugs,0)
adj0=np.where(adj>drug_sim_threshold ,1,0)
drug_graph=igraph.Graph.Adjacency(adj0.tolist(),mode="undirected")
drug_graph.simplify(loops=True)
# igraph.plot(drug_graph, labels=True)


target_sim_threshold = 0.7
adj1=np.where(sim_targets!=1 ,sim_targets,0)
adj2=np.where(adj1>target_sim_threshold ,1,0)
#target_graph=igraph.Graph.Adjacency(adj2.tolist(),mode="undirected")
drug_graph.simplify(loops=False)
# #igraph.plot(target_graph, labels=True)

drug_sim_threshold = 0.6
target_sim_threshold = 0.6
drug_graph = igraph.Graph()
target_graph = igraph.Graph()

drug_graph.add_vertices(len(sim_drugs))
target_graph.add_vertices(len(sim_targets))

for i, drug_1 in enumerate(sim_drugs):
    for j, drug_2 in enumerate(sim_drugs):
        if (sim_drugs.loc[drug_1,drug_2] > drug_sim_threshold) and (drug_1!=drug_2) :
            drug_graph.add_edges([(i, j)])

for i, tar_1 in enumerate(sim_targets):
    if i%50==0:
        print(i)
    for j, tar_2 in enumerate(sim_targets):
        if (sim_targets.loc[tar_1,tar_2] > target_sim_threshold) and (tar_1!=tar_2):
            target_graph.add_edges([(i, j)])

# igraph.plot(drug_graph)
# igraph.plot(target_graph)


# # 2.2.2 Number of neighbors and it's PageRank score:
for vertex in target_graph.vs:
    target_gene_names.loc[sorted(id1)[vertex.index], 't_n_neighbors'] = target_graph.neighborhood_size(vertex,mindist=1)
    target_gene_names.loc[sorted(id1)[vertex.index], 't_page_rank'] = target_graph.pagerank(vertex)

for vertex in drug_graph.vs:
    drug_pubchemIDs.loc[sorted(id2)[vertex.index], 'd_n_neighbors'] = drug_graph.neighborhood_size(vertex,mindist=1)
    drug_pubchemIDs.loc[sorted(id2)[vertex.index], 'd_page_rank'] = drug_graph.pagerank(vertex)

print("Target Gene Names")
print(target_gene_names.head())

print("Drug Pubchem IDs:")
print(drug_pubchemIDs.head())
print("")


# 2.3 Non-negative Matrix Factorization
# Define latent_dim variable, after that use train_data to build bindings matrix
# your matrix will have some na values fill them with 5 (Which is lowest binding score).
latent_dim = 3
train_binding_matrix = None
train_binding_matrix=pd.DataFrame(columns=sorted(id1),index=sorted(id2))#empty dataframe

for i in range(len(train_data)):
    x=train_data.iloc[i,:] #drug=x[0],target=x[1],binding_val=x[2]
    train_binding_matrix.loc[x[0],x[1]]=x[2]

train_binding_matrix=train_binding_matrix.fillna(5)

print("Train Binding Matrix:")
print(train_binding_matrix.head())
print("")


# Use sklearn NMF and factor 'train_binding_matrix' to P and Q.
# add columns of P and Q as features to 'drug_pubchemIDs' and 'target_gene_names'
model = NMF(n_components=latent_dim, init='random', random_state=0)
P = model.fit_transform(train_binding_matrix)
Q = model.components_

for idx,x in enumerate(sorted(id2)):
    drug_pubchemIDs.loc[x,"d_features1"]=P[idx][0]
    drug_pubchemIDs.loc[x,"d_features2"]=P[idx][1]
    drug_pubchemIDs.loc[x,"d_features3"]=P[idx][2]
#drug_pubchemIDs

for idx,x in enumerate(sorted(id1)):
    target_gene_names.loc[x,"t_features1"]=Q.T[idx][0]
    target_gene_names.loc[x,"t_features2"]=Q.T[idx][1]
    target_gene_names.loc[x,"t_features3"]=Q.T[idx][2]
#target_gene_names

print("Target Gene Names:")
print(target_gene_names.head())
print(target_gene_names.shape)
print("")

print("Drug Pubchem IDs:")
print(drug_pubchemIDs.shape)
print(drug_pubchemIDs.head())
print("")

print("Drug Target Bindings:")
print(drug_target_binding)
print("")


# Replace drug pubchemID and Target gene name with extracted features.
# and build X (drug and target features) and Y (binding_values) for each dataset
rows_list=[]
for i in id1:      #id1=list of target names
    for j in id2:  #id2=list of drug names
        d_row=drug_pubchemIDs.loc[j,:].tolist()
        t_row=target_gene_names.loc[i,:].tolist()
        rows_list.append(t_row+d_row)

X=pd.DataFrame(rows_list,columns=['d_avg-sim' ,'d_avg-binding', 'd_n_neighbors' ,'d_page_rank','d_features1' ,'d_features2', 'd_features3','t_avg-sim' ,'t_avg-binding', 't_n_neighbors' ,'t_page_rank','t_features1' ,'t_features2', 't_features3'])
Y=drug_target_binding.loc[:,"Binding_Val"]

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - train_ratio)

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + val_ratio))

print("X_train.shape:", X_train.shape)
print("Y_train.shape:", Y_train.shape)
print("X_val.shape:", X_val.shape)
print("Y_val.shape:", Y_val.shape)
print("X_test.shape:", X_test.shape)
print("Y_test.shape:", Y_test.shape)
print("")

model = xgboost.XGBRegressor()

param_grid ={
        'learning_rate': np.arange(0.0005,0.3,0.0005),
        'n_estimators':[5,10,15,20,30,40,50,60,70,80,200],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12,14,16,20],
        'colsample_bytree': np.arange(0.1,1.0,0.01),
        'subsample': np.arange(0.01,1.0,0.01)}

grid_search = RandomizedSearchCV(model, param_grid,random_state=0,cv=5, n_iter=10)
grid_result = grid_search.fit(X_train,Y_train)

print("Best parameters are:")
print(grid_result.best_params_)
print("")

def plot_model_results(results):
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
    ax.legend()
    plt.ylabel('RMSE')
    plt.show()


# Tune hyperparameters values and train model using .fit with Train data
# you can use l1 and l2 regularization terms too
# after training calculate Root Mean Square Error (RMSE) for validation data
learning_rate =grid_result.best_params_['learning_rate']
n_estimators =grid_result.best_params_['n_estimators']
max_depth =grid_result.best_params_['max_depth']
colsample_bytree =grid_result.best_params_['colsample_bytree']
subsample = grid_result.best_params_['subsample']

model = xgboost.XGBRegressor(objective ='reg:linear', learning_rate = learning_rate,
                             colsample_bytree = colsample_bytree,
                             max_depth = max_depth,
                             subsample = subsample,
                             n_estimators = n_estimators,
                             eval_metric='rmse')

model.fit(X_train,Y_train, #eval_metric="rmse",
          eval_set=[(X_train, Y_train), (X_val, Y_val)],
          verbose=False)

validation_rmse = mean_squared_error(Y_val,model.predict(X_val))**0.5
print("Validation RMSE: %.3f" % validation_rmse)
print(plot_model_results(model.evals_result()))
print("")

xgboost.plot_importance(model);


# Calculate RMSE for Test data
# Then calculate accuracy, precision, recall, f1-score for X_test
# Then  confusion_matrix (from sklearn)
y_true=np.where(Y_test>7,1,0)
y_pred=np.where(model.predict(X_test)>7,1,0)

tn, fp, fn, tp =confusion_matrix(y_true,y_pred).ravel()
print(tn, fp, fn, tp)

test_rmse = mean_squared_error(Y_test,model.predict(X_test))**0.5
test_acc =(tp+tn)/(tp+tn+fn+fp)
test_precision = tp/(tp+fp)
test_recall = tp/(tp+fn)
test_f1 = (2*test_precision*test_recall)/(test_precision+test_recall)

print("Test RMSE: %.3f" % test_rmse)
print("Test Accuracy: %.3f" % test_acc)
print("Test Precision: %.3f" % test_precision)
print("Test Recall: %.3f" % test_recall)
print("Test F1-Score: %.3f" % test_f1)
print("")

y_training = np.where(Y_train > 7, 1, 0)
y_testing = np.where(Y_test > 7, 1, 0)

Tree_model = DecisionTreeClassifier()

param = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
         "max_features": randint(1, 10),
         "min_samples_leaf": randint(1, 10),
         "criterion": ["gini", "entropy"]}

grid_search = RandomizedSearchCV(Tree_model, param, random_state=0, cv=10, n_iter=40)
grid_result = grid_search.fit(X_train, y_training)

c = grid_result.best_params_
max_depth = c['max_depth']
max_features = c['max_features']
min_samples_leaf = c['min_samples_leaf']
criterion = c['criterion']
# min_samples_split=c['min_samples_split']

best_model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features
                                    , min_samples_leaf=min_samples_leaf, criterion=criterion)

best_model.fit(X_train, y_training)

predictions = best_model.predict(X_test)
Tn, Fp, Fn, Tp = confusion_matrix(y_testing, predictions).ravel()

testt_acc = (Tp + Tn) / (Tp + Tn + Fn + Fp)
testt_precision = Tp / (Tp + Fp)
testt_recall = Tp / (Tp + Fn)
testt_f1 = (2 * testt_precision * testt_recall) / (testt_precision + testt_recall)
print("Test Accuracy: %.3f" % testt_acc)
print("Test Precision: %.3f" % testt_precision)
print("Test Recall: %.3f" % testt_recall)
print("Test F1-Score: %.3f" % testt_f1)
print("")

best_model= RandomForestClassifier(n_estimators=90)
best_model.fit(X_train,y_training)

predictions = best_model.predict(X_test)
Tn, Fp, Fn, Tp =confusion_matrix(y_testing,predictions).ravel()

testt_acc =(Tp+Tn)/(Tp+Tn+Fn+Fp)
testt_precision = Tp/(Tp+Fp)
testt_recall = Tp/(Tp+Fn)
testt_f1 = (2*testt_precision*testt_recall)/(testt_precision+testt_recall)
print("Test Accuracy: %.3f" % testt_acc)
print("Test Precision: %.3f" % testt_precision)
print("Test Recall: %.3f" % testt_recall)
print("Test F1-Score: %.3f" % testt_f1)
print("")

best_model = Perceptron(random_state=0)

best_model.fit(X_train, y_training)

predictions = best_model.predict(X_test)
Tn, Fp, Fn, Tp = confusion_matrix(y_testing, predictions).ravel()

testt_acc = (Tp + Tn) / (Tp + Tn + Fn + Fp)
testt_precision = Tp / (Tp + Fp)
testt_recall = Tp / (Tp + Fn)
testt_f1 = (2 * testt_precision * testt_recall) / (testt_precision + testt_recall)
print("Test Accuracy: %.3f" % testt_acc)
print("Test Precision: %.3f" % testt_precision)
print("Test Recall: %.3f" % testt_recall)
print("Test F1-Score: %.3f" % testt_f1)
print("")
