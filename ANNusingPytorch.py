import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scipy as sp
import numpy as np
# Loompy is only needed if using loom files
# import loompy
import anndata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


sc.settings.verbosity = 3
sc.logging.print_header()

import pySingleCellNet as pySCN

adTrain = sc.read("adLung_TabSen_100920.h5ad")
adTrain
# AnnData object with n_obs × n_vars = 14813 × 21969 ...

qDatT = sc.read_mtx("GSE124872_raw_counts_single_cell.mtx")
qDat = qDatT.T

genes = pd.read_csv("genes.csv")
qDat.var_names = genes.x

qMeta = pd.read_csv("GSE124872_Angelidis_2018_metadata.csv")
qMeta.columns.values[0] = "cellid"

qMeta.index = qMeta["cellid"]
qDat.obs = qMeta.copy()

# If your expression data is stored as a numpy array, convert it
# type(qDat.X)
# <class 'numpy.ndarray'>
# pySCN.check_adX(qDat)

genesTrain = adTrain.var_names
genesQuery = qDat.var_names

cgenes = genesTrain.intersection(genesQuery)
len(cgenes)
# 16543

adTrain1 = adTrain[:,cgenes]
adQuery = qDat[:,cgenes].copy()
adQuery = adQuery[adQuery.obs["nGene"]>=500,:].copy()
adQuery
# AnnData object with n_obs × n_vars = 4240 × 16543

expTrain, expVal = pySCN.splitCommonAnnData(adTrain1, ncells=200,dLevel="cell_ontology_class")

[cgenesA, xpairs, tspRF, X, y] = pySCN.scn_train(expTrain, nTopGenes = 100, nRand = 100, nTrees = 1000 ,nTopGenePairs = 100, dLevel = "cell_ontology_class", stratify=True, limitToHVG=True)

# H.Chen's work

# consider RF used in original code as benchmark
accuracy = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=1000, random_state=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
avg_accuracy = np.array(accuracy).mean()
print(f'Accuracy of random forest is {avg_accuracy * 100}%') # 97.35%

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
def get_y_incatval(y):
    """
    Input: labels in string
    Returns: labels in categorical number
    """
    label_names = []
    for name in y:
        if not name in label_names:
            label_names.append(name)
    num_class = len(label_names)
    label_catval = range(num_class)
    y_incatval = []
    for name in y:
        idx = label_names.index(name)
        catval2map = label_catval[idx]
        y_incatval.append(catval2map)
    y_incatval = np.array(y_incatval)
    return y_incatval, num_class, label_names

y_incatval, num_class, label_names = get_y_incatval(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_incatval, test_size=0.2)
X_train, X_test = torch.from_numpy(X_train).float(), torch.from_numpy(X_test).float() # data in tensor
y_train, y_test = torch.unsqueeze(torch.from_numpy(y_train), 1),  torch.unsqueeze(torch.from_numpy(y_test), 1) # data in tensor
train_data = torch.cat((X_train, y_train), dim=1)
test_data = torch.cat((X_test, y_test), dim=1)

# np.save('train_data.npy', train_data)
# np.save('test_data.npy', test_data)

input_size = X_train.shape[1] # 979
hidden_size = 200
output_size = num_class # 11

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork().to(device)
print(model)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, sample in enumerate(dataloader):
        # transform y_test in categorical value into one-hot encoding
        X, y = sample[:, :-1], sample[:, -1].long()
        # Forward pass, Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for sample in dataloader:
            X, y = sample[:, :-1], sample[:, -1].long()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # to obtain the number of correct prediction,
            # use pred.argmax(dim=1) to find the index of maximal value ---> prediction in categorical value
            # i.e. we don't need to transform y_test into one-hot encoding
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, model

# hyperparameters
learning_rate = 1e-4
batch_size = 256
epochs = 120

largest_correct = 0
correct_list = []
print('start')
for i in range(150):
    # create iterator to load data, every time load the number of "batch_size" data
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False)

    # select loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        correct, model = test_loop(test_dataloader, model, loss_fn)
    correct_list.append(correct)
    if correct >= largest_correct:
        largest_correct = correct
        optimal_model = model
    print("Done!")
print(f'Largest Accuracy of ANN is {largest_correct*100}%\n')
print(f'Avg Accuracy of ANN is {np.array(correct_list).mean()*100}%') # 97.94%

#torch.save(optimal_model.state_dict(), 'ANNmodel_weights.pth')




