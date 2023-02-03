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

adVal, X_test = pySCN.scn_classify(expVal, cgenesA, xpairs, tspRF, nrand = 0)

ax = sc.pl.heatmap(adVal, adVal.var_names.values, groupby='SCN_class', cmap='viridis', dendrogram=False, swap_axes=True)

assessment =  pySCN.assess_comm(expTrain, adVal, resolution = 0.005, nRand = 0, dLevelSID = "cell", classTrain = "cell_ontology_class", classQuery = "cell_ontology_class")

pySCN.plot_PRs(assessment)
plt.show()

adQlung = pySCN.scn_classify(adQuery, cgenesA, xpairs, tspRF, nrand = 0)

ax = sc.pl.heatmap(adQlung, adQlung.var_names.values, groupby='SCN_class', cmap='viridis', dendrogram=False, swap_axes=True)

ax = sc.pl.heatmap(adQlung, adQlung.var_names.values, groupby='celltype', cmap='viridis', dendrogram=False, swap_axes=True)

pySCN.add_classRes(adQuery, adQlung)

adM1Norm = adQuery.copy()
sc.pp.filter_genes(adM1Norm, min_cells=5)
sc.pp.normalize_per_cell(adM1Norm, counts_per_cell_after=1e4)
sc.pp.log1p(adM1Norm)

sc.pp.highly_variable_genes(adM1Norm, min_mean=0.0125, max_mean=4, min_disp=0.5)

adM1Norm.raw = adM1Norm
sc.pp.scale(adM1Norm, max_value=10)
sc.tl.pca(adM1Norm, n_comps=100)

sc.pl.pca_variance_ratio(adM1Norm, 100)

npcs = 20
sc.pp.neighbors(adM1Norm, n_neighbors=10, n_pcs=npcs)
sc.tl.leiden(adM1Norm,.1)
sc.tl.umap(adM1Norm, .5)
sc.pl.umap(adM1Norm, color=["leiden", "SCN_class"], alpha=.9, s=15, legend_loc='on data')

adQlung.obs['leiden'] = adM1Norm.obs['leiden'].copy()
adQlung.uns['leiden_colors'] = adM1Norm.uns['leiden_colors']
ax = sc.pl.heatmap(adQlung, adQlung.var_names.values, groupby='leiden', cmap='viridis', dendrogram=False, swap_axes=True)

sc.pl.umap(adM1Norm, color=["epithelial cell", "stromal cell", "B cell"], alpha=.9, s=15, legend_loc='on data', wspace=.3)

oTab = pd.read_csv("oTab.csv")
[adQuery,adTrain] = pySCN.csRenameOrth(adQuery, adTrain, oTab)



