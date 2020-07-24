import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import copy
import csv
import random
import seaborn as sns 
import math
import statsmodels.stats.multitest as multi


from sklearn.feature_selection import VarianceThreshold
from lifelines.utils import datetimes_to_durations,  survival_table_from_events
from scipy.stats import ranksums
from scipy import interp
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from sklearn.manifold import TSNE
from pymrmre import mrmr



def tsne_cluster(df):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=10000)
    embedded = tsne.fit_transform(df)
    xVal = embedded[:,0]
    yVal = embedded[:,1]
    return xVal, yVal


def variance_threshold_selector(data, threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]




def survival_info_adding(df: pd.DataFrame):
    df['Last Event'] = df['Last FU']
    df['Event'] = (df['Status'] == 'Dead')
    df.loc[df['Event'] , 'Last Event'] = df.loc[df['Event'], 'Date of Death']
    start_date = df['RT End']
    end_date = df['Last Event']
    T_old, _ = datetimes_to_durations(start_date, end_date)
    df['Survival Time'] = T_old /365
    df['High_Risk'] = df['Survival Time']<= 4
    return df

def manufacturer_splitting(dataframe):
    
    dataframe.replace({'Manufacturer': {'GE': 0, 'TOSHIBA': 1, 'PHILIPS': np.nan}}, inplace = True)
    dataframe.dropna(subset=['Manufacturer'], axis = 0, inplace = True)

    g = dataframe[(dataframe.Manufacturer == 0)]
    t = dataframe[(dataframe.Manufacturer == 1)]

    t.drop(['Manufacturer'], axis = 1, inplace = True)
    g.drop(['Manufacturer'], axis = 1, inplace = True)
    
    return t,g


def RankSumTest(x1,x2):
    
    feats = []
  
    for feat in x1.columns:
        if feat in x2.columns:
            g = np.asarray(x1[feat]).astype(np.float)
            t = np.asarray(x2[feat]).astype(np.float)
            p_val = ranksums(g,t)[1]
            if  p_val > 0.05:
                    feats.append(feat)          
    return(feats)



def RankSumTest1(x1,x2):
    
    pvals = []
    feats  =[]
  
    for feat in x1.columns:
        if feat in x2.columns:
            g = np.asarray(x1[feat]).astype(np.float)
            t = np.asarray(x2[feat]).astype(np.float)
            pvals.append(ranksums(g,t)[1])
            feats.append(feat)
    
    pval = multi.multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
    
    feats.append(feat)
    pvals.append(pval[1])
    return (feats,pval)


def RankSumTest2(x1,x2):
    
    result_table = pd.DataFrame(columns=['feat','pval'])
  
    for feat in x1.columns:
        if feat in x2.columns:
            g = np.asarray(x1[feat]).astype(np.float)
            t = np.asarray(x2[feat]).astype(np.float)
            p_val = ranksums(g,t)[1]
            result_table = result_table.append({'feat':feat,
                                                'pval':p_val}, ignore_index=True)          
    return(result_table)



def mrmr_feat(data,solution_length):
    solutions = mrmr.mrmr_ensemble(features=data, 
              target_features=[data.shape[1]-1], 
              feature_types=list(np.zeros(len(data.columns))), solution_length=solution_length)
    mrmr_ = solutions[0][0]
    return mrmr_



def feature_class (Features):
     
    shape = [feature for feature in Features if feature.split('_')[1] == 'shape' ]
    
    firstorder =             [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('original')]
    firstorder_wavelet =     [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('wavelet')]
    firstorder_exponential = [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('exponential')]
    firstorder_logarithm = [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('logarithm')]
    firstorder_gradient =    [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('gradient')]
    firstorder_lbp_2D =      [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('lbp-2D')]
    firstorder_lbp_3D_m1 =   [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('lbp-3D-m1')]
    firstorder_lbp_3D_m2 =   [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('lbp-3D-m2')]
    firstorder_lbp_3D_k =    [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('lbp-3D-k')]
    firstorder_square =      [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('square')]
    firstorder_squareroot =  [feature for feature in Features if feature.split('_')[1] == 'firstorder' and feature.split('_')[0].startswith('squareroot')]
    
    glcm =             [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('original')]
    glcm_wavelet =     [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('wavelet')]
    glcm_exponential = [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('exponential')]
    glcm_logarithm = [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('logarithm')]
    glcm_gradient =    [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('gradient')]
    glcm_lbp_2D =      [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('lbp-2D')]
    glcm_lbp_3D_m1 =   [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('lbp-3D-m1')]
    glcm_lbp_3D_m2 =   [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('lbp-3D-m2')]
    glcm_lbp_3D_k =    [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('lbp-3D-k')]
    glcm_square =      [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('square')]
    glcm_squareroot =  [feature for feature in Features if feature.split('_')[1] == 'glcm' and feature.split('_')[0].startswith('squareroot')]
    
    gldm =             [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('original')]
    gldm_wavelet =     [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('wavelet')]
    gldm_exponential = [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('exponential')]
    gldm_logarithm = [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('logarithm')]
    gldm_gradient =    [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('gradient')]
    gldm_lbp_2D =      [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('lbp-2D')]
    gldm_lbp_3D_m1 =   [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('lbp-3D-m1')]
    gldm_lbp_3D_m2 =   [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('lbp-3D-m2')]
    gldm_lbp_3D_k =    [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('lbp-3D-k')]
    gldm_square =      [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('square')]
    gldm_squareroot =  [feature for feature in Features if feature.split('_')[1] == 'gldm' and feature.split('_')[0].startswith('squareroot')]
    
    glrlm =             [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('original')]
    glrlm_wavelet =     [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('wavelet')]
    glrlm_exponential = [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('exponential')]
    glrlm_logarithm = [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('logarithm')]
    glrlm_gradient =    [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('gradient')]
    glrlm_lbp_2D =     [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('lbp-2D')]
    glrlm_lbp_3D_m1 =   [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('lbp-3D-m1')]
    glrlm_lbp_3D_m2 =   [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('lbp-3D-m2')]
    glrlm_lbp_3D_k =    [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('lbp-3D-k')]
    glrlm_square =      [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('square')]
    glrlm_squareroot =  [feature for feature in Features if feature.split('_')[1] == 'glrlm' and feature.split('_')[0].startswith('squareroot')]
    
    glszm =             [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('original')]
    glszm_wavelet =     [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('wavelet')]
    glszm_exponential = [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('exponential')]
    glszm_logarithm = [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('logarithm')]
    glszm_gradient =    [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('gradient')]
    glszm_lbp_2D =      [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('lbp-2D')]
    glszm_lbp_3D_m1 =   [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('lbp-3D-m1')]
    glszm_lbp_3D_m2 =   [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('lbp-3D-m2')]
    glszm_lbp_3D_k =    [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('lbp-3D-k')]
    glszm_square =      [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('square')]
    glszm_squareroot =  [feature for feature in Features if feature.split('_')[1] == 'glszm' and feature.split('_')[0].startswith('squareroot')]
    
    ngtdm =             [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('original')]
    ngtdm_wavelet =     [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('wavelet')]
    ngtdm_exponential = [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('exponential')]
    ngtdm_logarithm = [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('logarithm')]
    ngtdm_gradient =    [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('gradient')]
    ngtdm_lbp_2D =      [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('lbp-2D')]
    ngtdm_lbp_3D_m1 =   [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('lbp-3D-m1')]
    ngtdm_lbp_3D_m2 =   [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('lbp-3D-m2')]
    ngtdm_lbp_3D_k =    [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('lbp-3D-k')]
    ngtdm_square =      [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('square')]
    ngtdm_squareroot =  [feature for feature in Features if feature.split('_')[1] == 'ngtdm' and feature.split('_')[0].startswith('squareroot')]
    
   
    Classes = {'shape' : shape, 
               'firstorder' : firstorder,'firstorder_wavelet' : firstorder_wavelet,'firstorder_exponential': firstorder_exponential,'firstorder_logarithm':firstorder_logarithm,
               'firstorder_gradient': firstorder_gradient,'firstorder_lbp_2D' : firstorder_lbp_2D,'firstorder_lbp_3D_m1': firstorder_lbp_3D_m1,'firstorder_lbp_3D_m2': firstorder_lbp_3D_m2,
               'firstorder_lbp_3D_k': firstorder_lbp_3D_k,'firstorder_square': firstorder_square,'firstorder_squareroot': firstorder_squareroot,
               
               'glcm' : glcm,'glcm_wavelet' : glcm_wavelet,'glcm_exponential': glcm_exponential,'glcm_logarithm':glcm_logarithm,
               'glcm_gradient': glcm_gradient,'glcm_lbp_2D' : glcm_lbp_2D,'glcm_lbp_3D_m1': glcm_lbp_3D_m1,'glcm_lbp_3D_m2': glcm_lbp_3D_m2,
               'glcm_lbp_3D_k': glcm_lbp_3D_k,'glcm_square': glcm_square,'glcm_squareroot': glcm_squareroot,
                        
               'gldm' : gldm, 'gldm_wavelet' : gldm_wavelet,'gldm_exponential': gldm_exponential,'gldm_logarithm':gldm_logarithm,
               'gldm_gradient': gldm_gradient,'gldm_lbp_2D' : gldm_lbp_2D,'gldm_lbp_3D_m1': gldm_lbp_3D_m1,'gldm_lbp_3D_m2': gldm_lbp_3D_m2,
               'gldm_lbp_3D_k': gldm_lbp_3D_k,'gldm_square': gldm_square,'gldm_squareroot': gldm_squareroot,
                        
                 
               'glrlm' : glrlm,'glrlm_wavelet' : glrlm_wavelet,'glrlm_exponential': glrlm_exponential,'glrlm_logarithm':glrlm_logarithm,
               'glrlm_gradient': glrlm_gradient,'glrlm_lbp_2D' : glrlm_lbp_2D,'glrlm_lbp_3D_m1': glrlm_lbp_3D_m1,'glrlm_lbp_3D_m2': glrlm_lbp_3D_m2,
               'glrlm_lbp_3D_k': glrlm_lbp_3D_k,'glrlm_square': gldm_square,'glrlm_squareroot': glrlm_squareroot,
               
               'glszm' : glszm,'glszm_wavelet' : glszm_wavelet,'glszm_exponential': glszm_exponential,'glszm_logarithm':glszm_logarithm,
               'glszm_gradient': glszm_gradient,'glszm_lbp_2D' : glszm_lbp_2D,'glszm_lbp_3D_m1': glszm_lbp_3D_m1,'glszm_lbp_3D_m2': glszm_lbp_3D_m2,
               'glszm_lbp_3D_k': glszm_lbp_3D_k,'glszm_square': glszm_square,'glszm_squareroot': glszm_squareroot,
               
               'ngtdm' : ngtdm,'ngtdm_wavelet' : ngtdm_wavelet,'ngtdm_exponential': ngtdm_exponential,'ngtdm_logarithm':ngtdm_logarithm,
               'ngtdm_gradient': ngtdm_gradient,'ngtdm_lbp_2D' : ngtdm_lbp_2D,'ngtdm_lbp_3D_m1': ngtdm_lbp_3D_m1,'ngtdm_lbp_3D_m2': ngtdm_lbp_3D_m2,
               'ngtdm_lbp_3D_k': ngtdm_lbp_3D_k,'ngtdm_square': ngtdm_square,'ngtdm_squareroot': ngtdm_squareroot,}
    
    proportion_robust = []
    for key in Classes.keys():
        proportion_robust.append(len(Classes[key]))
    return list(Classes.keys()),proportion_robust


def filter_class (Features):
         
    original =    [feature for feature in Features if feature.split('_')[0].startswith('original')]
    wavelet =     [feature for feature in Features if feature.split('_')[0].startswith('wavelet')]
    exponential = [feature for feature in Features if feature.split('_')[0].startswith('exponential')]
    logarithm = [feature for feature in Features if feature.split('_')[0].startswith('logarithm')]
    gradient =    [feature for feature in Features if feature.split('_')[0].startswith('gradient')]
    lbp_2D =      [feature for feature in Features if feature.split('_')[0].startswith('lbp-2D')]
    lbp_3D_m1 =   [feature for feature in Features if feature.split('_')[0].startswith('lbp-3D-m1')]
    lbp_3D_m2 =   [feature for feature in Features if feature.split('_')[0].startswith('lbp-3D-m2')]
    lbp_3D_k =    [feature for feature in Features if feature.split('_')[0].startswith('lbp-3D-k')]
    square =      [feature for feature in Features if feature.split('_')[0].startswith('square')]
    squareroot =  [feature for feature in Features if feature.split('_')[0].startswith('squareroot')]

   
    Classes = {'original' : original, 
               'wavelet' : wavelet,
               'exponential' : exponential,
               'logarithm': logarithm,
               'gradient': gradient,
               'lbp_2D': lbp_2D,
               'lbp_3D_m1' : lbp_3D_m1,
               'lbp_3D_m2': lbp_3D_m2,
               'lbp_3D_k': lbp_3D_k,
               'squareroot': squareroot}
    
    proportion_robust = []
    for key in Classes.keys():
        proportion_robust.append(len(Classes[key]))
    return Classes.keys(),proportion_robust

# This is to calculate the number of featurs and then the 
# robust feature in each filter will be normalized to the number of features in each class
def feat_counter(df):
    Features = df.columns
    original =    [feature for feature in Features if feature.split('_')[0].startswith('original')]
    wavelet =     [feature for feature in Features if feature.split('_')[0].startswith('wavelet')]
    exponential = [feature for feature in Features if feature.split('_')[0].startswith('exponential')]
    logarithm = [feature for feature in Features if feature.split('_')[0].startswith('logarithm')]
    gradient =    [feature for feature in Features if feature.split('_')[0].startswith('gradient')]
    lbp_2D =      [feature for feature in Features if feature.split('_')[0].startswith('lbp-2D')]
    lbp_3D_m1 =   [feature for feature in Features if feature.split('_')[0].startswith('lbp-3D-m1')]
    lbp_3D_m2 =   [feature for feature in Features if feature.split('_')[0].startswith('lbp-3D-m2')]
    lbp_3D_k =    [feature for feature in Features if feature.split('_')[0].startswith('lbp-3D-k')]
    square =      [feature for feature in Features if feature.split('_')[0].startswith('square')]
    squareroot =  [feature for feature in Features if feature.split('_')[0].startswith('squareroot')]

    count = [np.shape(original)[0],np.shape(wavelet)[0],np.shape(exponential)[0],np.shape(logarithm)[0],
                                   np.shape(gradient)[0],np.shape(lbp_2D)[0],np.shape(lbp_3D_m1)[0],np.shape(lbp_3D_k)[0],
                                   np.shape(square)[0], np.shape(squareroot)[0]]
    return count

def train_model(x_train,y_train):
    clf = RandomForestClassifier()

    random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 300, stop = 600, num = 20)],
                   'max_features': ['auto'],
                   'max_depth': [10,15,20,25,30,35],
                   'min_samples_split': [3,4,5,6],
                   'min_samples_leaf': [2,3,4],
                   'bootstrap': [True]}
    
    RS = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,n_iter = 20, cv = 5, verbose=0,n_jobs = -1)

    RS.fit(x_train, y_train)
    clf.set_params(**RS.best_params_)
    
    model = clf.fit(x_train,y_train)
    return model


def test_model(model,x_test, y_test):
    y_prob = model.predict_proba(x_test)[:,1]
    auc    = roc_auc_score(y_test,y_prob)
    fpr, tpr, thresholds = roc_curve(y_test,y_prob)  
  
    return fpr,tpr,auc


def cox_selection (x_train, thresh1, thresh2):
    feats = []
    for column in x_train.columns[:-2]:
        cph.fit(x_train[[column,'duration','Tox']],'duration', event_col='Tox',step_size = 0.5)
        
        if cph.concordance_index_>thresh1 or cph.concordance_index_<thresh2:
            feats.append(column)
        
    return feats



def calculate_vif_(X, thresh=100):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]




def boxplotting (featVect, ylabel):
    NGTDM = (featVect.iloc[56:].mean(axis =1))
    GLSZM = (featVect.iloc[45:56].mean(axis =1))
    GLRLM = (featVect.iloc[34:45].mean(axis =1))
    GLDM =  (featVect.iloc[23:34].mean(axis =1))
    GLCM = (featVect.iloc[12:23].mean(axis =1))
    FO =  (featVect.iloc[1:12].mean(axis =1))
    SHAPE = (featVect.iloc[0:1].mean(axis =1))
    fig,ax = plt.subplots(figsize = (10,10))
    plt.boxplot([NGTDM,GLSZM,GLRLM,GLDM,GLCM,FO,SHAPE], showfliers=False)
    plt.xticks([1,2,3,4,5,6,7],['NGTDM','GLSZM','GLRLM',
    'GLDM','GLCM','FO','SHAPE'], fontsize = 20, rotation = 30)
    # plt.yticks([5,10,15,20],['5%','10%', '15%','20%'])
    plt.title('Total number of Rubust features',fontsize=30)
    plt.ylabel(ylabel,fontsize=20)
    plt.ylim([0,100])



def boxplotting_1 (featVect,title, ylabel,figsize):
    NGTDM = (featVect.iloc[56:].sum()/75)*100
    GLSZM = (featVect.iloc[45:56].sum()/270)*100
    GLRLM = (featVect.iloc[34:45].sum()/273)*100
    GLDM =  (featVect.iloc[23:34].sum()/240)*100
    GLCM = (featVect.iloc[12:23].sum()/360)*100
    FO =  (featVect.iloc[1:12].sum()/315)*100
    SHAPE = (featVect.iloc[0:1].sum()/14)*100
    fig,ax = plt.subplots(figsize = figsize)
    plt.boxplot([NGTDM,GLSZM,GLRLM,GLDM,GLCM,FO,SHAPE], showfliers=False)
    plt.xticks([1,2,3,4,5,6,7],['NGTDM','GLSZM','GLRLM',
    'GLDM','GLCM','FO','SHAPE'], fontsize = 20, rotation = 30)
    # plt.yticks([5,10,15,20],['5%','10%', '15%','20%'])
    plt.title(title,fontsize=30)
    plt.ylabel(ylabel,fontsize=20)




def boxplotting_2 (featVect, title, ylabel,figsize):  
    NGTDM = (featVect.iloc[56:].sum()/featVect.sum())*100
    GLSZM = (featVect.iloc[45:56].sum()/featVect.sum())*100
    GLRLM = (featVect.iloc[34:45].sum()/featVect.sum())*100
    GLDM =  (featVect.iloc[23:34].sum()/featVect.sum())*100
    GLCM = (featVect.iloc[12:23].sum()/featVect.sum())*100
    FO =  (featVect.iloc[1:12].sum()/featVect.sum())*100
    SHAPE = (featVect.iloc[0:1].sum()/featVect.sum())*100
    fig,ax = plt.subplots(figsize = figsize)
    plt.boxplot([NGTDM,GLSZM,GLRLM,GLDM,GLCM,FO,SHAPE], showfliers=False)
    plt.xticks([1,2,3,4,5,6,7],['NGTDM','GLSZM','GLRLM',
    'GLDM','GLCM','FO','SHAPE'],fontsize=20, rotation = 30)
    # plt.yticks([5,10,15,20],['5%','10%', '15%','20%'])
    plt.title(title,fontsize=30)
    plt.ylabel(ylabel,fontsize=20)


# normalized to the total number of features
def boxplotting_3 (featVect, title, ylabel,figsize):  
    NGTDM = (featVect.iloc[56:].sum()/1688)*100
    GLSZM = (featVect.iloc[45:56].sum()/1688)*100
    GLRLM = (featVect.iloc[34:45].sum()/1688)*100
    GLDM =  (featVect.iloc[23:34].sum()/1688)*100
    GLCM = (featVect.iloc[12:23].sum()/1688)*100
    FO =  (featVect.iloc[1:12].sum()/1688)*100
    SHAPE = (featVect.iloc[0:1].sum()/1688)*100
    fig,ax = plt.subplots(figsize = figsize)
    plt.boxplot([NGTDM,GLSZM,GLRLM,GLDM,GLCM,FO,SHAPE], showfliers=False)
    plt.xticks([1,2,3,4,5,6,7],['NGTDM','GLSZM','GLRLM',
    'GLDM','GLCM','FO','SHAPE'],fontsize=20, rotation = 30)
    # plt.yticks([5,10,15,20],['5%','10%', '15%','20%'])
    plt.title(title,fontsize=30)
    plt.ylabel(ylabel,fontsize=20)

import matplotlib.font_manager as font_manager

def plotting(acc_mrmr,acc_mrmr_robust,low_rng,ylabel,x_label, inner, type1,type2):
    acc = acc_mrmr.melt()
    acc2 =acc_mrmr_robust.melt()
    acc['Type'] = type1
    acc2['Type'] = type2
    plt.figure(figsize = (20,10))
    a = pd.concat([acc,acc2])
    sns.violinplot(x= 'variable', y = 'value', hue = 'Type',data = a,palette="Set2", color= 'blue',split=True, inner = inner)
    plt.xticks(fontsize = 20, rotation = 20)
    plt.yticks(np.arange(low_rng, 1, step=0.1))
    font = font_manager.FontProperties(family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=15)
    plt.legend(bbox_to_anchor=(0.9, 0.3), prop =font )
    plt.ylabel(ylabel,fontsize = 20)
    plt.xlabel(x_label,fontsize = 20)
    


def calc_avg_values(result_table):     
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 10)
    for i in result_table.index:

        interp_tpr = interp(mean_fpr, result_table.loc[i]['fpr'], result_table.loc[i]['tpr'])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(result_table.loc[i]['auc'])
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
        
    return  mean_tpr, mean_fpr, mean_auc, std_auc, tprs

def feat_values(nparray):
    vals = []
    df_feat = pd.DataFrame(nparray)
    vals.append(np.round(df_feat.iloc[:,:1].mean().sum()))
    vals.append(np.round(df_feat.iloc[:,1:12].mean().sum()))
    vals.append(np.round(df_feat.iloc[:,12:23].mean().sum()))
    vals.append(np.round(df_feat.iloc[:,23:34].mean().sum()))
    vals.append(np.round(df_feat.iloc[:,34:45].mean().sum()))
    vals.append(np.round(df_feat.iloc[:,45:56].mean().sum()))
    vals.append(np.round(df_feat.iloc[:,56:].mean().sum()))
    return vals


def filter_values(nparray):
    vals= []
    df_filt = pd.DataFrame(nparray)
    for i in range(df_filt.shape[1]):
        vals.append(np.round(df_filt.iloc[:,i].mean()))
    return vals



def make_auc(files):
    df = pd.DataFrame()
    cols = ['GE-GE','GE-MIX','GE-TOSHIBA','MIX-GE','MIX-MIX','MIX-TOSHIBA','TOSHIBA-GE','TOSHIBA-MIX','TOSHIBA-TOSHIBA']
    auc  = []
    for file in files:
        auc.append(eval(file).auc.values)

    df = pd.DataFrame(auc).T
    df.columns = cols
    df    
    return df
