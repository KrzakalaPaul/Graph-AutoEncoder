from datasets.base import Dataset
import torch as torch
from torch.nn import Module
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC,LinearSVR
from sklearn.linear_model import Ridge,SGDRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def eval_embedding(encoder: Module, 
                   dataset: Dataset,
                   n_samples = 100000,
                   batchsize = 512,
                   device = 'cuda',
                   save_dir = None):
    
    encoder.to(device)
    encoder.eval()

    embedding_list = []
    feature_list = []

    dataloader = dataset.get_dataloader_reg(batchsize=batchsize, TSNE = False, dataset_size=n_samples)

    size=0


    for inputs,targets in dataloader: 

        inputs = inputs.to(device)
        targets = targets

        # Save

        embedding_list.append(encoder(inputs).detach().cpu().numpy())
        feature_list.append(targets.numpy())

        size+=batchsize

        if size>n_samples:
            break

    
    
    X=np.concatenate(embedding_list)
    y=np.concatenate(feature_list)

    scores = {}

    feature = dataset.feature
    type = feature.type
    scoring = feature.scoring


    if type == 'discrete':

        print('')
        print('LDA')

        model = LDA()
        score = cross_val_score(model,X,y,cv=5,scoring=scoring)
        scores[f'LDA'] = score.mean()
        print(score.mean())

        print('')
        print('SVC + grid search')

        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("SVC", LinearSVC(dual=False))])

        param_grid = {"SVC__C": [1e-4,1e-3,1e-2,1e-1,1,10],"SVC__class_weight": ['balanced',None]}

        search = GridSearchCV(pipe, param_grid, n_jobs=-1,scoring=scoring)  
        search.fit(X,y)
        C = search.best_params_["SVC__C"]
        class_weight = search.best_params_["SVC__class_weight"]

        model = Pipeline(steps=[("scaler", StandardScaler()), ("SVC", LinearSVC(C=C,class_weight=class_weight,dual=False))])
        score = cross_val_score(model,X,y,cv=5,scoring=scoring)

        scores[f'SVC + grid search'] = score.mean()
        print(score.mean())

        '''
        print('')
        print('Random Forest')
        clf = RandomForestClassifier(n_estimators=50)
        score = cross_val_score(clf,X,y,cv=5,scoring=scoring)
        scores['Random Forest'] = score.mean()
        print(score.mean())
        '''
        
    elif type == 'continuous':

        print('')
        print('Ridge + grid search')

        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("Ridge", SGDRegressor())])

        param_grid = {"Ridge__alpha": [1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]}

        search = GridSearchCV(pipe, param_grid, n_jobs=-1,scoring=scoring)  
        search.fit(X,y.ravel())
        alpha = search.best_params_["Ridge__alpha"]

        model = Pipeline(steps=[("scaler", StandardScaler()), ("Ridge", SGDRegressor(alpha=alpha))])
        score = cross_val_score(model,X,y.ravel(),cv=5,scoring=scoring)

        scores[f'SVR + grid search'] = -score.mean()
        print(-score.mean())

        '''
        print('')
        print('Random Forest')
        clf = RandomForestRegressor(n_estimators=50,criterion="absolute_error")
        score = cross_val_score(clf,X,y.ravel(),cv=5,scoring=scoring)
        scores['Random Forest'] = -score.mean()
        print(-score.mean())
        '''

    if save_dir!=None:
        pd.Series(scores).to_csv(save_dir+'/scores.csv')
