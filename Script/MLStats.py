#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* Author : Zhou Wei                                       *
* Date   : Fri Jul 27 12:36:42 2018                       *
* E-mail : welljoea@gmail.com                             *
* You are using the program scripted by Zhou Wei.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''
from sklearn.feature_selection import (VarianceThreshold,  GenericUnivariateSelect,
                                       SelectKBest, SelectFromModel,
                                       f_classif, f_regression,
                                       chi2, mutual_info_classif, mutual_info_regression)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_X_y, resample

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from itertools import combinations
from collections import OrderedDict
from statsmodels.stats.multitest import multipletests
from scipy.stats import (wilcoxon, ranksums, mannwhitneyu, ttest_ind, chi2_contingency,
                         fisher_exact, pearsonr, spearmanr, kendalltau, linregress)
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)

class STATms():
    def __init__(self, _X, _y, *array, **dicts):
        self.X     = _X
        self.y     = _y
        self.array = array
        self.dicts = dicts

    def CHi2_Fish(self, verbose=False, stat = chi2_contingency ): #chi2_contingency fisher_exact
        def _statistic(stat, DF):
            x_uniq = np.unique(DF[:, 0])
            y_uniq = np.unique(DF[:,-1])
            pointX = np.unique( np.percentile(x_uniq, np.linspace(10,90,81), interpolation='nearest') )

            if  stat in [fisher_exact]:
                _Tables = [np.array([
                                    [DF[(DF[:,0] <= x) & (DF[:,-1] == y) ].shape[0], DF[(DF[:,0] <= x) & (DF[:,-1] != y) ].shape[0] ],
                                    [DF[(DF[:,0] >  x) & (DF[:,-1] == y) ].shape[0], DF[(DF[:,0] >  x) & (DF[:,-1] != y) ].shape[0] ],
                                    ]) for x in pointX for y in y_uniq
                            ]
            elif stat in [chi2_contingency]:
                _Tables = [np.array([
                                    [DF[(DF[:,0] <= x_) & (DF[:,-1] == y_) ].shape[0] for y_ in y_uniq ],
                                    [DF[(DF[:,0] >  x_) & (DF[:,-1] == y_) ].shape[0] for y_ in y_uniq ],
                                    ]) for x_ in pointX
                            ]

            _Fishes = np.array([stat(i)
                                for i in _Tables
                                if ((i==0).sum(0).max()<2) & ((i==0).sum(1).max()<len(y_uniq) )
                                ])[:,:2]

            _Fishmin= _Fishes[ _Fishes[:,-1] == _Fishes[:,-1].min() ][0]
            return _Fishmin

        _DF = np.c_[ self.X, self.y ]
        _Fishs   = Parallel( n_jobs=-1 , backend="threading")( delayed( _statistic )( stat, _DF[:, [_n, -1]] ) for _n in range(self.X.shape[1]) )
        _Fishs   = np.array( _Fishs )
        return(_Fishs[:,0], _Fishs[:,1])

    def ClassTest(self, stat=ranksums):
        def rankt(Xi, yi, _combi):
            fs = []
            fp = []
            for i,j in _combi:
                data1 = Xi[np.where(yi == i)]
                data2 = Xi[np.where(yi == j)]

                if (stat== wilcoxon) & (data1.shape[0] != data2.shape[0]):
                    s_p  = [np.inf, np.inf]
                elif (stat== wilcoxon) & (data1.shape[0] == data2.shape[0]):
                    s_p  = stat( sorted(data1) , sorted(data2) )
                else:
                    s_p  = stat(data1, data2 )

                if stat in [ranksums, ttest_ind]:
                    fs.append( np.abs(s_p[0]) )
                    fp.append( s_p[1] )
                elif stat in [wilcoxon, mannwhitneyu]:
                    fs.append( -s_p[0] )
                    fp.append( s_p[1] )

                fs.append( s_p[0] )
                fp.append( s_p[1] )

            _fp = min(fp)
            _fs = [ i for i,j in zip(fs, fp) if j==_fp][0]
            return  [ _fs, _fp ]

        X = self.X
        y = self.y
        combi = list(combinations(np.unique(y), 2))
        f = Parallel( n_jobs=-1, backend="threading")( delayed( rankt )( X[:, _n], y, combi ) for _n in range( X.shape[1]) )
        f = np.array(f)
        return( f[:,0], f[:,1] )

    def VTh(self, thd=-1):
        X = MinMaxScaler().fit_transform(self.X)
        return VarianceThreshold(threshold=thd).fit(X).variances_

    def MI(self,  rs=0):
        return mutual_info_classif(self.X, self.y ,random_state=0)

    def Chi2(self):
        X = MinMaxScaler().fit_transform(self.X)
        return chi2( X, self.y )

    def ANVF(self):
        return f_classif( self.X, self.y )

    def AUC(self):
        def auci(Xi, yi, _combi):
            auc_ = []
            for i,j in _combi:
                Tix = np.where((yi == i) |(yi == j))
                X_i = Xi[ Tix ]
                Y_i = yi[ Tix ]
                auC = roc_auc_score(Y_i, X_i)
                auc_.append( max(1-auC, auC)  )
            return max(auc_)

        X = self.X
        y = self.y
        combi = list(combinations(np.unique(y), 2))
        aucs = Parallel( n_jobs=-1, backend="threading" )( delayed( auci )( X[:, _n], y, combi ) for _n in range( X.shape[1]) )
        return np.array(aucs)

    def FISH(self):
        return self.CHi2_Fish(stat=fisher_exact)

    def Chi2C(self):
        return self.CHi2_Fish(stat=chi2_contingency)

    def RKs(self):
        return self.ClassTest(stat=ranksums)

    def MWU(self):
        return self.ClassTest(stat=mannwhitneyu)

    def TTI(self):
        return self.ClassTest(stat=ttest_ind)

    def _get_stat(self, _name, _X_names ):
        _funct = eval( 'self.%s()'%_name )
        _Cs = OrderedDict()
        if _name in ['VTh', 'MI' ]:
            _Cs[_name + '_score'] = _funct
        elif _name in ['AUC' ]:
            _Cs[_name + '_scorR'] = _funct
            _Cs[_name + '_score'] = _funct.round(2)
        elif _name in ['Chi2', 'ANVF', 'FISH', 'Chi2C','WiC', 'RKs', 'MWU', 'TTI' ]:
            _Cs[_name + '_score']   = _funct[0]
            _Cs[_name + '_P']       = _funct[1]
            _Cs[_name + '_Pfdrbh']  = multipletests( _funct[1], alpha=0.05, method='fdr_bh')[1]
        return pd.DataFrame( _Cs, index=_X_names)

class STATSpy():
    def __init__(self, *array, **dicts):
        self.array = array
        self.dicts = dicts

    def bootstrap(self, data, num_samples, statistics, alpha, repeat=1000):
        n = len(data)
        for _ in range(repeat):
            #boot = resample(data, replace=True, n_samples=n, random_state=None)
            idx = np.random.randint(0, n, size=(num_samples, n))
            samples = data[idx]
            stat = np.sort(statistics(samples, 1))
            return (stat[int((alpha/2)*num_samples)], stat[int((1-alpha/2)*num_samples)])

    def StatC(self, DF, _X_names, _y_name, Fisher=200):
        Xdf = DF[_X_names]
        Y   = DF[_y_name]
        _X, _y = check_X_y( Xdf , Y )

        if _X.shape[1] > Fisher:
            self._chose = ['VTh', 'MI', 'AUC','Chi2', 'ANVF', 'Chi2C', 'RKs', 'MWU', 'TTI']
        else:
            self._chose = ['VTh', 'MI', 'AUC','Chi2', 'ANVF', 'FISH', 'Chi2C', 'RKs', 'MWU', 'TTI']
        _stats = Parallel( n_jobs=len(self._chose) )( delayed( STATms(_X, _y)._get_stat )( _n, _X_names ) for _n in self._chose )
        self._stats = pd.concat( _stats, axis=1)
        self._spraw = self._stats[ [ i+ '_score' if i in ['VTh', 'MI', 'AUC'] else  i+ '_P'      for i in self._chose ] ]
        self._spadj = self._stats[ [ i+ '_score' if i in ['VTh', 'MI', 'AUC'] else  i+ '_Pfdrbh' for i in self._chose ] ]
        return self

    def SKBest(self, scores, K ):
        mask = np.zeros(scores.shape, dtype=bool)
        mask[np.argsort(scores, kind="mergesort")[-int(K):]] = 1
        return mask

    def SPBest(self, pvalues, P, rank='low'):
        if rank =='low':
            return pvalues <= P
        elif rank =='high':
            return pvalues >= P

    def KPBest(self, scores, K, P, Prank):
        Bool = ( self.SKBest(scores, K) & self.SPBest(scores, P, rank=Prank) )
        return Bool
