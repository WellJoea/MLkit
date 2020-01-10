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
                                       chi2, RFECV,
                                       mutual_info_classif, mutual_info_regression)

from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, balanced_accuracy_score,
                             classification_report, make_scorer, precision_score,
                             precision_recall_curve, mean_squared_error,
                             roc_curve, auc, r2_score, mean_absolute_error)

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from scipy.stats import pearsonr, spearmanr, kendalltau, linregress

from functools import partial
import os
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)

from .MLPlots import ClusT, MPlot
from .MLEstimators import ML
from .MLOpenWrite import OpenM, Openf
from .MLUnsupervising import Decomposition
from .MLStats import STATSpy
from .MLUtilities import CrossvalidationSplit

class Featuring():
    def __init__(self, arg, log, *array, score=None, model='RF', Type='C',  **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.score = score
        self.model = model
        self.dicts = dicts
        self.SPLIT = [ CrossvalidationSplit(n_splits=10, test_size=n, CVt=arg.CVmodel) for n in np.linspace(0.2,0.4,20, endpoint=False) ]
        self.Retim = 1 if arg.CVmodel=='LOU' else len(self.SPLIT)
        self.estimator= ML(self.model, Type=Type).GetPara().estimator

    def SelectKBest_C(self, DF, _X_names, _y_name):
        _Stat = STATSpy().StatC(DF, _X_names, _y_name, Fisher=self.arg.Fisher)

        if len(_X_names) > self.arg.Fisher:
            self.log.CWA( "with slow computing speed, fisher exact test is discarded as features number is larger than the %s."%self.arg.Fisher)

        Openf('%s%s_Class_selectkBset.xls'%(self.arg.output, _y_name), _Stat._stats).openv()

        if self.arg.pvaluetype== 'P':
            F_Stat = _Stat._spraw
        elif self.arg.pvaluetype== 'Padj':
            F_Stat = _Stat._spadj
        F_Stat.columns = [ i.split('_')[0] for i in F_Stat.columns]

        if self.arg.SelectB :
            F_Stat = F_Stat.filter(items=self.arg.SelectB)
            Openf('%s%s_Class_selectkBset.Select.xls'%(self.arg.output, _y_name),F_Stat).openv()

            k_F = [ int(K) if K > 1 else round(K*len(_X_names), 0) for K in self.arg.SelectK ]
            s_F = self.arg.SelectS
            spstat = ['VTh','MI', 'AUC', 'Chi2', 'ANVF', 'FISH', 'Chi2C','WiC', 'RKs', 'MWU', 'TTI' ]
            rule_  = {}
            for i in range(len(spstat)):
                if i < 3 :
                    rule_[ spstat[i] ] = [k_F[i], s_F[i], 'high']
                else:
                    rule_[ spstat[i] ] = [k_F[-1], s_F[-1], 'low']

            Fearture_bool = pd.DataFrame(index = F_Stat.index)
            for _f in F_Stat.columns:
                Fearture_bool[_f] = STATSpy().KPBest(F_Stat[_f], *rule_[_f])

            Fearture = Fearture_bool.astype(int)
            Fearture['True_ratio'] = Fearture.mean(1)
            Fearture.sort_values(by=['True_ratio'], ascending=[False], inplace=True)

            Fearture_final = Fearture[ (Fearture['True_ratio'] >= self.arg.SelectR) ]
            Fearture_drop  = list( set(Fearture.index) - set( Fearture_final.index) )

            self.log.CIF( ('%s: SelectKBest_C'%_y_name).center(45, '-') )
            self.log.CIF( 'Raw Feature: %s' % Fearture.shape[0]  )
            self.log.CIF( 'drop Feature: %s'% len(Fearture_drop))
            self.log.CIF( 'drop Features: %s' % Fearture_drop )
            self.log.CIF( 'final Feature: %s' % Fearture_final.index.to_list() )
            self.log.NIF( 'Feature bools:\n%s' % Fearture)
            self.log.CIF(45 * '-')
            return Fearture_final.index.to_list()
        else:
            return _X_names

    def SelectKBest_R(self, Xdf, Ydf):
        def tranvisR(X, y, stat=''):
            f = []
            for n in range( X.shape[1]):
                s_p  = stat( X[:, n], y )
                if 'correlation' in dir(s_p):
                    f.append( [ s_p.correlation, s_p.pvalue ] )
                elif 'rvalue' in dir(s_p):
                    f.append( [ s_p.rvalue, s_p.pvalue ] )
                else:
                    f.append( [ s_p[0], s_p[1] ] )
            f = np.array(f)
            return( np.abs(f[:,0]), f[:,1] )

        k_F = int(self.arg.SelectK if ( self.arg.SelectK > 1) else round(self.arg.SelectK*len(Xdf.columns), 0))
        Fearture_select ={
            'VTh'  : VarianceThreshold(threshold=0.8*(1-0.8)),
            'ANVF' : SelectKBest(f_regression, k=k_F),
            'MI'   : SelectKBest(score_func=partial(mutual_info_regression, random_state=0), k=k_F),
            'PEAS' : SelectKBest(score_func=partial(tranvisR, stat=pearsonr  ), k=k_F),
            'SPM'  : SelectKBest(score_func=partial(tranvisR, stat=spearmanr ), k=k_F),
            'KDT'  : SelectKBest(score_func=partial(tranvisR, stat=kendalltau), k=k_F),
            'LR'   : SelectKBest(score_func=partial(tranvisR, stat=linregress), k=k_F),
            #'PRS'  : SelectKBest(lambda Xi, Yi: np.array(list(map(lambda x:pearsonr(x, Yi), Xi.T))).T[0], k=k_F),
        }

        Fearture_bool = []
        Fearture_name = []
        Fearture_PVS  = []
        for j in self.arg.SelectB:
            if j in Fearture_select.keys():
                i = Fearture_select[j]
                #F_t = i.fit_transform(Xdf, Ydf)
                G_s = i.get_support()

                if j in [ 'VTh', 'MI' ]:
                    P_v = i.scores_
                else:
                    P_v = i.pvalues_

                Fearture_name.append(j)
                Fearture_bool.append(G_s)
                Fearture_PVS.append(P_v)

        Fearture = pd.DataFrame(np.array(Fearture_bool).T, columns=Fearture_name,index=Xdf.columns)
        FearturP = pd.DataFrame(np.array(Fearture_PVS ).T, columns=Fearture_name,index=Xdf.columns)

        Openf('%s%s_Survival_SelectkBset.xls'%(self.arg.output, Ydf.name),Fearture).openv()
        Openf('%s%s_Survival_SelectkBset.Pvalues.xls'%(self.arg.output, Ydf.name),FearturP).openv()

        Fearture_filer = Fearture[~Fearture.any(1)]
        Fearture_final = list( set(Fearture.index) - set( Fearture_filer.index) )

        self.log.CIF( ('%s: SelectKBest_R'%Ydf.name).center(45, '-') )
        self.log.CIF( 'Raw   Feature: %s' % Fearture.shape[0]  )
        self.log.CIF( 'drop  Feature: %s\n%s' % (Fearture_filer.shape[0], Fearture_filer.index.tolist()) )
        self.log.CIF( 'final Feature: %s'  % Fearture_final )
        self.log.NIF( 'Feature bools:\n%s' % Fearture)
        self.log.CIF(45 * '-')
        return Fearture_final

    def RFECV_RC(self, Xdf, Ydf):
        select_Sup = []
        select_Sco = []
        for _ in range(self.arg.SelectCV_rep) :
            for _x in range( self.Retim  ) :
                selector = RFECV(self.estimator,
                                 step=1,
                                 cv=self.SPLIT[_x],
                                 n_jobs=self.arg.n_job,
                                 scoring =self.score)
                selector = selector.fit(Xdf, Ydf)
                select_Sco.append(selector.grid_scores_)
                select_Sup.append(selector.ranking_)
                self.log.CIF('RFECV: %s -> %s, %2d%% completed.'%(self.arg.model, self.model, (_x+1)*100/self.Retim  ))
        select_Sco = pd.DataFrame(np.array(select_Sco).T, index=range(1, Xdf.shape[1]+1))
        select_Sup = pd.DataFrame(np.array(select_Sup).T, index=Xdf.columns)
        select_Sco.columns = range(select_Sco.shape[1])
        select_Sup.columns = [ '%s_%s'%(self.model, i) for i in range(select_Sup.shape[1]) ]
        select_feature = (select_Sup==1).sum(0).values

        Openf('%s%s_Class.Regress_RFECV_%s_ranking.xls'%(self.arg.output, Ydf.name, self.model),select_Sup).openv()
        MPlot('%s%s_Class.Regress_RFECV_%s_Fscore.pdf'%(self.arg.output, Ydf.name, self.model)).Feature_Sorce(select_Sco, select_feature, self.model, Ydf.name, 'RFECV' )
        #select_Sup = select_Sup.reindex( select_Sup.sum(1).sort_values().index, axis=0)

        return( select_Sup )

    def SFSCV_RC(self, Xdf, Ydf):
        select_Sup = []
        select_Sco = []
        k_features = ()

        if set(self.arg.k_features) & set(['best', 'parsimonious']):
            k_features = self.arg.k_features[0]
        elif len(self.arg.k_features) == 1:
            k_features = int(float(self.arg.k_features[0]) * len(Xdf.columns))
        elif len(self.arg.k_features) == 2:
            k_features = tuple( [ int(float(i) * len(Xdf.columns)) for i in self.arg.k_features ] )

        for _ in range(self.arg.SelectCV_rep):   #pool
            for _x in range( self.Retim  ) :
                selector = SFS(self.estimator,
                        k_features= k_features,
                        forward=True,
                        floating=False, 
                        verbose=0,
                        scoring=self.score,
                        n_jobs=self.arg.n_job,
                        cv=self.SPLIT[_x])
                selector = selector.fit(Xdf, Ydf)
                select_feat = selector.k_feature_names_
                select_feat = pd.DataFrame([[1]]*len(select_feat),index=select_feat)
                avg_score = pd.DataFrame(selector.subsets_).loc['avg_score']
                select_Sup.append(select_feat)
                select_Sco.append(avg_score)
                self.log.CIF('SFSCV: %s -> %s, %2d%% completed.'%(self.arg.model, self.model, (_x+1)*100/self.Retim  ))

        select_Sup = pd.concat(select_Sup,axis=1,sort=False).fillna(0).astype(int)
        select_Sco = pd.concat(select_Sco,axis=1,sort=False).fillna(0)
        select_Sco.columns = range(select_Sco.shape[1])
        select_Sup.columns = [ '%s_%s'%(self.model, i) for i in range(select_Sup.shape[1]) ]
        select_feature = (select_Sup==1).sum(0).values

        Openf('%s%s_Class_Regress_SFSCV_%s_ranking.xls'%(self.arg.output, Ydf.name, self.model),select_Sup).openv()
        MPlot('%s%s_Class.Regress_SFSCV_%s_Fscore.pdf'%(self.arg.output, Ydf.name, self.model)).Feature_Sorce( select_Sco, select_feature, self.model, Ydf.name, 'SFSCV' )

        return( select_Sup )

    def Coef_SC(self, Xdf, Ydf):
        self.arg.CWA("Coef_SC has been deprecated since version 0.2." )
        pass

class Feature_selection():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        self.arg.output = '%s/02FeatureSLT/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs( os.path.dirname(self.arg.output) , exist_ok=True)

    def Serial(self, _All_FS_models, _XYdf, _X_names, _Y_name, Type = 'C' ):
        _Xdf = _XYdf[_X_names]
        _Ydf = _XYdf[_Y_name]
        for _modeli, _packai, _rate_i in _All_FS_models:
            _set_Rank = []
            for model_j in _modeli:
                self.log.CIF('the %s model use the %s estimator for %s.'%(self.arg.model, model_j, _packai ))
                _i_Rank = eval( 'Featuring(self.arg, self.log, model =model_j, Type = Type).%s(_Xdf, _Ydf)'%_packai ) 
                _set_Rank.append(_i_Rank)

            _set_Rank = pd.concat(_set_Rank, axis=1)
            _set_Rank['SUM_Rank'] = _set_Rank.sum(1)
            _set_Rank['SUM_Bool'] = (_set_Rank==1).sum(1)
            _set_Rank.sort_values(by=['SUM_Bool', 'SUM_Rank'], ascending=[False, True], inplace = True)

            if 0 < _rate_i <=1 :
                _rate_i  =  int(_set_Rank.shape[0] * _rate_i)
            if (_rate_i > _set_Rank.shape[0]) or (_rate_i == 0 ):
                _rate_i = _set_Rank.shape[0]

            B_number = _set_Rank.iloc[int(_rate_i-1), : ]['SUM_Bool']
            if B_number ==0: B_number += 1
            _set_Rank = _set_Rank[_set_Rank['SUM_Bool'] >= B_number]

            Drop = list( set( _Xdf.columns) - set(_set_Rank.index))
            KEEP = _set_Rank.index.tolist()
            _Xdf = _Xdf[KEEP]

            self.log.CIF( ('%s: SelectUseModel'%_Y_name).center(45, '-') )
            self.log.CIF( '%s drop %s Features: %s'%( _packai, len(Drop), Drop ) )
            self.log.CIF( '%s keep %s Features: %s'%( _packai, len(KEEP), KEEP ) )
            self.log.NIF( 'Features Selected: \n%s' % _set_Rank )
            self.log.CIF(45 * '-')

        return(KEEP)

    def Parallel(self, _All_FS_models, _XYdf, _X_names, _Y_name, Type = 'C' ):
        _Xdf = _XYdf[_X_names]
        _Ydf = _XYdf[_Y_name]
        _set_Rank,_set_pack, _set_rate= [], [], []

        for _modeli, _packai, _rate_i in _All_FS_models:
            _set_pack.append(_packai)
            _set_rate.append(_rate_i)
            for model_j in _modeli:
                self.log.CIF('the %s model use the %s estimator for %s.'%(self.arg.model, model_j, _packai ))
                _i_Rank = eval( 'Featuring(self.arg, self.log, model =model_j, Type = Type).%s(_Xdf, _Ydf)'%_packai ) 
                _set_Rank.append(_i_Rank)

        _set_Rank = pd.concat(_set_Rank, axis=1, sort =False)
        _set_Rank['SUM_Rank'] = _set_Rank.sum(1)
        _set_Rank['SUM_Bool'] = (_set_Rank==1).sum(1)
        _set_Rank.sort_values(by=['SUM_Bool', 'SUM_Rank'], ascending=[False, True], inplace = True)

        _rate_m = _set_rate[0]
        _pack_a = '+'.join(_set_pack)

        if 0 < _rate_m <=1 :
            _rate_m  =  int(_set_Rank.shape[0] * _rate_m)
        if (_rate_m > _set_Rank.shape[0]) or (_rate_m == 0 ):
            _rate_m = _set_Rank.shape[0]

        B_number = _set_Rank.iloc[int(_rate_m-1), : ]['SUM_Bool']
        if B_number ==0: B_number += 1
        _set_Rank = _set_Rank[_set_Rank['SUM_Bool'] >= B_number]

        Drop = list( set( _Xdf.columns) - set(_set_Rank.index))
        KEEP = _set_Rank.index.tolist()
        _Xdf = _Xdf[KEEP]

        self.log.CIF( ('%s: SelectUseModel'%_Y_name).center(45, '-') )
        self.log.CIF( '%s drop %s Features: %s'%( _pack_a, len(Drop), Drop ) )
        self.log.CIF( '%s keep %s Features: %s'%( _pack_a, len(KEEP), KEEP ) )
        self.log.NIF( 'Features Selected: \n%s' % _set_Rank )
        self.log.CIF(45 * '-')

        return(KEEP)

    def SelectUseModel(self, _XYdf, _X_names, _Y_name, Type = 'C'):
        def transmodel(model):
            for  _m in model:
                if _m in ['SVM', 'SVMrbf', 'KNN','RNN'] :
                    trans = 'SVMlinear'
                elif _m in ['MLP','AdaB_DT' ]:
                    trans = 'XGB'
                elif _m in ['MNB','CNB','BNB','GNB']:
                    trans = 'MNB'
                else:
                    trans = _m
                yield trans

        fmodel = self.arg.specifM if self.arg.specifM  else [ self.arg.model ]
        tmodel = transmodel(fmodel)

        _All_FS_models = []
        if len(self.arg.SelectE) >0:
            for _fs in self.arg.SelectE:
                if _fs =='RFECV':
                    _All_FS_models.append([tmodel, 'RFECV_RC', self.arg.RFE_rate])
                elif _fs =='SFSCV':
                    _All_FS_models.append([fmodel, 'SFSCV_RC', self.arg.SFS_rate])
                elif _fs =='CoefSC':
                    _All_FS_models.append([tmodel, 'Coef_SC' , self.arg.CFS_rate])

        if len(_All_FS_models) > 0:
            if self.arg.set   == 'parallel':
                return(self.Parallel(_All_FS_models, _XYdf, _X_names, _Y_name, Type = 'C') )
            elif self.arg.set == 'serial':
                return(self.Serial(_All_FS_models, _XYdf, _X_names, _Y_name, Type = 'C') )
        else:
            return _X_names

    def Fselect(self):
        (group, RYa, CYa, Xa, Xg) = OpenM(self.arg, self.log).openg()
        Standfile =  '%s/01PreProcess/%sTrainTest.standard.data.xls'%(self.arg.outdir, self.arg.header )
        dfall = Openf(Standfile, index_col=0).openb()

        _XALL  = [i for i in dfall.columns if i in Xa]
        YType  =  group[ (group.Variables.isin( RYa + CYa)) ][['Variables','Type']]

        for Yi,Typi in YType.values:
            self.log.CIF( ('%s: Feature Selecting'%Yi).center(45, '*') )

            _Xdf = dfall[_XALL]
            _Ydf = dfall[Yi]
            KEEP = _XALL

            if self.arg.SelectB:
                if Typi == 'C':
                    self.log.NIF( '%s Value Counts:\n%s' %(Yi, dfall[Yi].value_counts().to_string()) )
                    KEEP = Featuring(self.arg, self.log).SelectKBest_C( dfall, KEEP, Yi )
                elif Typi == 'R':
                    KEEP = Featuring(self.arg, self.log).SelectKBest_R( _Xdf, _Ydf)
                    continue

            Univa_XYdf = dfall[ KEEP + [Yi] ]
            Openf('%s%s_Class_Regress_uni_FeatureSLT.Data.xls'%(self.arg.output, Yi), Univa_XYdf).openv()
            ClusT('%s%s_Class_Regress_uni_FeatureSLT.Data_complete.pdf'%(self.arg.output, Yi) ).Plot_heat(dfall[KEEP] , dfall[[Yi]], Xg, method='complete')
            ClusT('%s%s_Class_Regress_uni_FeatureSLT.Data_average.pdf' %(self.arg.output, Yi) ).Plot_heat(dfall[KEEP] , dfall[[Yi]], Xg, method='average' )

            KEEP = self.SelectUseModel(dfall, KEEP, Yi, Type=Typi )
            Final_XYdf = dfall[ KEEP + [Yi] ]
            Openf('%s%s_Class_Regress_FeatureSLT.Data.xls'%(self.arg.output, Yi), Final_XYdf).openv()
            ClusT('%s%s_Class_Regress_FeatureSLT.Data_complete.pdf'%(self.arg.output, Yi) ).Plot_heat(dfall[KEEP] , dfall[[Yi]], Xg, method='complete')
            ClusT('%s%s_Class_Regress_FeatureSLT.Data_average.pdf' %(self.arg.output, Yi) ).Plot_heat(dfall[KEEP] , dfall[[Yi]], Xg, method='average' )

            Decomposition(self.arg, self.log).PCA( Final_XYdf, KEEP, [Yi], '%s%s_Class_Regress_FeatureSLT'%(self.arg.output, Yi) )

            self.log.CIF( ('%s: Feature Selecting Finish'%Yi).center(45, '*') )


