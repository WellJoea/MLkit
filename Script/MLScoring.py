from sklearn.metrics import (accuracy_score, f1_score, 
                             classification_report,
                             precision_recall_curve, mean_squared_error, 
                             roc_curve, auc, r2_score, mean_absolute_error, 
                             average_precision_score, explained_variance_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
from scipy.stats import pearsonr, stats

import joblib
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import math
import os

from .MLOpenWrite import OpenM, Openf
from .MLEstimators import ML
from .MLPlots import ClusT, ROCPR, MPlot
from .MLSupervising import Processing

class Binning():
    def __init__(self, arg, log, *array, score=None, model='RF', **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.score = score
        self.model = model
        self.dicts = dicts

    def ChimergeX(self, Data, X_name, Y_name):
        X = Data[X_name]
        Y = Data[Y_name]
        max_intervals = self.arg.max_intervals
        distinct_X = sorted(set(X))

        init_count = {l: 0 for l in sorted(set(Y)) } # A helper function for padding the Counter()
        intervals  = np.array([distinct_X]*2).T      # Initialize the intervals for each attribute

        min_c = 0
        while (len(intervals) > max_intervals) or (min_c == 0):
            chi = []
            for i in range(len(intervals)-1):
                obs0  = Data[X.between(intervals[i, 0],   intervals[i, 1])]
                obs1  = Data[X.between(intervals[i+1, 0], intervals[i+1, 1])]
                total = len(obs0) + len(obs1)

                count_0 = np.array([v for i, v in {**init_count, **Counter(obs0[Y_name])}.items()])
                count_1 = np.array([v for i, v in {**init_count, **Counter(obs1[Y_name])}.items()])
                count_total = count_0 + count_1

                expected_0 = count_total*sum(count_0)/total
                expected_1 = count_total*sum(count_1)/total

                chi_ = 0
                for i, v in enumerate(count_total):
                    if expected_0[i] != 0 :
                        chi_ += (count_0[i] - expected_0[i])**2/expected_0[i]
                    if expected_1[i] != 0 :
                        chi_ += (count_1[i] - expected_1[i])**2/expected_1[i]
                chi.append(chi_)

            min_c = min(chi)
            min_i = chi.index(min_c)
            min_int = intervals[min_i : min_i+2]
            intervals[min_i] = [min_int.min(), min_int.max()]
            intervals = np.delete(intervals, min_i + 1 ,axis=0)
        intervals = [float('-inf')] + intervals[:-1,1].tolist() + [float('+inf')]
        return intervals

    def SpearManr(self, Data, X_name, Y_name):
        X = Data[X_name]
        Y = Data[Y_name]
        #max_intervals = self.arg.max_intervals
        max_intervals = int(X.shape[0]/2)
        distinct_Y = {v:i for i, v in enumerate(sorted(set(Y)))}
        Y_new = Y.map(distinct_Y)

        r = 0
        w = np.array([1,2,3])
        while (np.abs(r) <= 0.5) or np.isnan(w).any() :
            inter_df = pd.DataFrame({"X": X, "Y": Y_new, "Bucket": pd.qcut(X, max_intervals, duplicates='drop')})
            inter_df = inter_df.groupby('Bucket', as_index = True)

            intervals = pd.qcut(X, max_intervals, duplicates='drop', retbins =True)[-1]
            intervals = [float('-inf')] + intervals.tolist()[1:-1] + [float('+inf')]
            max_intervals -= 1
            r, p = stats.spearmanr(inter_df.mean()['X'], inter_df.mean()['Y'])
            w = inter_df['Y'].mean()

        return(intervals)

    def decession_tree_bin(self, Data, X_name, Y_name):
        X = Data[[X_name]]
        Y = Data[Y_name]
        clf=DecisionTreeClassifier(criterion = 'entropy',
                                    max_leaf_nodes= self.arg.max_intervals,
                                    min_samples_leaf = 0.05).fit(X,Y)

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold
        boundary = []
        for i in range(n_nodes):
            if children_left[i]!=children_right[i]:
                boundary.append(threshold[i])
        sort_boundary = sorted(boundary)
        return [float('-inf')] + sort_boundary + [float('+inf')]

    def WOE(self, Data, X_name, Y_name, intervals):
        X = Data[X_name]
        Y = Data[Y_name]

        distinct_Y = {v:i for i, v in enumerate(sorted(set(Y)))}
        Y_new = Y.map(distinct_Y)

        Bad  = Y_new.value_counts()[0]
        Good = Y_new.value_counts()[1]

        def Group_attribute(_interval):
            inter_df = pd.DataFrame({"X": X, "Y": Y_new, "Bucket": pd.cut(X, _interval, right=True)})
            inter_df = inter_df.groupby('Bucket', as_index = True)

            inter_ = pd.DataFrame(inter_df.count().Y, columns = ['all_count'])
            inter_['all_count']  = inter_df.count().Y
            inter_['good_count'] = inter_df.sum().Y
            inter_['bed_count']  = inter_['all_count'] - inter_['good_count']
            inter_['min_count']  = inter_[['good_count','bed_count']].min(1)
            return(inter_)

        inter_New = Group_attribute(intervals)
        min_count = inter_New['min_count'].min()

        while min_count==0:
            min_index = inter_New['min_count'].tolist().index(min_count) + 1 

            if min_index == 1:
                intervals = [v for i,v  in enumerate(intervals) if i != min_index]
            elif min_index == len(intervals)-1:
                intervals = [v for i,v  in enumerate(intervals) if i != min_index-1]
            else:
                interv_before = [v for i,v  in enumerate(intervals) if i != min_index-1]
                interv_after  = [v for i,v  in enumerate(intervals) if i != min_index  ]

                _before = Group_attribute(interv_before)
                _after  = Group_attribute(interv_after)

                _before_iv = _before['good_count'][min_index-1]
                _after_iv  = _after['good_count'][min_index-1]

                intervals = interv_before if _before_iv > _after_iv else interv_after
            inter_New = Group_attribute(intervals)
            min_count = inter_New['min_count'].min()

        inter_New['good_rate']  = inter_New['good_count']/Good
        inter_New['bed_rate']   = inter_New['bed_count']/Bad
        inter_New['woe']        = np.log(inter_New['good_rate']/inter_New['bed_rate'])
        inter_New['iv']         = (inter_New['good_rate'] - inter_New['bed_rate'])*inter_New['woe']
        IV_sum = inter_New['iv'].sum()
        return( [inter_New, intervals, IV_sum ])

    def Final_WOEIV(self, Data, X_name, Y_name):
        '''
        All_split = [[ 'DT', 'CM', 'SM' ],
                     [ self.decession_tree_bin, self.ChimergeX, self.SpearManr ],
                    ]
        '''
        All_split = [[ 'DT', 'CM' ],
                     [ self.decession_tree_bin, self.ChimergeX ],
                    ]

        parallel = Parallel(n_jobs=2)
        All_inter = parallel(delayed(_m)(
                                Data, X_name, Y_name )
                                for _m in All_split[1])

        parallel = Parallel(n_jobs=2)
        All_WoeIv = parallel(delayed(self.WOE)(
                                Data, X_name, Y_name, v )
                                for v in All_inter)

        All_WoeIv = pd.DataFrame(All_WoeIv, index= All_split[0], columns=['descri_', 'interv_', 'IV_'])

        MaxIV_Way = All_WoeIv.IV_.idxmax()
        Final_woeiv = All_WoeIv.loc[MaxIV_Way,]
        return( [X_name, Final_woeiv] )

    def _Fill_WOE(self, Final_, Data, X_name):
        X = Data[X_name]
        Final_inter = Final_['interv_']
        Final_descr = Final_['descri_']
        inter_df = pd.DataFrame({"X": X, "Bucket": pd.cut(X, Final_inter, right=True)})
        inter_df[X_name] = inter_df['Bucket'].map(Final_descr['woe'])

        return(inter_df[X_name])

    def _TT_WOEIV(self, X_train, X_test, y_train, y_test ):
        X_name = X_train.columns
        y_name = y_train.name
        D_train   = pd.concat((X_train,y_train), axis=1, sort=False)
        D_test    = pd.concat((X_test ,y_test ), axis=1, sort=False)

        parallel = Parallel(n_jobs=15)
        Data_WOEIV = parallel(delayed(self.Final_WOEIV)(
                                D_train[[X_i ,y_name]], X_i, y_name )
                                for X_i in X_name)
        Data_WOEIV = { i[0] : i[1] for i in Data_WOEIV }

        parallel = Parallel(n_jobs=15)
        train_WOE = parallel(delayed(self._Fill_WOE)(
                        Data_WOEIV[X_j], D_train, X_j)
                        for X_j in X_name )

        parallel = Parallel(n_jobs=15)
        test_WOE = parallel(delayed(self._Fill_WOE)(
                        Data_WOEIV[X_k], D_test, X_k)
                        for X_k in X_name)

        train_WOE = pd.concat(train_WOE,axis=1 )
        train_WOE[y_name] = y_train

        test_WOE = pd.concat(test_WOE,axis=1 )
        test_WOE[y_name] = y_test

        return(train_WOE[X_name], test_WOE[X_name], train_WOE[y_name], test_WOE[y_name], Data_WOEIV )
    
    def _PD_WOEIV(self, _F_WOEIV, X_predict, y_predict):
        X_name = X_predict.columns
        y_name = y_predict.name
        parallel = Parallel(n_jobs=15)
        pred_WOE = parallel(delayed(self._Fill_WOE)(
                        _F_WOEIV[X_p], X_predict, X_p)
                        for X_p in X_name)
        pred_WOE = pd.concat(pred_WOE, axis=1 )
        pred_WOE[y_name] = y_name
        return( pred_WOE[X_name], pred_WOE[y_name] )

    def LRScore(self, Data, X_name, Y_name ):
        Xdf = Data[X_name]
        Ydf = Data[Y_name]

        basepoints = self.arg.basepoints
        baseodds = self.arg.baseodds
        pdo = self.arg.PDO
        beta = pdo/math.log(2)
        alpha = basepoints + beta * math.log(baseodds)

        LG_p, LG_e = ML('LRCV', SearchCV = 'GSCV').Classification()

        _X_SCOs = []
        _X_WOEs = []
        _X_MODEL= []

        CVM = Processing(self.arg, self.log).CrossvalidationSplit(Xdf, Ydf)
        for train_index, test_index in CVM:
            X_train, X_test = Xdf.iloc[train_index, :], Xdf.iloc[test_index, :]
            try:
                y_train, y_test = Ydf.iloc[train_index, :], Ydf.iloc[test_index, :]
            except pd.core.indexing.IndexingError:
                y_train, y_test = Ydf[train_index], Ydf[test_index]

            X_train, X_test, y_train, y_test , _WOEIV = self._TT_WOEIV(X_train, X_test, y_train, y_test)
            clf = GridSearchCV(LG_e, LG_p, n_jobs=15, cv=10, return_train_score=True, iid=True)
            clf.fit(X_train, y_train)
            intercept = clf.best_estimator_.intercept_
            coef_s    = clf.best_estimator_.coef_
            base_sore     = alpha - beta * intercept
            Y_tespred     = clf.best_estimator_.predict(X_test)
            Y_tespred_pro = clf.best_estimator_.predict_proba(X_test)

            X_Scor = X_test*coef_s*beta
            X_Scor['_Scores'] = X_Scor.sum(1) + base_sore
            X_Scor[Y_name]    = y_test
            X_Scor['predict'] = Y_tespred
            X_Scor['pred_pro']= Y_tespred_pro[:,1]

            X_Woes = X_test
            X_Woes[Y_name] = y_test
            _X_SCOs.append(X_Scor)
            _X_WOEs.append(X_Woes)
            Model_j = {'Features': X_name, 
                       'Target'  : Y_name,
                       'Best_mod': clf.best_estimator_, 
                       'F_WOEIV' : _WOEIV,
                       'beta'    : beta,
                       'alpha'   : alpha,
                      }
            _X_MODEL.append(Model_j)

            self.log.CIF( ('%s %s Model Fitting and hyperparametering'% (Y_name, 'LRCV')).center(45, '-') )
            self.log.CIF( '%s best parameters: \n%s' %('LRCV', clf.best_params_) )
            self.log.CIF( '%s best score: %s' %('LRCV', clf.best_score_) )
            self.log.CIF( 45 * '-' )

        joblib.dump(_X_MODEL, '%s%s_Class_LG_model_WOEIV.data' % (self.arg.output, Y_name), compress=1)
        _X_SCOs = pd.concat(_X_SCOs, axis=0)
        _X_SCOs = _X_SCOs.groupby([_X_SCOs.index]).mean()
        _X_SCOs['mean_predict'] = _X_SCOs['pred_pro'].apply( lambda x : 1 if x >0.5 else 0 )
        _X_SCOs['mode_predict'] = _X_SCOs['predict' ].apply( lambda x : 1 if x >0.5 else 0 )

        _X_WOEs = pd.concat(_X_WOEs, axis=0)
        _X_WOEs = _X_WOEs.groupby([_X_WOEs.index]).mean()

        _accuracy  = accuracy_score(_X_SCOs[Y_name], _X_SCOs['mean_predict'])
        _R2_score  = r2_score(_X_SCOs[Y_name], _X_SCOs['mean_predict'])

        precision, recall, threshold_pr = precision_recall_curve(_X_SCOs[Y_name], _X_SCOs['pred_pro'])
        fpr, tpr, threshold_roc = roc_curve(_X_SCOs[Y_name], _X_SCOs['pred_pro'])
        roc_auc = auc(fpr, tpr)
        average_precision = average_precision_score(_X_SCOs[Y_name], _X_SCOs['pred_pro'])
        ks_score=(tpr-fpr).max()

        self.log.CIF( ('%s TrainTesting parameters of the LRCV'%Y_name).center(45, '-') )
        self.log.CIF( 'alpha: %s, beta: %s' %(alpha, beta) )
        self.log.NIF( classification_report(y_true=_X_SCOs[Y_name], y_pred=_X_SCOs['mean_predict']) )
        self.log.CIF("LG TT R2_score: %s"% _R2_score)
        self.log.CIF("LG TT accuracy: %s"% _accuracy)
        self.log.CIF("LG TT Roc_Auc : %s" % roc_auc)
        self.log.CIF("LG TT ks_score: %s"% ks_score)
        self.log.CIF("LG TT average_precision: %s" % average_precision)
        self.log.CIF(45 * '-')

        return(_X_WOEs, _X_SCOs)

    def LR_Predict(self, pCf_xy, pXa, pYi, _i_MODEL):
        _X_Features = _i_MODEL['Features']
        clf         = _i_MODEL['Best_mod']
        F_WOEIV     = _i_MODEL['F_WOEIV']
        beta        = _i_MODEL['beta']
        alpha       = _i_MODEL['alpha']

        pYi_DF = pCf_xy[pYi]
        pXa_DF = pCf_xy[_X_Features]

        X_predict, y_predict = self._PD_WOEIV(F_WOEIV, pXa_DF, pYi_DF )

        intercept = clf.intercept_
        coef_s    = clf.coef_
        base_sore = alpha - beta * intercept
        Y_tespred = clf.predict(X_predict)
        Y_tespred_pro = clf.predict_proba(X_predict)

        X_Scor = X_predict*coef_s*beta
        X_Scor['predict'] = Y_tespred
        X_Scor['pred_pro']= Y_tespred_pro[:,1]
        X_Scor['_Scores'] = X_Scor.sum(1) + base_sore

        return(X_Scor[_X_Features], X_Scor[['predict', 'pred_pro', '_Scores']], X_predict)

    def Pro_odds(self):
        (prob_, basepoints, pdo )  = self.array
        beta       = pdo/math.log(2)
        baseodds   = prob_ / (1- prob_)
        alpha = basepoints + beta * math.log(baseodds)
        return (alpha)

class CRDScore():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        self.arg.output = '%s/09CRDScore/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs(os.path.dirname(self.arg.output), exist_ok=True)

    def Scoring(self):
        (group, RYa, CYa, Xa, Xg) = OpenM(self.arg, self.log).openg()
        YType  =  group[ (group.Variables.isin( RYa + CYa)) ][['Variables','Type']]

        for Yi,Typi in YType.values:
            if self.arg.pca:
                Fsltfile = '%s/02FeatureSLT/%s%s_Class_Regress_FeatureSLT.PCA.xls' %(self.arg.outdir, self.arg.header, Yi )
            else :
                Fsltfile = '%s/02FeatureSLT/%s%s_Class_Regress_FeatureSLT.Data.xls'%(self.arg.outdir, self.arg.header, Yi )

            DFall = Openf(Fsltfile, index_col=0).openb()
            _Xi   = list( set(DFall.columns)- set([Yi]) )

            self.log.CIF( ('%s: Credit Scoring'%Yi).center(45, '*') )
            self.log.CIF( '%s Value Counts:\n%s' %(Yi, DFall[Yi].value_counts().to_string()) )

            Data_WOE, Data_Scores = Binning(self.arg, self.log).LRScore( DFall, _Xi, Yi )
            Data_WOE['_Scores'] = Data_Scores['_Scores']

            Openf('%s%s_Class_TrainTest_woe.xls'%(self.arg.output,   Yi), (Data_WOE), index=True, index_label=0).openv()
            Openf('%s%s_Class_TrainTest_score.xls'%(self.arg.output, Yi), (Data_Scores), index=True, index_label=0).openv()

            ClusT('%s%s_Class_TrainTest_LR_WOE_complete.pdf'%(self.arg.output, Yi) ).Plot_heat( Data_WOE[_Xi] , Data_WOE[[Yi, '_Scores']], Xg, median=60, method='complete')
            ClusT('%s%s_Class_TrainTest_LR_WOE_average.pdf' %(self.arg.output, Yi) ).Plot_heat( Data_WOE[_Xi] , Data_WOE[[Yi, '_Scores']], Xg, median=60, method='average')

            ClusT('%s%s_Class_TrainTest_LR_Scores_complete.pdf'%(self.arg.output, Yi) ).Plot_heat( Data_Scores[_Xi], Data_Scores[[Yi, '_Scores']], Xg, median=60, method='complete')
            ClusT('%s%s_Class_TrainTest_LR_Scores_average.pdf' %(self.arg.output, Yi) ).Plot_heat( Data_Scores[_Xi], Data_Scores[[Yi, '_Scores']], Xg, median=60, method='average')

            ROCPR( '%s%s_Class_TrainTest_LR_ROC_KS.pdf'%(self.arg.output, Yi) ).ROC_KS( Data_Scores[Yi], Data_Scores['pred_pro'], Data_Scores['mean_predict'], 'LogsticR')
            self.log.CIF( ('%s: Credit Scoring Finish'%Yi).center(45, '*') )
