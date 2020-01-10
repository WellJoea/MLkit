from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix as cm
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, LeavePOut, 
                                     LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold)
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             classification_report, make_scorer, balanced_accuracy_score,
                             precision_recall_curve, mean_squared_error,
                             roc_curve, auc, r2_score, mean_absolute_error,
                             average_precision_score, explained_variance_score)
from sklearn.preprocessing import  OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize
from sklearn_pandas import DataFrameMapper, cross_val_score

from scipy.stats import pearsonr, stats
import joblib
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)

from .MLOpenWrite import OpenM, Openf
from .MLEstimators import ML
from .MLPlots import ClusT, ROCPR, MPlot

class Processing():
    def __init__(self, arg, log, *array, score= None , model='XGB', Type='C', **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.score = score
        self.model = model
        self.dicts = dicts
        self.Type  = Type
        self.SSS   = StratifiedShuffleSplit(n_splits=10, test_size=self.arg.GStestS , random_state=1)
        self.estimator =  ML(self.model, Type=self.Type).GetPara().estimator
        self.parameters = ML(self.model, Type=self.Type, SearchCV=self.arg.SearchCV ).GetPara().parameters
    def CrossvalidationSplit(self, Xdf, Ydf, Type='SSA', random_state = 20 ):
        if Type == 'SSS':
            CVS = StratifiedShuffleSplit(n_splits =self.arg.crossV,
                                         test_size=self.arg.testS,
                                         random_state=random_state).split(Xdf,Ydf)
        elif Type == 'SKF':
            CVS = StratifiedKFold(n_splits=self.arg.crossV,
                                  random_state=random_state).split(Xdf,Ydf)
        elif Type == 'RSKF':
            CVS = RepeatedStratifiedKFold(n_splits = 4,
                                          n_repeats= 3,
                                          random_state=random_state).split(Xdf,Ydf)
        elif Type == 'LOU':
            CVS = LeaveOneOut().split(Xdf)
        elif Type == 'LPO':
            CVS = LeavePOut(self.arg.leavP).split(Xdf)

        elif Type == 'SSA':
            CVS = StratifiedShuffleSplit(n_splits =self.arg.crossV,
                                         test_size=self.arg.testS,
                                         random_state=random_state).split(Xdf,Ydf)
        elif Type == 'RKF':
            CVS = RepeatedKFold(n_splits = 4,
                                n_repeats= 3,
                                random_state=random_state).split(Xdf)
        all_test = []
        all_split= []
        for train_index, test_index in CVS:
            all_test.extend(test_index)
            all_split.append( [train_index.tolist(), test_index.tolist()])

        if Type == 'SSA':
            if len(set(all_test))< len(Ydf):
                SFK = StratifiedKFold(n_splits=round(1/self.arg.testS) ,
                                      random_state=random_state).split(Xdf,Ydf)
                all_add = [ [tr.tolist(), te.tolist()] for tr, te in SFK]
                all_split += all_add

        return(all_split)

    def Clfing(self, X_train, X_test, y_train, y_test, cv_method):
        if ( len(set(y_train)) >2 ) & (self.model in ['XGB']):
            self.estimator = self.estimator.set_params(objective='multi:softprob')

        if self.model in ['XGB']:
            _n_jobs = 20
            if 'scale_pos_weight' in self.parameters.keys():
                pos_weight = y_train.value_counts(normalize=True)
                self.parameters['scale_pos_weight'] += ( (1-pos_weight)/pos_weight ).to_list()
                self.parameters['scale_pos_weight']  = list(set( self.parameters['scale_pos_weight'] ))
        else:
            _n_jobs = self.arg.n_job

        ## add hyperopt Spearmint hyperparameter
        if self.arg.SearchCV == 'GSCV':
            clf = GridSearchCV(self.estimator, self.parameters,
                               n_jobs=_n_jobs,
                               cv=cv_method,
                               scoring=self.score,
                               error_score = np.nan,
                               return_train_score=True,
                               refit = True,
                               iid=True)
        elif self.arg.SearchCV == 'RSCV':
            clf = RandomizedSearchCV(self.estimator, self.parameters,
                                     n_jobs=_n_jobs,
                                     cv=cv_method,
                                     n_iter = self.arg.n_iter,
                                     scoring=self.score,
                                     return_train_score=True,
                                     iid=True,
                                     refit = True,
                                     error_score='raise')
        if self.model in ['XGB_']:
            clf.fit(X_train, y_train,
                    #eval_metric=["error", "logloss"],
                    #eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_set=[(X_test, y_test)],
                    eval_metric= 'logloss',
                    early_stopping_rounds=15,
                    verbose=False)
        else:
            clf.fit(X_train, y_train)

        if self.model in ['MNB','BNB'] :
            importances= np.exp(clf.best_estimator_.coef_)
            self.log.CWA('Note: Use coef_ exp values as the feature importance of the estimator %s.'%self.model)
        elif self.model in ['CNB'] :
            if len(set(y_train)) ==2:
                importances= np.exp(-clf.best_estimator_.feature_log_prob_)[1]
            else:
                importances= np.exp(-clf.best_estimator_.feature_log_prob_)
            self.log.CWA('Note: Use feature_log_prob_ negative exp values as the feature importance of the estimator %s.'%self.model)
        elif self.model in ['GNB'] :
            if len(set(y_train)) ==2:
                importances= clf.best_estimator_.theta_[1]
            else:
                importances= clf.best_estimator_.theta_
            self.log.CWA('Note: Use theta_ values as the feature importance of the estimator %s.'%self.model)
        elif self.model in ['MLP'] :
            def collapse(coefs):
                Coefs = coefs[0]
                for b in coefs[1:]:
                    #A = Coefs
                    #Coefs = [ [np.abs(A[i] * b[:,j]).sum() for j in range(b.shape[1]) ] for i in range(A.shape[0]) ]
                    #Coefs = np.array(Coefs)
                    Coefs = Coefs.dot(b)
                return(Coefs/Coefs.sum(0))
            importances = collapse(clf.best_estimator_.coefs_).T
            self.log.CWA('Note: Use coefs_ weighted average as the feature importance of the estimator %s.'%self.model)
        elif self.model in ['SVM', 'SVMrbf', 'nuSVMrbf']:
            dot_coef_ = clf.best_estimator_.dual_coef_.dot( clf.best_estimator_.support_vectors_ )
            importances = (1-np.exp(-dot_coef_)) / (1+np.exp(-dot_coef_))
            self.log.CWA('Note: Use exp dot of support_vectors_ and dual_coef_ values as the feature importance of the estimator %s.'%self.model)
        else:
            for i in ['feature_importances_', 'coef_']:
                try:
                    importances= eval('clf.best_estimator_.%s'%i)
                    self.log.CWA('Note: Use %s as the feature importance of the estimator %s.'%(i,self.model))
                    break
                except AttributeError:
                    importances = []
                    self.log.CWA('Note: Cannot find the feature importance attributes of estimator %s.'%i)

        df_import = pd.DataFrame(np.array(importances).T)
        if not df_import.empty :
            df_import.index=X_train.columns
            df_import = df_import[(df_import != 0).any(1)]

        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        return(clf, df_import)

    def Add_LR(self, X_train, X_test, y_train, y_test ):
        X_all_new = np.concatenate((X_train, X_test), axis=0)
        X_all_new = X_all_new.astype(np.int32)
        OHE = OneHotEncoder(categories='auto')
        X_trans = OHE.fit_transform(X_all_new)

        X_train_ = X_trans[:X_train.shape[0],:]
        X_test_  = X_trans[X_train.shape[0]:,:]

        _LR_P, _LR_E = ML( self.arg.LRmode,  SearchCV = 'GSCV').Classification()

        clf_lr = GridSearchCV(_LR_E, _LR_P,
                               cv=self.SSS,
                               n_jobs=self.arg.n_job,
                               return_train_score=True,
                               #scoring='accuracy',
                               #error_score = np.nan,
                               iid =True,)
        clf_lr.fit(X_train_, y_train)

        Y_trai_lr = clf_lr.best_estimator_.predict(X_train_)
        Y_pred_lr = clf_lr.best_estimator_.predict(X_test_)
        y_prob_lr = clf_lr.best_estimator_.predict_proba(X_test_)
        Y_test_df = y_test.to_frame()
        Y_test_df['LR_predict'] = Y_pred_lr
        Y_test_df['LR_predict_proba_1'] = y_prob_lr[:,1]

        Y_trian_accuracy = accuracy_score(y_train, Y_trai_lr)
        Y_test_accuracy  = accuracy_score(y_test, Y_pred_lr)
        Y_test_R2_score  = r2_score(y_test, Y_pred_lr)

        precision, recall, threshold_pr = precision_recall_curve(y_test, y_prob_lr[:, 1])
        fpr, tpr, threshold_roc = roc_curve(y_test, y_prob_lr[:, 1])
        average_precision = average_precision_score(y_test, y_prob_lr[:, 1])
        roc_auc = auc(fpr, tpr)

        self.log.CIF( ('%s %s Model Fitting and hyperparametering'% (y_train.name, self.arg.LRmode)).center(45, '-') )
        self.log.CIF( '%s best parameters: %s' %(self.arg.LRmode, clf_lr.best_params_) )
        self.log.CIF( '%s best score: %s' %(self.arg.LRmode, clf_lr.best_score_) )
        self.log.NIF( classification_report(y_true=y_test, y_pred=Y_pred_lr) )
        self.log.CIF( "%s _train accuracy: %s" %(self.arg.LRmode, Y_trian_accuracy) )
        self.log.CIF( "%s _test accuracy: %s" %(self.arg.LRmode,Y_test_accuracy) )
        self.log.CIF( "%s _test R2_score: %s" %(self.arg.LRmode, Y_test_R2_score) )
        self.log.CIF( "%s _test Roc_Auc: %s" %(self.arg.LRmode, roc_auc) )
        self.log.CIF( "%s _test average_precision: %s" %(self.arg.LRmode, average_precision) )
        self.log.CIF(45 * '-')

        return (Y_test_df, clf_lr.best_estimator_)

    def LearNing_C(self, X_train, X_test, y_train, y_test):
        Y_name = y_train.name
        clf, df_import = self.Clfing(X_train, X_test, y_train, y_test, self.SSS)

        Y_trapred = clf.predict(X_train)
        Y_tespred = clf.predict(X_test)
        Y_trian_accuracy = accuracy_score(y_train, Y_trapred)
        Y_test_accuracy  = accuracy_score(y_test, Y_tespred)
        Y_test_R2_score  = r2_score(y_test, Y_tespred)

        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        clas_rep = classification_report(y_true=y_test, y_pred=Y_tespred, output_dict=dict)
        '''
        def _predict_proba_lr(self, X):
            """
            Probability estimation for OvR logistic regression.
            Positive class probabilities are computed as
            1. / (1. + np.exp(-self.decision_function(X)));
            multiclass is handled by normalizing that over all classes.
            """
            prob = self.decision_function(X)
            expit(prob, out=prob)
            if prob.ndim == 1:
                return np.vstack([1 - prob, prob]).T
            else:
                # OvR normalization, like LibLinear's predict_probability
                prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
        '''

        try:
            Y_tespred_pro = clf.best_estimator_.predict_proba(X_test)
            self.log.CWA('Note: %s use _predict_proba as predict_proba'%self.model)
        except AttributeError:
            Y_tespred_pro = clf.best_estimator_._predict_proba_lr(X_test)
            self.log.CWA('Note: %s use _predict_proba_lr as predict_proba'%self.model)
            self.log.CWA('Note: LinearSVC, SGB use _predict_proba_lr based on decision_function as predict_proba')
        except:
            '''clf=CalibratedClassifierCV(clf.best_estimator_,cv=10, method="sigmoid")
            clf.fit(X_train, y_train)
            Y_trapred = clf.predict(X_train)
            Y_tespred = clf.predict(X_test)
            Y_tespred_pro    = clf.predict_proba(X_test)'''
            self.log.CWA('predict_proba AttributeError!')

        y_true = label_binarize(y_test, classes= range(Y_tespred_pro.shape[1]))
        if y_true.shape[1] ==1:
            y_true_0 = label_binarize(y_test, classes=[1,0])
            y_true_1 = label_binarize(y_test, classes=[0,1])
            y_true = np.c_[y_true_0,y_true_1]
        average_precision, roc_auc = [], []
        for i in range(Y_tespred_pro.shape[1]):
            precision, recall, threshold_pr = precision_recall_curve(y_true[:,i], Y_tespred_pro[:, i])
            fpr, tpr, threshold_roc = roc_curve(y_true[:,i], Y_tespred_pro[:, i])
            average_precision.append(average_precision_score( y_true[:, i], Y_tespred_pro[:, i]) )
            roc_auc.append( auc(fpr, tpr) )

        metrics_pvalues = [['Model', 'Y_trian_accuracy', 'Y_test_accuracy', 'Y_test_R2_score', 'roc_auc'],
                           [self.model ,Y_trian_accuracy, Y_test_accuracy, Y_test_R2_score, roc_auc]]
        metrics_pvalues = pd.DataFrame(metrics_pvalues[1:], columns=metrics_pvalues[0])
        ROC_PR          = [roc_auc, Y_test_accuracy, average_precision]

        self.log.CIF( ('%s %s Model Fitting and hyperparametering'% (Y_name, self.model)).center(45, '-') )
        self.log.CIF( '%s best parameters: \n%s' %(self.model, clf.best_params_) )
        self.log.CIF( '%s best score: %s' %(self.model, clf.best_score_) )
        self.log.NIF( classification_report(y_true=y_test, y_pred=Y_tespred) )
        self.log.CIF( "%s _weight: \n%s" % (self.model, df_import) )
        self.log.CIF( "%s _train accuracy: %s" %(self.model, Y_trian_accuracy) )
        self.log.CIF( "%s _test accuracy: %s" %(self.model,Y_test_accuracy) )
        self.log.CIF( "%s _test R2_score: %s" %(self.model, Y_test_R2_score) )
        self.log.CIF( "%s _test Roc_Auc: %s" %(self.model, roc_auc) )
        self.log.CIF( "%s _test average_precision: %s" %(self.model, average_precision) )
        self.log.CIF(45 * '-')

        Best_Model = { 'ML' : clf.best_estimator_,
                       'Features' : X_train.columns.tolist()
                     }

        _Y_TEST = pd.DataFrame(Y_tespred_pro,
                               columns = ['Prob_%s'%x for x in range(Y_tespred_pro.shape[1]) ],
                               index = y_test.index )
        _Y_TEST['Predict'] = Y_tespred
        _Y_TEST['True']    = y_test

        if len(set(y_train)) == 2 and self.model in ['GBDT', 'XGB', 'RF']:
            if  self.model in ['GBDT']:
                X_train_New = clf.best_estimator_.apply(X_train)[:,:,0]
                X_test_New  = clf.best_estimator_.apply(X_test)[:,:,0]
            elif self.model in ['XGB', 'RF']:
                X_train_New = clf.best_estimator_.apply(X_train)
                X_test_New  = clf.best_estimator_.apply(X_test)
            LR_test_pre, LR_ML = self.Add_LR( X_train_New, X_test_New, y_train, y_test )
            Best_Model['X_train'] = X_train_New
            Best_Model['LR_ML']   = LR_ML
            _Y_TEST[['LR_predict', 'LR_predict_proba_1']] = LR_test_pre[['LR_predict', 'LR_predict_proba_1']]

        return (_Y_TEST, df_import, metrics_pvalues, ROC_PR, Best_Model)

    def Evaluate_C(self, y_ture, y_pred, y_prob, model):
        _accuracy  = accuracy_score(y_ture, y_pred)
        _R2_score  = r2_score(y_ture, y_pred)

        y_true_lb = label_binarize(y_ture, classes= range(y_prob.shape[1]))
        if y_true_lb.shape[1] ==1:
            y_true_0 = label_binarize(y_ture, classes=[1,0])
            y_true_1 = label_binarize(y_ture, classes=[0,1])
            y_true_lb= np.c_[y_true_0,y_true_1]

        average_precision, roc_auc = [], []
        for i in range(y_prob.shape[1]):
            precision, recall, threshold_pr = precision_recall_curve(y_true_lb[:,i], y_prob[:, i])
            fpr, tpr, threshold_roc = roc_curve(y_true_lb[:,i], y_prob[:, i])
            average_precision.append(average_precision_score( y_true_lb[:,i], y_prob[:, i]) )
            roc_auc.append( auc(fpr, tpr) )
        ROC_PR = [roc_auc, _accuracy, average_precision]

        self.log.CIF( ('%s %s Model Predicting Parameters:'% (y_ture.name, model)).center(45, '-') )
        self.log.NIF( classification_report(y_true=y_ture, y_pred=y_pred) )
        self.log.CIF( "%s _predict accuracy: %s" %(model, _accuracy) )
        self.log.CIF( "%s _predict R2_score: %s" %(model, _R2_score) )
        self.log.CIF( "%s _predict Roc_Auc: %s"  %(model, roc_auc) )
        self.log.CIF( "%s _predict average_precision: %s" %(model, average_precision) )
        self.log.CIF(45 * '-')
        return (ROC_PR)

    def Predicting_C(self, pCf_xy, pXa, pYi , Best_Model):
        pXa_DF = pCf_xy[pXa]
        pYi_DF = pCf_xy[pYi]

        _Y_predict , _ROC_PR, _ROC_PR_LR = [], [], []
        for i, _model_i in enumerate(Best_Model):
            _X_Features = _model_i['Features']
            _clf        = _model_i['ML']
            _X_pred_DF  = pXa_DF[_X_Features]
            Y_predict   = _clf.predict(_X_pred_DF)

            try:
                Y_predict_pro = _clf.predict_proba(_X_pred_DF)
            except AttributeError:
                Y_predict_pro = _clf._predict_proba_lr(_X_pred_DF)
                self.log.CWA('Note: LinearSVC, SGB_hinge use _predict_proba_lr as predict_proba')
            except AttributeError:
                Y_tespred_pro = _clf.decision_function(_X_pred_DF)
                Y_tespred_pro = (Y_tespred_pro - Y_tespred_pro.min()) / (Y_tespred_pro.max() - Y_tespred_pro.min())
                self.log.CWA('Note: LinearSVC, SGB_hinge also can use decision_function as predict_proba')
            except:
                self.log.CWA('Warning: predict_proba AttributeError!')

            _Y_PRED = pd.DataFrame(Y_predict_pro,
                                   columns = ['Prob_%s'%x for x in range(Y_predict_pro.shape[1]) ],
                                   index   = _X_pred_DF.index )
            _Y_PRED['Predict'] = Y_predict
            _Y_PRED['True']    = pYi_DF

            if not pYi_DF.isnull().any():
                roc_pr_ = self.Evaluate_C(pYi_DF, Y_predict, Y_predict_pro, self.model )
                _ROC_PR.append(roc_pr_)

            if 'LR_ML' in _model_i.keys():
                clf_lr  = _model_i['LR_ML']
                X_train = _model_i['X_train']

                if  self.model in ['GBDT']:
                    X_pred  = _clf.apply(_X_pred_DF)[:,:,0]
                elif self.model in ['XGB', 'RF']:
                    X_pred  = _clf.apply(_X_pred_DF)

                X_all_new = np.concatenate((X_train, X_pred), axis=0)
                X_all_new = X_all_new.astype(np.int32)
                OHE = OneHotEncoder(categories='auto')
                X_trans = OHE.fit_transform(X_all_new)
                X_pred  = X_trans[X_train.shape[0]:,:]

                Y_pred_lr = clf_lr.predict(X_pred)
                Y_prob_lr = clf_lr.predict_proba(X_pred)

                _Y_PRED['LR_predict'] = Y_pred_lr
                _Y_PRED['LR_predict_proba_1'] = Y_prob_lr[:,1]

                if not pYi_DF.isnull().any():
                    roc_pr_lr = self.Evaluate_C(pYi_DF, Y_pred_lr, Y_prob_lr, '+LR' )
                    _ROC_PR_LR.append(roc_pr_lr)
            _Y_predict.append(_Y_PRED)

        return (_Y_predict , _ROC_PR, _ROC_PR_LR)

    def LearNing_R(self,):
        X_train, X_test, y_train, y_test = self.array
        Y = y_train.name
        clf = GridSearchCV(self.estimator, self.parameters, n_jobs=self.arg.n_job, cv=10,return_train_score =True,iid=True)
        clf.fit(X_train, y_train)
        if 'joblibR' in self.dicts.keys():
            joblib.dump(clf.best_estimator_, '%sRegress_%s_%s_best_estimator.pkl'%(self.arg.outdir,Y,self.model), compress = 1)

        try:
            importances= clf.best_estimator_.feature_importances_
            df_import = pd.DataFrame(np.array(importances).T,columns=[self.model + '_import'],index=X_train.columns)
            #.sort_values(by=[self.model + '_import'],ascending=[False])
        except AttributeError:
            importances= clf.best_estimator_.coef_
            df_import = pd.DataFrame(np.array(importances).T,columns=[self.model + '_import'],index=X_train.columns)
            #.sort_values(by=[self.model + '_import'],ascending=[False])
        except AttributeError:
            importances= clf.best_estimator_.coefs_
            df_import = pd.DataFrame(np.array(importances).T)
            print(df_import)
        except AttributeError:
            print('aaaaaaaaaa')
        df_import = df_import[df_import[self.model + '_import'] != 0]

        #cross_validate cross_val_score cross_val_predict
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        score = clf.score(X_test, y_test)
        Y_trapred = clf.predict(X_train)
        Y_tespred = clf.predict(X_test)

        model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
        tmp_score = [m(y_test, Y_tespred)  for m in model_metrics_name]
        tmp_score.append( pearsonr(y_test, Y_tespred)[0] )

        print(80 * '-')
        print('The parameters of the best model are: ')
        print(self.model, clf.best_params_, clf.best_score_)
        print(self.model, "_weight:", df_import)
        print(self.model, "_test cor:", tmp_score)
        print(self.model, "_test score:", score)
        print(80 * '-')
        return (tmp_score, df_import, Y_tespred)

    def Predict_R(self):
        (Cf_xy, Xa, Yi ) =self.array
        Y = Yi.name
        clf = joblib.load('%sClass_%s_%s_best_estimator.pkl'%(self.arg.outdir,Y,self.model))
        df_import = Openf('%sClass_%s_%s_best_estimator.imp'%(self.arg.outdir,Y,self.model) , index_col='features', index=True).openb()
        X_predict = Cf_xy[df_import.index]
        All_Y_predict = []
        All_Y_predict_pro = []
        for i in range(10):
            Y_predict = clf.predict(X_predict)
            Y_predict_pro    = clf.predict_proba(X_predict)

            if not Yi.isnull().any():
                Y_predict_accuracy  = accuracy_score(Yi, Y_predict)
                Y_predict_R2_score  = r2_score(Yi, Y_predict)
                Y_predict_pro       = clf.predict_proba(X_predict)
                Y_predict_pro_val   = Y_predict_pro[:, 1]
                fpr,tpr,threshold   = roc_curve(Yi, Y_predict_pro_val)
                roc_auc = auc(fpr,tpr)
                metrics_pvalues = [['Model', 'Y_predict_accuracy', 'Y_predict_R2_score', 'roc_auc'],
                                   [self.model , Y_predict_accuracy, Y_predict_R2_score, roc_auc]]
                metrics_pvalues = pd.DataFrame(metrics_pvalues[1:], columns=metrics_pvalues[0])
                ROC_list        = [fpr, tpr, roc_auc, Y_predict_accuracy, self.model]

class Modeling():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        self.arg.output = '%s/03ModelFit/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs(os.path.dirname(self.arg.output), exist_ok=True)

    def R_Model(self, Xdf, Ydf,  Xg, Type= 'R'):
        pass

    def C_Model(self, Xdf, Ydf, Xg, Type= 'C'):
        All_test_pd, All_import, All_pvalues, All_score, All_parameter  = [], [], [], [], []

        CVM = Processing(self.arg, self.log).CrossvalidationSplit(Xdf, Ydf, Type= self.arg.CVfit)
        for train_index, test_index in CVM:
            X_train, X_test = Xdf.iloc[train_index, :], Xdf.iloc[test_index, :]
            try:
                y_train, y_test = Ydf.iloc[train_index, :], Ydf.iloc[test_index, :]
            except pd.core.indexing.IndexingError:
                y_train, y_test = Ydf[train_index], Ydf[test_index]

            _Y_TEST, df_import, metrics_pvalues, ROC_PR, Best_Model = Processing(
                                self.arg, self.log, model=self.arg.model, Type = Type ).LearNing_C( X_train, X_test, y_train, y_test )

            All_test_pd.append(_Y_TEST)
            All_pvalues.append(metrics_pvalues)
            All_score.append(ROC_PR)
            All_parameter.append(Best_Model)
            if not df_import.empty:
                All_import.append(df_import)

       ### All_pvalues
        All_pvalues = pd.concat(All_pvalues,  axis=0).reset_index(drop=True)
        Openf('%s%s_Class_metrics_pvalues.xls'%(self.arg.output, Ydf.name), (All_pvalues), index=True, index_label='cv_times').openv()

       ### All_parameter
        joblib.dump(All_parameter,
               '%s%s_Class_best_estimator.pkl' %(self.arg.output, Ydf.name), compress=1)

       ### All_score
        All_score  = pd.DataFrame(All_score, columns = ['Roc_Auc', 'accuracy', 'average_precision'])
        All_score['mean_Roc_Auc'] = list(map(lambda x: np.mean(x), All_score['Roc_Auc']))
        All_score['ROC_ACC'] = list(map(lambda x: ' '.join( map(lambda i: '%.2f' % i, x[0]+[x[1]])), All_score[['Roc_Auc', 'accuracy']].values))
        Best_index = All_score[['mean_Roc_Auc', 'accuracy']].sum(axis=1).idxmax()

        ### All_import
        if All_import:
            All_import = pd.concat(All_import, axis=1,sort=False).fillna(0)
            column = sorted(set(All_import.columns))
            All_import = All_import[column]
            for i in column:
                All_import['%s_mean'%i]  = All_import[i].mean(axis=1)
                All_import['%s_std'%i]   = All_import[i].std(axis=1)
                All_import['%s_median'%i]= All_import[i].median(axis=1)
            All_import.sort_values(by=['%s_mean'%i for i in column], ascending=[False]*len(column), inplace=True, axis=0)

            Openf('%s%s_Class_features_importance.xls' %(self.arg.output, Ydf.name), (All_import)).openv()
            MPlot('%s%s_Class_Features_importance.lines.pdf' %(self.arg.output, Ydf.name)).Feature_Import( All_score['ROC_ACC'].tolist(), Best_index,  All_import, self.arg.model, Ydf.name )
            MPlot('%s%s_Class_Features_importance.boxI.pdf'  %(self.arg.output, Ydf.name)).Feature_Import_box( All_import, Xg, Ydf.name, self.arg.model, sort_by_group=False )
            MPlot('%s%s_Class_Features_importance.boxII.pdf' %(self.arg.output, Ydf.name)).Feature_Import_box( All_import, Xg, Ydf.name, self.arg.model, sort_by_group=True)

        ### All_test_
        #True Predict Prob_ LR_predict LR_predict_proba_1  # mode_predict  mean_predict  LR_mode_predict LR_mean_predict
        All_test_ = pd.concat(All_test_pd,axis=0)
        All_test = All_test_.groupby([All_test_.index]).mean()
        All_test_pro = All_test.filter(regex=("^Prob_"))
        All_test_pro.columns = [ int(x.replace('Prob_','')) for x in All_test_pro.columns ]

        All_test['mode_predict'] = All_test_['Predict'].groupby([All_test_.index]).apply(lambda x : x.mode(0)[0])
        All_test['mean_predict'] = All_test_pro.idxmax(1)

        ROCPR('%s%s_Class_Test_ROC.pdf'%(self.arg.output, Ydf.name)).ROC_CImport(
            All_test['True'],  All_test_pro, All_test['mean_predict'], self.arg.model )
        ROCPR('%s%s_Class_Test_PR.pdf' %(self.arg.output, Ydf.name)).PR_CImport(
            All_test['True'],  All_test_pro, All_test['mean_predict'], self.arg.model)

        if len(set(Ydf.values)) == 2:
            ROCPR( '%s%s_Class_TrainTest_CV_ROC_curve.pdf' % (self.arg.output, Ydf.name) ).ROC_Import( All_test_pd, Best_index, self.arg.model, Ydf.name)
            ROCPR( '%s%s_Class_TrainTest_CV_PR_curve.pdf'  % (self.arg.output, Ydf.name) ).PR_Import(  All_test_pd, Best_index, self.arg.model, Ydf.name)
        else:
            ROCPR( '%s%s_Class_TrainTest_CV_ROC_curve.pdf' % (self.arg.output, Ydf.name) ).ROC_MImport(All_test_pd, Best_index, self.arg.model, Ydf.name)
            ROCPR( '%s%s_Class_TrainTest_CV_PR_curve.pdf'  % (self.arg.output, Ydf.name) ).PR_MImport( All_test_pd, Best_index, self.arg.model, Ydf.name)

        ### + LR
        if 'LR_predict' in  All_test.columns:
            All_test['LR_mode_predict']  =  All_test['LR_predict'].apply( lambda x : 1 if x >0.5 else 0 )
            All_test['LR_mean_predict']  =  All_test['LR_predict_proba_1'].apply( lambda x : 1 if x >0.5 else 0 )
            ROCPR( '%s%s_Class_+LR_Test_ROC.pdf'%(self.arg.output, Ydf.name) ).ROC_CImport(All_test['True'],  All_test['LR_predict_proba_1'], All_test['LR_mean_predict'], self.arg.model + '+LR')
            ROCPR( '%s%s_Class_+LR_Test_ROC.pdf'%(self.arg.output, Ydf.name) ).PR_CImport( All_test['True'],  All_test['LR_predict_proba_1'], All_test['LR_mean_predict'], self.arg.model + '+LR')
            ROCPR( '%s%s_Class_+LR_Test_ROC_KS.pdf'%(self.arg.output, Ydf.name) ).ROC_KS(  All_test['True'],  All_test['LR_predict_proba_1'], All_test['LR_mean_predict'], self.arg.model + '+LR')

            All_dt = [ [self.arg.model, All_test['True'], All_test['Prob_1'], All_test['mean_predict'] ],
                       [self.arg.model + '+LR', All_test['True'], All_test['LR_predict_proba_1'] , All_test['LR_mean_predict'] ],
                     ]
            ROCPR( '%s%s_Class_+LR_Test_ROC_COMpare.pdf'%(self.arg.output, Ydf.name) ).ROC_COMpare( All_dt, Ydf.name )
            ROCPR( '%s%s_Class_+LR_Test_PR_COMpare.pdf' %(self.arg.output, Ydf.name) ).PR_COMpare( All_dt, Ydf.name )

        Openf('%s%s_Class_Test_summary.xls'%(self.arg.output, Ydf.name), (All_test), index=True, index_label=0).openv()
        _All_Y  = All_test[['True', 'mean_predict']].copy()
        if 'LR_predict' in All_test.columns:
            _All_Y[['LR_predict','LR_Score']]  = All_test[['LR_mean_predict', 'LR_predict_proba_1']]
        ClusT('%s%s_Class_Test_Feature.Data_complete.pdf'%(self.arg.output, Ydf.name) ).Plot_heat( Xdf , _All_Y, Xg, method='complete')
        ClusT('%s%s_Class_Test_Feature.Data_average.pdf' %(self.arg.output, Ydf.name) ).Plot_heat( Xdf , _All_Y, Xg, method='average')

    def Fitting(self):
        (group, RYa, CYa, Xa, Xg) = OpenM(self.arg, self.log).openg()
        YType  =  group[ (group.Variables.isin( RYa + CYa)) ][['Variables','Type']]

        for Yi,Typi in YType.values:
            if self.arg.pca:
                Fsltfile = '%s/02FeatureSLT/%s%s_Class_Regress_FeatureSLT.PCA.xls' %(self.arg.outdir, self.arg.header, Yi )
            else :
                Fsltfile = '%s/02FeatureSLT/%s%s_Class_Regress_FeatureSLT.Data.xls'%(self.arg.outdir, self.arg.header, Yi )

            DFall = Openf(Fsltfile, index_col=0).openb()
            _Xi   = list( set(DFall.columns)- set([Yi]) )

            self.log.CIF( ('%s: Supervised MODELing'%Yi).center(45, '*') )
            self.log.NIF( '%s Value Counts:\n%s' %(Yi, DFall[Yi].value_counts().to_string()) )

            if Typi =='C':
                self.C_Model( DFall[_Xi], DFall[Yi], Xg, Type= Typi)
            elif Typi =='R':
                self.R_Model( DFall[_Xi], DFall[Yi], Xg, Type= Typi)
            self.log.CIF( ('%s: Supervised MODELing Finish'%Yi).center(45, '*') )
