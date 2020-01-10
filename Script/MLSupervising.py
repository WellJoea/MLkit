from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix as cm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             classification_report, make_scorer, balanced_accuracy_score,
                             precision_recall_curve, mean_squared_error, roc_auc_score, 
                             roc_curve, auc, r2_score, mean_absolute_error,
                             average_precision_score, explained_variance_score)
from sklearn.preprocessing import  OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.calibration import CalibratedClassifierCV

from scipy.stats import pearsonr, stats
from scipy.sparse import hstack, vstack
import joblib
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
import re
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)

from .MLOpenWrite import OpenM, Openf
from .MLEstimators import ML
from .MLPlots import ClusT, ROCPR, MPlot
from .MLUtilities import _predict_proba_lr, CrossvalidationSplit, Check_Label, Check_Binar

class Baseinit:
    def __init__(self, arg, log, score= None , MClass=2, Y_name='Group', model='XGB', Type='C', *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.score = score
        self.model = model
        self.dicts = dicts
        self.Type  = Type
        self.MClass= MClass
        self.Y_name= Y_name
class Processing(Baseinit):
    def CVSplit(self, Xdf, Ydf, CVt='SSA', random_state = None ):
        all_test = []
        all_split= []
        if self.arg.crossV ==1:
            all_split = [ [range(Xdf.shape[0]), range(Xdf.shape[0])] ]
        else:
            CVS = CrossvalidationSplit(n_splits=self.arg.crossV, test_size=self.arg.testS, CVt=CVt  , n_repeats=2, leavP=self.arg.leavP )
            SFA = CrossvalidationSplit(n_splits=self.arg.crossV, test_size=self.arg.testS, CVt='SFA', n_repeats=2, leavP=self.arg.leavP )
            for train_index, test_index in CVS.split(Xdf, Ydf):
                all_test.extend( test_index )
                all_split.append( [train_index.tolist(), test_index.tolist()])

            if CVt == 'SSA':
                if len(set(all_test))< len(Ydf):
                    all_add = [ [tr.tolist(), te.tolist()] for tr, te in SFA.split(Xdf, Ydf)]
                    all_split += all_add
        return(all_split)

    def Pdecorator(_func):
        def wrapper(self, *args, **kargs):
            if self.Type=='C':
                self.SSS = CrossvalidationSplit(n_splits=9, test_size=self.arg.GStestS, CVt='SSS')
            elif self.Type=='R':
                self.SSS = CrossvalidationSplit(n_splits=3, n_repeats= 3, CVt='RSKF')
            else:
                self.SSS = ''

            if 'Model'in kargs:
                self.model = kargs['Model']
                self.estimator =  ML(self.model, Type=self.Type).GetPara().estimator
                if self.arg.SearchCV:
                    self.parameters = ML(self.model, Type=self.Type, SearchCV=self.arg.SearchCV ).GetPara().parameters
                else:
                    self.parameters = {}
            return _func(self, *args, **kargs)
        return wrapper

    def Hyperparemet(self, X_train, X_test, y_train, y_test):
        self.log.CIF( 'hyperparameter optimization in the %s model......'%self.model )
        if ( self.MClass >2 ) & (self.model in ['XGB']):
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
        if not self.arg.SearchCV:
            clf = self.estimator
            self.log.CIF( 'decrepate hyperparameter optimization......' )
        elif self.arg.SearchCV == 'GSCV':
            clf = GridSearchCV(self.estimator, self.parameters,
                               n_jobs=_n_jobs,
                               cv=self.SSS,
                               scoring=self.score,
                               error_score = np.nan,
                               return_train_score=True,
                               refit = True)
            self.log.CIF( 'GSCV hyperparameter optimization......')
        elif self.arg.SearchCV == 'RSCV':
            clf = RandomizedSearchCV(self.estimator, self.parameters,
                                     n_jobs=_n_jobs,
                                     cv=self.SSS,
                                     n_iter = self.arg.n_iter,
                                     scoring=self.score,
                                     return_train_score=True,
                                     refit = True,
                                     error_score='raise')
            self.log.CIF( 'RSCV hyperparameter optimization......')

        if self.model in ['XGB_']:
            clf.fit(X_train, y_train,
                    #eval_metric=["error", "logloss"],
                    #eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_set=[(X_test, y_test)],
                    eval_metric= 'auc',
                    early_stopping_rounds=15,
                    verbose=False)
        else:
            clf.fit(X_train, y_train)

        if hasattr(clf, 'best_estimator_'):
            self.log.CIF( '%s best parameters in %s: \n%s' %(self.model, self.arg.SearchCV, clf.best_estimator_.get_params()) )
            return clf.best_estimator_
        else:
            self.log.CIF( '%s best parameters: \n%s' %(self.model, clf.get_params()) )
            return clf

    def Featurescoef(self, clf, index_=None):
        if self.model in ['MNB','BNB'] :
            importances= np.exp(clf.coef_)
            self.log.CWA('Note: Use coef_ exp values as the feature importance of the estimator %s.'%self.model)
        elif self.model in ['CNB'] :
            if self.MClass ==2:
                importances= np.exp(-clf.feature_log_prob_)[1]
            else:
                importances= np.exp(-clf.feature_log_prob_)
            self.log.CWA('Note: Use feature_log_prob_ negative exp values as the feature importance of the estimator %s.'%self.model)
        elif self.model in ['GNB'] :
            if self.MClass ==2:
                importances= clf.theta_[1]
            else:
                importances= clf.theta_
            self.log.CWA('Note: Use theta_ values as the feature importance of the estimator %s.'%self.model)
        elif self.model in ['MLP'] :
            def collapse(coefs):
                Coefs = coefs[0]
                for b in coefs[1:]:
                    Coefs = Coefs.dot(b)
                return(Coefs/Coefs.sum(0))
            importances = collapse(clf.coefs_).T
            self.log.CWA('Note: Use coefs_ weighted average as the feature importance of the estimator %s.'%self.model)
        elif self.model in ['SVM', 'SVMrbf', 'nuSVMrbf']:
            dot_coef_ = clf.dual_coef_.dot( clf.support_vectors_ )
            importances = (1-np.exp(-dot_coef_)) / (1+np.exp(-dot_coef_))
            self.log.CWA('Note: Use exp dot of support_vectors_ and dual_coef_ values as the feature importance of the estimator %s.'%self.model)
        else:
            for i in ['feature_importances_', 'coef_']:
                try:
                    importances= eval('clf.%s'%i)
                    self.log.CWA('Note: Use %s as the feature importance of the estimator %s.'%(i,self.model))
                    break
                except AttributeError:
                    importances = []
                    self.log.CWA('Note: Cannot find the feature importance attributes of estimator %s.'%i)

        df_import = pd.DataFrame(np.array(importances).T)
        if not df_import.empty :
            df_import.index=index_
            df_import = df_import[(df_import != 0).any(1)]
        return df_import

    def OneHot(self, _clf, _X_Data, _OHE='N'):
        if  self.model in ['GBDT']:
            _X_Trans = _clf.apply(_X_Data)[:,:,0]
        elif self.model in ['XGB', 'RF']:
            _X_Trans = _clf.apply(_X_Data)
        if _OHE == 'N':
            _OHE = OneHotEncoder(categories='auto')
            _OHE.fit(_X_Trans)
        _X_Trans = _OHE.transform( _X_Trans.astype(np.int32) )
        return [_X_Trans, _OHE]

    def GetPredt_C(self, _clf, X_matrix, y_label=None, mclass=False):
        if y_label:
            _clf.fit(X_matrix, y_label)

        _predict = _clf.predict(X_matrix)
        try:
            _proba = _clf.predict_proba(X_matrix)
        except AttributeError:
            _proba = _clf._predict_proba_lr(X_matrix)
            self.log.CWA('Note: LinearSVC, SGB use _predict_proba_lr based on decision_function as predict_proba in GridsearchCV.')
        except:
            _proba = _predict_proba_lr( _clf.decision_function(X_matrix) )
            self.log.CWA('predict_proba use sigmoid transversion based on decision_function!')

        if (mclass & mclass != _proba.shape[1]):
            raise Exception('The columns length of predict probability is wrong!') 
        return np.c_[_predict, _proba]

    def GetEvalu_C(self, y_true, y_predict, y_score, name=None):
        y_trueb = Check_Binar(y_true)

        Model_ = pd.Series({  
                    'Accuracy'  : accuracy_score(y_true, y_predict),
                    'Accuracy_B': balanced_accuracy_score(y_true, y_predict),
                    'F1_score'  : f1_score(y_true, y_predict),
                    'Precision' : precision_score(y_true, y_predict),
                    'Recall'    : recall_score(y_true, y_predict),
        })
        Model_['Roc_auc'] = round(roc_auc_score(y_trueb, y_score), 6)
        Model_['Precision_A'] = round(average_precision_score( y_trueb, y_score), 6)

        if y_score.shape[1] > 2:
            for i in range(y_score.shape[1]):
                Model_['Roc_auc_' + str(i)] = round(roc_auc_score(y_trueb[:,i], y_score[:, i]), 6)
                Model_['Precision_A_' + str(i)] = round(average_precision_score( y_trueb[:, i], y_score[:, i]), 6)

        #Report = classification_report(y_true=y_true, y_pred=y_predict, output_dict=False)
        Model_.name = name
        return Model_.sort_index()

    def GetSplit(self, trprpr, label=None):
        _m = [re.sub('^Pred','', _i) for _i in trprpr.columns if re.match('Pred', _i)]
        return pd.concat([ self.GetEvalu_C(trprpr['TRUE'],
                                trprpr['Pred'+_j],
                                trprpr[sorted([_n for _n in trprpr.columns if re.search('Prob'+_j, _n)])],
                                name= label + _j) 
                                for _j in _m 
                        ], axis=1)

    def GetScore_C(self, train_pre, test_pre):
        return pd.concat( (self.GetSplit(train_pre, label='Train'),
                            self.GetSplit(test_pre, label='Test')), axis=1)

    @Pdecorator
    def GetFit_C(self, X_train, X_test, y_train, y_test, Model=None):
        clf_G = self.Hyperparemet(X_train, X_test, y_train, y_test)

        clf_C = CalibratedClassifierCV(clf_G, cv=self.SSS, method=self.arg.calibme)
        clf_C.fit(X_train, y_train)

        try:
            X_ = pd.concat( (X_train, X_test), axis=0)
        except TypeError:
            X_ = vstack( (X_train, X_test) )

        y_ = pd.concat((y_train, y_test), axis=0)

        G_prepro = self.GetPredt_C(clf_G, X_, mclass=self.MClass)
        C_prepro = self.GetPredt_C(clf_C, X_, mclass=self.MClass)

        Pre_Pro = pd.DataFrame( np.c_[y_, G_prepro, C_prepro], index=y_.index)
        Pre_Pro.columns = ['TRUE', 'Pred_%s'%self.model]+ ['%s_Prob_%s'%(x, self.model)  for x in range(self.MClass) ] + \
                                ['Pred_C_%s'%self.model] + ['%s_Prob_C_%s'%(x, self.model) for x in range(self.MClass) ]

        return { 'clf_G':clf_G, 'clf_C':clf_C, 'Pre_Pro' : Pre_Pro }

    def LearNing_C(self, X_train, X_test, y_train, y_test):
        MODEL = self.GetFit_C(X_train, X_test, y_train, y_test, Model = self.arg.model)

        clf_G = MODEL['clf_G']
        clf_C = MODEL['clf_C']
        Pre_Pro = MODEL['Pre_Pro']
        FeaCoef = self.Featurescoef(clf_G, index_=X_train.columns)

        Best_MD = [ {'model': self.arg.model     , 'clf': clf_G, 'features': X_train.columns.tolist() },
                    {'model': 'C_'+self.arg.model, 'clf': clf_C, 'features': X_train.columns.tolist() },]

        if (self.arg.model in ['GBDT', 'XGB', 'RF']) and (self.arg.Addmode !='N'):
            X_train_N, OHE = self.OneHot(clf_G, X_train)
            X_test_N , OHE = self.OneHot(clf_G, X_test, OHE)

            MODEL_A = self.GetFit_C( X_train_N, X_test_N, y_train, y_test, Model= self.arg.Addmode )
            Pre_ProA = MODEL_A['Pre_Pro'].iloc[:,1:]
            Pre_Pro = pd.concat([Pre_Pro, Pre_ProA], axis=1)

            Best_MD +=[ {'model': self.arg.Addmode     , 'clf': MODEL_A['clf_G'], 'leaf': OHE, 'features': X_train.columns.tolist() },
                        {'model': 'C_'+self.arg.Addmode, 'clf': MODEL_A['clf_C'], 'leaf': OHE, 'features': X_train.columns.tolist() },]

        if len(y_test)>2:
            Test_Mel = self.GetScore_C(Pre_Pro.loc[y_train.index], Pre_Pro.loc[y_test.index])
        else:
            Test_Mel = pd.DataFrame()

        return (Pre_Pro.loc[y_test.index], FeaCoef, Test_Mel, Best_MD)

    def Predicting_C(self,Best_Model,  pDFall, _X_names, _Y_name):
        Y_TRUE  = pDFall[_Y_name].to_frame(name= 'TRUE')
        Y_PRED , Y_Eval = [], []
        for _k, _cv in enumerate(Best_Model):
            Y_pred = [Y_TRUE]
            for _i, _m in enumerate(_cv):
                _mod = _m['model']
                clf_ = _m['clf']
                _Xft = _m['features']

                X_Pred = pDFall[_Xft]
                if 'leaf' in _m.keys():
                    leaf = _m['leaf']
                    X_Pred, _ = self.OneHot(_cv[0]['clf'], X_Pred, leaf)

                _prepro  = self.GetPredt_C(clf_, X_Pred, mclass=self.MClass)

                _prepro = pd.DataFrame( _prepro,
                                        index=pDFall.index,
                                        columns = ['Pred_%s'%_mod] + ['%s_Prob_%s'%(x, _mod) for x in range(self.MClass) ]
                                    )
                Y_pred.append(_prepro)

            Y_pred  = pd.concat( Y_pred, axis=1 )
            Y_predv = Y_pred[ ~Y_pred['TRUE'].isna() ]

            self.log.CIF( ('%s %s Model Predicting Parameters'% (_Y_name, self.arg.model)).center(45, '-') )
            if Y_predv.shape[0] > 2:
                Pred_Mel = self.GetSplit(Y_predv, label='Predict')
                self.log.CIF( "%s Modeling evaluation: \n%s" %(self.arg.model, Pred_Mel) )
            else:
                Pred_Mel = pd.DataFrame()
            self.log.CIF( ('Completed %2d%%'% ((_k+1)*100/len(Best_Model))).center(45, '-') )

            Y_PRED.append(Y_pred)
            Y_Eval.append(Pred_Mel)

        return (Y_PRED, Y_Eval)

    def FeatCoeffs(self, All_import, _Y_name, Xg):
        if All_import:
            All_import = pd.concat(All_import, axis=1,sort=False).fillna(0)
            column = sorted(set(All_import.columns))
            All_import = All_import[column]
            for i in column:
                All_import['%s_mean'%i]  = All_import[i].mean(axis=1)
                All_import['%s_std'%i]   = All_import[i].std(axis=1)
                All_import['%s_median'%i]= All_import[i].median(axis=1)
            All_import.sort_values(by=['%s_mean'%i for i in column], ascending=[False]*len(column), inplace=True, axis=0)

            Openf('%s%s_Class_features_importance.xls' %(self.arg.output, _Y_name), (All_import)).openv()
            MPlot('%s%s_Class_Features_importance.lines.pdf' %(self.arg.output, _Y_name)).Feature_Import_line( All_import, self.arg.model, _Y_name )
            MPlot('%s%s_Class_Features_importance.boxI.pdf'  %(self.arg.output, _Y_name)).Feature_Import_box( All_import, Xg, _Y_name, self.arg.model, sort_by_group=False )
            MPlot('%s%s_Class_Features_importance.boxII.pdf' %(self.arg.output, _Y_name)).Feature_Import_box( All_import, Xg, _Y_name, self.arg.model, sort_by_group=True)

    def ModelScore(self, All_evluat, _Y_name, label='Test'):
        if All_evluat:
            All_pvalues = pd.concat(All_evluat,  axis=0)
            Openf('%s%s_Class_%s_metrics_pvalues.xls'%(self.arg.output, _Y_name, label), (All_pvalues), index=True, index_label='Score').openv()

            All_pvalues['Score'] = All_pvalues.index
            MPlot('%s%s_Class_%s_metrics_scores.pdf'%(self.arg.output, _Y_name, label)).EvaluateCV( All_pvalues, 'Score',  All_pvalues.columns.drop('Score'))

    def CVMerge(self, All_test_pd, DFall, _X_names, _Y_name, Xg, label='Test'):
        All_test_ = pd.concat(All_test_pd,axis=0)

        _model = [re.sub('^Pred_','', _i) for _i in All_test_.columns if re.match('Pred_.*', _i)]
        All_Prob_mean = All_test_.filter(regex=("_Prob_")).groupby([All_test_.index]).mean()
        All_Prob_mean.columns +=  '_mean'
        All_Prob_median = All_test_.filter(regex=("_Prob_")).groupby([All_test_.index]).median()
        All_Prob_median.columns += '_median'
        All_Pred =  All_test_.filter(regex=("^TRUE|^Pred_")).groupby([All_test_.index]).apply(lambda x : x.mode(0).loc[0,:])
        All_Pred.columns = [ i + '_mode' if i !='TRUE' else i for i in All_Pred.columns ]

        def pred_(All_Pb_, All_Pred):
            _m = All_Pb_.columns.str.replace('.*_Prob','Prob').drop_duplicates(keep='first')
            for _x in _m:
                All_Pred[_x.replace('Prob','Pred') ] = All_Pb_.filter(regex=_x, axis=1).values.argmax(1)
            return All_Pred

        All_Pred = pred_(All_Prob_mean, All_Pred)
        All_Pred = pred_(All_Prob_median, All_Pred)

        ALL_Result  = pd.concat([All_Pred, All_Prob_mean, All_Prob_median], axis=1)

        Openf('%s%s_Class_%s_detials.xls'%(self.arg.output, _Y_name, label), (All_test_), index=True, index_label='sample').openv()
        Openf('%s%s_Class_%s_summary.xls'%(self.arg.output, _Y_name, label), (ALL_Result), index=True, index_label='sample').openv()
        ClusT('%s%s_Class_%s_Feature.Data_complete.pdf'%(self.arg.output, _Y_name, label) ).Plot_heat( DFall[_X_names], ALL_Result.filter(regex=("^TRUE|^Pred_.*_mean")), Xg, method='complete')
        ClusT('%s%s_Class_%s_Feature.Data_average.pdf' %(self.arg.output, _Y_name, label) ).Plot_heat( DFall[_X_names], ALL_Result.filter(regex=("^TRUE|^Pred_.*_mean")), Xg, method='average')

        ALL_Resultv = ALL_Result[~ ALL_Result['TRUE'].isna()]
        if ALL_Resultv.shape[0] > 2:
            Parallel( n_jobs=-1 )( delayed( _p)
                                    ( ALL_Resultv['TRUE'], ALL_Resultv['Pred_%s_mean'%_m], ALL_Resultv.filter(regex='.*_Prob_%s_mean'%_m, axis=1), _m ) 
                                    for _m in _model 
                                    for _p in [ ROCPR('%s%s_Class_%s_%s_ROC.pdf'%(self.arg.output, _Y_name, label, _m)).ROC_CImport, 
                                                ROCPR('%s%s_Class_%s_%s_PR.pdf' %(self.arg.output, _Y_name, label, _m)).PR_CImport]
                                )

            Parallel( n_jobs=-1 )( delayed( _p)
                        ( All_test_pd, _Y_name, _m )
                        for _m in _model 
                        for _p in [ ROCPR('%s%s_Class_%s_%s_CVROC.pdf'%(self.arg.output, _Y_name, label, _m)).ROC_EImport, 
                                    ROCPR('%s%s_Class_%s_%s_CVPR.pdf' %(self.arg.output, _Y_name, label, _m)).PR_EImport]
                    )

            ROCPR( '%s%s_Class_%s_All_ROC.pdf' % (self.arg.output, _Y_name, label) ).ROC_MImport(ALL_Resultv, _model, _Y_name)
            ROCPR( '%s%s_Class_%s_All_PR.pdf'  % (self.arg.output, _Y_name, label) ).PR_MImport( ALL_Resultv, _model, _Y_name)

            _EvalueA = self.GetSplit(ALL_Resultv.filter(regex='^TRUE|_mean$', axis=1), label=label)
            self.log.CIF( "The final mean evaluation: \n%s" %_EvalueA )

        self.log.CIF( ('Finish %s'% label).center(45, '-') )

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
        self.MClass= 2
        self.arg.output = '%s/03ModelFit/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs(os.path.dirname(self.arg.output), exist_ok=True)

    def Model_X(self, DFall, _X_names, _Y_name, Xg, Type= 'C'):
        self.MClass = DFall[_Y_name].nunique()
        DFall[_Y_name] = Check_Label(DFall[_Y_name])
        CVRlt = Processing( self.arg, self.log, model=self.arg.model, Y_name=_Y_name, MClass= self.MClass, Type = Type )

        All_Test, All_coefs_, All_evluat, All_estmt_ = [], [], [], []
        for _k in range(self.arg.Repeatime):
            CVM   = CVRlt.CVSplit(DFall[_X_names], DFall[_Y_name], CVt = self.arg.CVfit)
            for _n, (train_index, test_index) in enumerate(CVM):
                _Train , _Test   = DFall.iloc[train_index, :], DFall.iloc[test_index, :]
                X_train, y_train = _Train[_X_names], _Train[_Y_name]
                X_test , y_test  = _Test[ _X_names], _Test[_Y_name]

                self.log.CIF( ('%s %s Model Fitting and hyperparametering.'% (_Y_name, self.arg.model)).center(45, '-') )
                _Test, _coefs_, _evluat, _estmt_  = CVRlt.LearNing_C( X_train, X_test, y_train, y_test )

                All_Test.append(_Test)
                All_estmt_.append(_estmt_)

                if not _coefs_.empty:
                    self.log.CIF( "%s Feature coefficiency: \n%s" % (self.arg.model, _coefs_) )
                    All_coefs_.append(_coefs_)
                if not _evluat.empty:
                    self.log.CIF( "%s Modeling evaluation: \n%s" %(self.arg.model, _evluat) )
                    All_evluat.append(_evluat)

                self.log.CIF( ('Modeling has been Completed %2d%%'% ((_k+1)*(_n+1)*100/(self.arg.Repeatime*len(CVM)))).center(45, '-') )

        ### All_Test_predict
        CVRlt.CVMerge( All_Test, DFall, _X_names, _Y_name, Xg, label='Test' )
        ### All_coefs_
        CVRlt.FeatCoeffs(All_coefs_, _Y_name, Xg)
        ### All_evaluate_score_
        CVRlt.ModelScore(All_evluat, _Y_name, label='Test')
        ### All_evaluate_model_
        joblib.dump(All_estmt_, '%s%s_Class_best_estimator.pkl' %(self.arg.output, _Y_name), compress=1)

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
            self.Model_X( DFall, _Xi , Yi, Xg, Type= Typi )
            self.log.CIF( ('%s: Supervised MODELing Finish'%Yi).center(45, '*') )
