from sklearn.metrics import (accuracy_score, f1_score, 
                             classification_report,
                             precision_recall_curve, mean_squared_error, 
                             roc_curve, auc, r2_score, mean_absolute_error, 
                             average_precision_score, explained_variance_score)

import joblib
import os
import numpy as np
import pandas as pd

from .MLOpenWrite import OpenM, Openf
from .MLPreprocessing import PreProcessing
from .MLSupervising  import Processing
from .MLPlots import ClusT, ROCPR, MPlot
from .MLScoring import Binning

class Supervised():
    def __init__(self, arg, log, Type='C', MClass='2',  *array, **dicts):
        self.arg = arg
        self.log = log
        self.Type=Type
        self.MClass= MClass
        self.array = array
        self.dicts = dicts

    def Predict_X(self, DFall, _X_names, _Y_name, Xg):
        Best_Model = joblib.load('%s/03ModelFit/%s%s_Class_best_estimator.pkl'%(self.arg.outdir, self.arg.header, _Y_name))
        _X_names = Best_Model[0][0]['features']
        CVRlt = Processing( self.arg, self.log, model=self.arg.model, Y_name=_Y_name,
                            MClass= self.MClass, Type = self.Type)
        Y_PRED, Y_Eval = CVRlt.Predicting_C( Best_Model, DFall, _X_names, _Y_name )

        ### All_Pred_predict
        CVRlt.CVMerge( Y_PRED, DFall, _X_names, _Y_name, Xg, label='Predict' )
        ### All_evaluate_score_
        CVRlt.ModelScore(Y_Eval, _Y_name, label='Predict')

    def Score_predict(self, pCf_xy, pXa, pYi, Xg):
        __MODEL = joblib.load('%s%s_Class_LG_model_WOEIV.data' % (self.arg.output, pYi))
        _X_SCOs = []
        _X_WOEs = []
        _X_PREd = []

        for _i_MODEL in __MODEL:
            X_Scor, X_PREd, X_Woe  = Binning(self.arg, self.log).LR_Predict( pCf_xy, pXa, pYi, _i_MODEL )
            _X_SCOs.append(X_Scor)
            _X_PREd.append(X_PREd)
            _X_WOEs.append(X_Woe)

        _X_WOEs = pd.concat(_X_WOEs, axis=0, sort=False)
        _X_WOEs = _X_WOEs.groupby([_X_WOEs.index]).mean()

        _X_SCOs = pd.concat(_X_SCOs, axis=0, sort=False)
        _X_SCOs = _X_SCOs.groupby([_X_SCOs.index]).mean()
        _X_Features = _X_SCOs.columns

        _X_PREd = pd.concat(_X_PREd, axis=1, sort=False)
        _X_PREd[pYi] = pCf_xy[pYi]
        _X_PREd['mean_pred_pro'] = _X_PREd['pred_pro'].mean(1)
        _X_PREd['mean_Scores']   = _X_PREd['_Scores'].mean(1)
        _X_PREd['mode_Predict']  = _X_PREd['predict'].mode(1)[0]
        _X_PREd['mean_Predict']  = _X_PREd['mean_pred_pro'].apply( lambda x : 1 if x >0.5 else 0 )

        _X_WOEs = pd.concat([_X_WOEs,
                             _X_PREd[[pYi, 'mean_pred_pro', 'mean_Scores', 'mode_Predict', 'mean_Predict']]
                            ],axis=1, sort=False)

        _X_SCOs = pd.concat([_X_SCOs,
                             _X_PREd[[pYi, 'mean_pred_pro', 'mean_Scores', 'mode_Predict', 'mean_Predict']]
                            ],axis=1, sort=False)


        Openf('%s%s_Class_Predict_woe.xls'  %(self.arg.output, pYi), (_X_WOEs), index=True, index_label=0).openv()
        Openf('%s%s_Class_Predict_score.xls'%(self.arg.output, pYi), (_X_SCOs), index=True, index_label=0).openv()
        Openf('%s%s_Class_Predict_valuesxls'%(self.arg.output, pYi), (_X_PREd), index=True, index_label=0).openv()

        ClusT('%s%s_Class_Predict_LR_WOE_complete.pdf'%(self.arg.output, pYi) ).Plot_heat( _X_WOEs[_X_Features] , _X_WOEs[[pYi, 'mean_Predict', 'mean_Scores']], Xg, median=60, method='complete')
        ClusT('%s%s_Class_Predict_LR_WOE_average.pdf' %(self.arg.output, pYi) ).Plot_heat( _X_WOEs[_X_Features] , _X_WOEs[[pYi, 'mean_Predict', 'mean_Scores']], Xg, median=60, method='average')

        ClusT('%s%s_Class_Predict_LR_Scores_complete.pdf'%(self.arg.output, pYi) ).Plot_heat( _X_SCOs[_X_Features] , _X_SCOs[[pYi, 'mean_Predict', 'mean_Scores']], Xg, median=60, method='complete')
        ClusT('%s%s_Class_Predict_LR_Scores_average.pdf' %(self.arg.output, pYi) ).Plot_heat( _X_SCOs[_X_Features] , _X_SCOs[[pYi, 'mean_Predict', 'mean_Scores']], Xg, median=60, method='average')


        if not pCf_xy[pYi].isnull().any():
            _accuracy  = accuracy_score(_X_PREd[pYi], _X_PREd['mean_Predict'])
            _R2_score  = r2_score(_X_PREd[pYi], _X_PREd['mean_Predict'])

            precision, recall, threshold_pr = precision_recall_curve( _X_PREd[pYi], _X_PREd['mean_pred_pro'])
            fpr, tpr, threshold_roc = roc_curve( _X_PREd[pYi], _X_PREd['mean_pred_pro'] )
            roc_auc = auc(fpr, tpr)
            average_precision = average_precision_score(_X_PREd[pYi], _X_PREd['mean_pred_pro'])
            ks_score=(tpr-fpr).max()

            ROCPR( '%s%s_Class_Predict_LR_ROC_KS.pdf'%(self.arg.output, pYi) ).ROC_KS( _X_PREd[pYi], _X_PREd['mean_pred_pro'], _X_PREd['mean_Predict'],  'LogsticR')

            self.log.CIF( ('%s Predicting parameters of the LRCV'%pYi).center(45, '-') )
            self.log.NIF( classification_report(y_true=_X_PREd[pYi], y_pred=_X_PREd['mean_Predict']) )
            self.log.CIF("LG PD R2_score: %s"% _R2_score)
            self.log.CIF("LG PD accuracy: %s"% _accuracy)
            self.log.CIF("LG PD Roc_Auc : %s" % roc_auc)
            self.log.CIF("LG PD ks_score: %s"% ks_score)
            self.log.CIF("LG PD average_precision: %s" % average_precision)
            self.log.CIF(45 * '-')

class Prediction():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        if not ( bool(self.arg.out_predict) | bool(self.arg.modelpath) ):
            self.arg.output = '%s/04Prediction/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs(os.path.dirname(self.arg.output), exist_ok=True)

    def SpvPredicting(self):
        self.log.CIF( 'Supervisor Predicting'.center(45, '*') )
        (group, RYa, CYa, Xa, Xg) = OpenM(self.arg, self.log).openg()
        YType  =  group[ (group.Variables.isin( RYa + CYa)) ][['Variables','Type']]

        Predt_set = OpenM(self.arg, self.log).openp()
        TTdata = Openf( '%s/01PreProcess/%sTrainTest.set.miss_fill_data.xls'%( self.arg.outdir, self.arg.header ), index_col=0).openb()

        Pr_Xa = Predt_set.columns
        TT_Xa = TTdata.columns
        Xa  = [i for i in Pr_Xa if ( (i in Xa) & (i in TT_Xa) ) ]
        AYa = CYa + RYa

        _PredictF, pXa = PreProcessing(self.arg, self.log).Fill_Miss( TTdata, Xa, AYa, Xg, Pdata= Predt_set)
        _Predict       = PreProcessing(self.arg, self.log).Standard_Feature( TTdata, pXa, AYa, Xg, Pdata=_PredictF )

        for Yi,Typi in YType.values:
            if os.path.exists( '%s/03ModelFit/%s%s_Class_best_estimator.pkl'%(self.arg.outdir, self.arg.header, Yi) ):
                self.log.CIF( ('%s: Supervised Predicting'%Yi).center(45, '*') )
                Supervised(self.arg, self.log, Type=Typi, MClass=TTdata[Yi].nunique() ).Predict_X( _Predict, pXa, Yi, Xg )
                self.log.CIF( ('%s: Supervised Predicting Finish'%Yi).center(45, '*') )

            if os.path.exists( '%s/09CRDScore/%s%s_Class_LG_model_WOEIV.data'%(self.arg.outdir, self.arg.header, Yi) ):
                self.arg.output = '%s/09CRDScore/%s'%( self.arg.outdir, self.arg.header )
                os.makedirs(os.path.dirname(self.arg.output), exist_ok=True)
                self.log.CIF( ('%s: Credit Predicting'%Yi).center(45, '*') )
                Supervised(self.arg, self.log).Score_predict( _Predict, pXa, Yi, Xg )
                self.log.CIF( ('%s: Credit Predicting Finish'%Yi).center(45, '*') )
        self.log.CIF( 'Supervisor Predicting Finish'.center(45, '*') )