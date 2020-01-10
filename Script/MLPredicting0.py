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
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts

    def C_predict( self, pCf_xy, pXa, pYi, Xg ):
        pYi_DF = pCf_xy[pYi]
        pXa_DF = pCf_xy[pXa]

        Best_model = joblib.load('%s/03ModelFit/%s%s_Class_best_estimator.pkl'%(self.arg.outdir, self.arg.header, pYi))

        _Y_predict, _ROC_PR, _ROC_PR_LR = Processing(
                self.arg, self.log, model=self.arg.model ).Predicting_C( pCf_xy, pXa, pYi, Best_model )

        _Predict   = pd.concat(_Y_predict,axis=0)
        _Predict_G = _Predict.groupby([_Predict.index]).mean()

        _Predict_pro = _Predict_G.filter(regex=("^Prob_"))
        _Predict_pro.columns = [ int(x.replace('Prob_','')) for x in _Predict_pro.columns ]

        _Predict_G['mode_predict'] = _Predict['Predict'].groupby([_Predict.index]).apply(lambda x : x.mode(0)[0])
        _Predict_G['mean_predict'] = _Predict_pro.idxmax(1)

        if not pYi_DF.isnull().any():
            _Y_len = _Y_predict[0].filter(regex=("^Prob_")).shape[1]

            ROCPR('%s%s_Class_Predict_ROC.pdf'%(self.arg.output, pYi)).ROC_CImport(_Predict_G['True'],  _Predict_pro, _Predict_G['mean_predict'], self.arg.model )
            ROCPR('%s%s_Class_Predict_PR.pdf' %(self.arg.output, pYi)).PR_CImport( _Predict_G['True'],  _Predict_pro, _Predict_G['mean_predict'], self.arg.model)

            if _Y_len == 2:
                ROCPR( '%s%s_Class_Predict_CV_ROC_curve.pdf' % (self.arg.output, pYi) ).ROC_Import( _Y_predict, '', self.arg.model, pYi)
                ROCPR( '%s%s_Class_Predict_CV_PR_curve.pdf'  % (self.arg.output, pYi) ).PR_Import(  _Y_predict, '', self.arg.model, pYi)
            else:
                ROCPR( '%s%s_Class_Predict_CV_ROC_curve.pdf' % (self.arg.output, pYi) ).ROC_MImport(_Y_predict, '', self.arg.model, pYi)
                ROCPR( '%s%s_Class_Predict_CV_PR_curve.pdf'  % (self.arg.output, pYi) ).PR_MImport( _Y_predict, '', self.arg.model, pYi)

        ### + LR
        if 'LR_predict' in  _Predict_G.columns:
            _Predict_G['LR_mode_predict']  =  _Predict_G['LR_predict'].apply( lambda x : 1 if x >0.5 else 0 )
            _Predict_G['LR_mean_predict']  =  _Predict_G['LR_predict_proba_1'].apply( lambda x : 1 if x >0.5 else 0 )

            if not pYi_DF.isnull().any():
                ROCPR( '%s%s_Class_+LR_Predict_ROC.pdf'%(self.arg.output, pYi)    ).ROC_CImport(_Predict_G['True'],  _Predict_G['LR_predict_proba_1'], _Predict_G['LR_mean_predict'], self.arg.model + '+LR')
                ROCPR( '%s%s_Class_+LR_Predict_ROC.pdf'%(self.arg.output, pYi)    ).PR_CImport( _Predict_G['True'],  _Predict_G['LR_predict_proba_1'], _Predict_G['LR_mean_predict'], self.arg.model + '+LR')
                ROCPR( '%s%s_Class_+LR_Predict_ROC_KS.pdf'%(self.arg.output, pYi) ).ROC_KS(     _Predict_G['True'],  _Predict_G['LR_predict_proba_1'], _Predict_G['LR_mean_predict'], self.arg.model + '+LR')

                All_dt = [ [self.arg.model, _Predict_G['True'], _Predict_G['Prob_1'], _Predict_G['mean_predict'] ],
                           [self.arg.model + '+LR', _Predict_G['True'], _Predict_G['LR_predict_proba_1'] , _Predict_G['LR_mean_predict'] ],
                         ]
                ROCPR( '%s%s_Class_+LR_Predict_ROC_COMpare.pdf'%(self.arg.output, pYi) ).ROC_COMpare( All_dt, pYi )
                ROCPR( '%s%s_Class_+LR_Predict_PR_COMpare.pdf' %(self.arg.output, pYi) ).PR_COMpare(  All_dt, pYi )


        Openf('%s%s_Class_Predict_summary.xls'%(self.arg.output, pYi), (_Predict_G), index=True, index_label=0).openv()

        _X_Features = Best_model[0]['Features'] 
        _ALL_DF = pd.concat([pCf_xy, _Predict_G], axis=1, sort=False)

        _All_Y  = _ALL_DF['mean_predict'].to_frame()
        if not pYi_DF.isnull().any():
            _All_Y['True']  = _ALL_DF['True']
        if 'LR_predict' in  _Predict_G.columns:
            _All_Y[['LR_predict','LR_Score']]  = _ALL_DF[['LR_mean_predict', 'LR_predict_proba_1']]

        ClusT('%s%s_Class_Predict_Feature.Data_complete.pdf'%(self.arg.output, pYi) ).Plot_heat( _ALL_DF[_X_Features], _All_Y, Xg, method='complete')
        ClusT('%s%s_Class_Predict_Feature.Data_average.pdf' %(self.arg.output, pYi) ).Plot_heat( _ALL_DF[_X_Features], _All_Y, Xg, method='average')

        ClusT('%s%s_Class_Predict_Feature.Data_Raw_complete.pdf'%(self.arg.output, pYi) ).Plot_heat( _ALL_DF[pXa], _All_Y, Xg, method='complete')
        ClusT('%s%s_Class_Predict_Feature.Data_Raw_average.pdf' %(self.arg.output, pYi) ).Plot_heat( _ALL_DF[pXa], _All_Y, Xg, method='average')

    def R_predict( self, pCf_xy, pXa, pYi, Xg ):
        pass

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
                if Typi == 'C':
                    Supervised(self.arg, self.log).C_predict( _Predict, pXa, Yi, Xg )
                elif Typi == 'R':
                    Supervised(self.arg, self.log).R_predict( _Predict, pXa, Yi, Xg )
                self.log.CIF( ('%s: Supervised Predicting Finish'%Yi).center(45, '*') )

            if os.path.exists( '%s/09CRDScore/%s%s_Class_LG_model_WOEIV.data'%(self.arg.outdir, self.arg.header, Yi) ):
                self.arg.output = '%s/09CRDScore/%s'%( self.arg.outdir, self.arg.header )
                os.makedirs(os.path.dirname(self.arg.output), exist_ok=True)
                self.log.CIF( ('%s: Credit Predicting'%Yi).center(45, '*') )
                Supervised(self.arg, self.log).Score_predict( _Predict, pXa, Yi, Xg )
                self.log.CIF( ('%s: Credit Predicting Finish'%Yi).center(45, '*') )
        self.log.CIF( 'Supervisor Predicting Finish'.center(45, '*') )