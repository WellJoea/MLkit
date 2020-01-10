from sklearn.preprocessing import (label_binarize,  OneHotEncoder, FunctionTransformer,
                                    MinMaxScaler, minmax_scale, MaxAbsScaler,
                                    StandardScaler, RobustScaler, Normalizer,
                                    QuantileTransformer, PowerTransformer, OrdinalEncoder)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold

from sklearn_pandas import DataFrameMapper, cross_val_score
import pandas as pd
import numpy as np
import os

from .MLOpenWrite import Openf, OpenM
from .MLPlots import ClusT
from .MLUnsupervising import Decomposition

class PreProcessing():
    def __init__(self, arg, log,  *array, score=None, model='RF', **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.score = score
        self.model = model
        self.dicts = dicts
        self.SSS   = StratifiedShuffleSplit(n_splits=10, test_size=0.25 , random_state=0)

    def StratifiedShuffleSplit_add(self, Xdf, Ydf):
        sss = StratifiedShuffleSplit(n_splits=self.arg.crossV, test_size=self.arg.testS, random_state=0)
        all_test = []
        all_split= []
        for train_index, test_index in sss.split(Xdf,Ydf):
            all_test.extend(test_index)
            all_split.append( [train_index.tolist(), test_index.tolist()])

        y = range(len(Ydf))
        if len(set(all_test))< len(y):
            all_index = y
            test_index= set(all_test)
            test_num = int(len(y)*0.25)
            lack_indx= set(all_index) - set(test_index)
            lack_num = len(lack_indx)

            if lack_num < test_num:
                fill_num = test_num-lack_num
                fill_set = list(test_index)[::int(round(len(test_index)/fill_num))]
                lack_indx = (lack_indx | set(fill_set))
                train_test_add = [ [list(set(all_index) - lack_indx), list(lack_indx)] ] 
            else:
                lack_indx = list(lack_indx)
                split_tim = int(lack_num/test_num)
                split_bin = int(lack_num/split_tim) +1
                split_set = [ lack_indx[i: i+split_bin] for i in range(0, lack_num, split_bin)]
                train_test_add = [ [list(set(all_index) - set(i)), i ] for i in split_set ]
            all_split += train_test_add
        return(all_split)

    def Remove_Fill(self, Dfall, verbose_eval=True, drop=False):
        self.log.CIF('Missing Values Processing'.center(45, '-'))
        self.log.CIF('Primary Samples and Feauters numbers: %s , %s' %Dfall.shape)
        dfa = Dfall.copy()
        if drop:
            dfa.dropna(thresh=round(dfa.shape[0]*self.arg.MissCol,0),inplace=True,axis=1)
            dfa.dropna(thresh=round(dfa.shape[1]*self.arg.MissRow,0),inplace=True,axis=0)

            for _c in dfa.columns:
                mostfreq = dfa[_c].value_counts().max()/dfa.shape[0]
                if  mostfreq > self.arg.Mtfreq:
                    dfa.drop([_c], inplace=True,axis=1)
                    if verbose_eval:
                        self.log.CIF('The modal number of %s is %s, higher than %s and droped in trainingtest set!!!' %(_c, mostfreq, self.arg.Mtfreq) )

            self.log.CIF('Final Samples and Feauters numbers: %s , %s' %dfa.shape)
            self.log.CIF('The Removed Samples : %s' % (list(set(Dfall.index)-set(dfa.index))) )
            self.log.CIF('The Removed Features: %s' % (list(set(Dfall.columns)-set(dfa.columns))) )

        Xa_drop = [ i for i in Dfall.columns if i in dfa.columns]
        imp = SimpleImputer(missing_values=np.nan,
                            strategy  =self.arg.MissValue,
                            fill_value=self.arg.FillValue,
                            copy=True)
        imp.fit( dfa[Xa_drop] )
        self.log.NIF('SimpleImputer paramaters:\n%s' %imp)
        self.log.CIF(45 * '-')
        return ( imp, Xa_drop )

    def Fill_Miss(self, TTdata, _X_names, _Y_names, Xg, Pdata=pd.DataFrame()):
        if Pdata.empty:
            FitData, head, drop = (TTdata[_X_names], 'TrainTest', True)
            TransData = TTdata[_X_names +_Y_names ]
        else:
            TransData, head, drop = Pdata[_X_names + _Y_names], 'Predict', False
            if self.arg.refset == 'train':
                FitData = TTdata[_X_names]
            elif self.arg.refset == 'predict':
                FitData = Pdata[_X_names]
            elif self.arg.refset == 'all':
                FitData = pd.concat( (TTdata[_X_names], Pdata[_X_names]), axis=0,  ignore_index=False )

        clf, Xa_drop  = self.Remove_Fill(FitData, drop=drop, verbose_eval=True)
        OutXY = pd.DataFrame(clf.transform( TransData[Xa_drop] ), index=TransData.index, columns=Xa_drop)
        OutXY[_Y_names] = TransData[_Y_names]

        Openf( '%s%s.set.miss_fill_data.xls'%(self.arg.output, head), (OutXY)).openv()
        #ClusT( '%s%s.set.raw.person.VIF.pdf'%(self.arg.output, head)).Plot_person( OutXY, Xa_drop, Xg )
        #ClusT( '%s%s.set.raw.pair.plot.pdf'%(self.arg.output, head) ).Plot_pair( OutXY )

        return (OutXY, Xa_drop)

    def Standard_(self, dfa, scale = 'S'):
        Scalers = {
            'S' : StandardScaler(),
            'R' : RobustScaler(quantile_range=tuple(self.arg.QuantileRange)),
            'M' : MinMaxScaler(),
            'MA': MaxAbsScaler(),
            'OE': OrdinalEncoder(),
            'OH': OneHotEncoder(),
            'NL' : Normalizer(),
            'QT': QuantileTransformer(),
            'PT': PowerTransformer(),
            'N' : FunctionTransformer( validate=False ),
        }
        Sca_map = [Scalers[i] for i in scale]
        Xa = list( dfa.columns )

        mapper = DataFrameMapper([ ( Xa, Sca_map ) ])
        clfit = mapper.fit( dfa )

        self.log.CIF('Standardization Pocessing'.center(45, '-'))
        self.log.NIF('Scale paramaters:\n%s' %clfit)
        self.log.CIF(45 * '-')

        return clfit

    def Standard_Feature(self, TTdata, _X_names, _Y_names, Xg, Pdata=pd.DataFrame() ):
        if Pdata.empty:
            FitData, head = (TTdata[_X_names], 'TrainTest')
            TransData = TTdata[_X_names +_Y_names ]
        else:
            TransData, head = Pdata[_X_names + _Y_names], 'Predict'
            if self.arg.refset == 'train':
                FitData = TTdata[_X_names]
            elif self.arg.refset == 'predict':
                FitData = Pdata[_X_names]
            elif self.arg.refset == 'all':
                FitData = pd.concat( (TTdata[_X_names], Pdata[_X_names]), axis=0,  ignore_index=False )

        clf  = self.Standard_(FitData, scale= self.arg.scaler)
        Xa_F = clf.features[0][0]

        OutXY = pd.DataFrame(clf.transform( TransData[Xa_F] ), index=TransData.index, columns=Xa_F)
        OutXY[_Y_names] = TransData[_Y_names]

        Openf('%s%s.standard.data.xls'%(self.arg.output, head), OutXY).openv()
        ClusT('%s%s.standard.person.VIF.pdf'%(self.arg.output, head)  ).Plot_person(OutXY, Xa_F, Xg)
        ClusT('%s%s.standard.pair.plot.pdf'%(self.arg.output, head)   ).Plot_pair(OutXY)
        ClusT('%s%s.standard.compair.hist.pdf'%(self.arg.output, head)).Plot_hist(TransData, OutXY, Xa_F)

        return(OutXY)

class Engineering():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        self.arg.output = '%s/01PreProcess/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs(os.path.dirname(self.arg.output), exist_ok=True)

    def Common(self):
        self.log.CIF( 'Feature Engineering'.center(45, '*') )
        (group, RYa, CYa, Xa, Xg) = OpenM(self.arg, self.log).openg()
        dfall  = OpenM(self.arg, self.log).openi()
        Xa  = [i for i in dfall.columns if i in Xa]
        AYa = CYa + RYa

        self.log.CIF('Group Variable Category'.center(45, '-'))
        self.log.CIF('The Samples and Feauters numbers: %s , %s' %dfall[Xa].shape )
        self.log.CIF('The discrete   labels: %s'   % CYa)
        self.log.CIF('The continuous labels: %s' % RYa)
        self.log.CIF(45 * '-')

        _AllDF_, _Xa = PreProcessing(self.arg, self.log).Fill_Miss( dfall, Xa, AYa, Xg )
        _AllDF  = PreProcessing(self.arg, self.log).Standard_Feature( _AllDF_, _Xa, AYa, Xg )

        ClusT(self.arg.output + 'Features.stand.MTX_complete.pdf' ).Plot_heat(_AllDF[_Xa] , _AllDF[AYa], Xg, method='complete')
        ClusT(self.arg.output + 'Features.stand.MTX_average.pdf'  ).Plot_heat(_AllDF[_Xa] , _AllDF[AYa], Xg, method='average' )

        Decomposition(self.arg, self.log).PCA( _AllDF, _Xa, AYa, self.arg.output + 'Features.stand'  )

        self.log.CIF( 'Feature Engineering Finish'.center(45, '*') )

