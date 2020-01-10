from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd
from .MLOpenWrite import Openf
from .MLPlots import ClusT

class Decomposition():
    def __init__(self, arg, log,  *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts

    def DOPCA(_func):
        def wrapper(self, *args):
            if args[0].shape[1] >300:
                pass
            else:
                _func(self, *args)
        return wrapper

    def PCA(self, _AllDF, Xa, Ya, out ):
        Xdf = _AllDF[Xa]
        if Xdf.shape[0] >= Xdf.shape[1]:
            pca = PCA(n_components='mle',
                  svd_solver ='auto',
                  copy=True,
                  whiten=False,
                  random_state=0)
        else :
            pca = PCA(n_components=None,
                  svd_solver ='auto',
                  copy=True,
                  whiten=False,
                  random_state=0)

        pca.fit(Xdf)
        Xdf_new = pca.fit_transform(Xdf)
        Xdf_new = pd.DataFrame(Xdf_new)
        Xdf_new.index = Xdf.index

        pca_ratio = pca.explained_variance_ratio_
        pca_ratio_sum = np.cumsum(np.round(pca_ratio , decimals=8))

        index = 0
        if sum(pca_ratio) <= self.arg.pcathreshold:
            index = len(pca_ratio)
        else:
            for i,j in enumerate(pca_ratio_sum, 1):
                if j >=self.arg.pcathreshold :
                    index = i
                    break
            if index <=5 :
                index = len(pca_ratio)

        Xdf_new = Xdf_new[range(index)]
        _Xa_N   = ['feature_' + str(i) for i in range(Xdf_new.shape[1])]
        Xdf_new.columns =_Xa_N

        self.log.CIF('PCA'.center(45, '-'))
        self.log.CIF('PCA features length: %s' %len(pca_ratio))
        self.log.CIF('PCA variance ratio : \n%s' %pca_ratio )
        self.log.CIF(45 * '-')

        _AllDF_N = pd.concat( [ Xdf_new, _AllDF[Ya] ], axis = 1)
        Openf(out + '.PCA.xls', _AllDF_N ).openv()
        ClusT(out + '.PCA_complete.pdf' ).Plot_heat(_AllDF_N[_Xa_N] , _AllDF_N[Ya], [], method='complete')
        ClusT(out + '.PCA_average.pdf'  ).Plot_heat(_AllDF_N[_Xa_N] , _AllDF_N[Ya], [], method='average' )

        return(Xdf_new, pca_ratio )
