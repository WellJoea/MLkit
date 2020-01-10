from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
from statsmodels.stats.outliers_influence import variance_inflation_factor
#import scikitplot as skplt
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import seaborn as sns
import fastcluster
import os

from .MLUtilities import Check_Label, Check_proba, Check_Binar

class Baseset():
    def __init__(self, outfile, *array, **dicts):
        self.out = outfile
        os.makedirs( os.path.dirname(self.out), exist_ok=True )
        self.array = array
        self.dicts = dicts
        self.color_ = [ '#00BD89', '#DA3B95', '#16E6FF', '#E7A72D', '#8B00CC', '#EE7AE9',
                        '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF',
                        '#FF2121', '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C',
                        '#FB9A99', '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2',
                        '#95A5A6', '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC',
                        '#A473AE', '#FF0000', '#EE7777', '#009E73', '#ED5401', '#CC0073',]
        self.linestyle_ = [
            (0, ()),                    # 'solid',
            (0, (1, 1)),                # 'densely dotted',
            (0, (5, 5)),                # 'dashed',
            (0, (5, 1)),                # 'densely dashed',
            (0, (3, 5, 1, 5)),          # 'dashdotted',
            (0, (3, 1, 1, 1)),          # 'densely dashdotted',
            (0, (3, 5, 1, 5, 1, 5)),    # 'dashdotdotted',
            (0, (3, 5, 1, 1, 1, 1)),    # 'densely dashdotdotted'
            (0, (1, 10)),               # 'loosely dotted',
            (0, (5, 10)),               # 'loosely dashed',
            (0, (3, 10, 1, 10)),        # 'loosely dashdotted',
            (0, (3, 10, 1, 10, 1, 10)), # 'loosely dashdotdotted'
        ]

        self.markertyle_ = ['X', 'P', '*', 'v', 's','o','^']
        self.Sequential  = [plt.cm.Oranges , plt.cm.Reds , plt.cm.YlOrBr, plt.cm.YlOrRd  , plt.cm.OrRd,
                            plt.cm.PuRd , plt.cm.OrRd , plt.cm.BuPu  , plt.cm.Purples   , plt.cm.Blues,
                            plt.cm.GnBu, plt.cm.PuBu, plt.cm.Greens, plt.cm.YlGnBu, plt.cm.PuBuGn,
                            plt.cm.BuGn , plt.cm.YlGn , plt.cm.Greys , ]
        self.Sequential2 = [plt.cm.cool, plt.cm.spring, plt.cm.summer, plt.cm.autumn, plt.cm.winter, 
                            plt.cm.Wistia, plt.cm.pink, plt.cm.viridis, plt.cm.plasma, plt.cm.inferno, 
                            plt.cm.magma, plt.cm.cividis, plt.cm.hot, plt.cm.afmhot, plt.cm.gist_heat, 
                            plt.cm.copper, plt.cm.binary, plt.cm.gist_gray, plt.cm.gray, plt.cm.bone,]
        self.Diverging   = [plt.cm.bwr , plt.cm.coolwarm, plt.cm.seismic, plt.cm.RdYlGn, plt.cm.RdYlBu,
                            plt.cm.RdBu, plt.cm.Spectral, plt.cm.PiYG   , plt.cm.PRGn  , plt.cm.BrBG  ,
                            plt.cm.PuOr, plt.cm.RdGy    , ]

        font = {#'family' : 'normal',
                'weight' : 'normal',
                'size'   : 14}
        plt.rc('font', **font)
        plt.figure(figsize=(10,10))
        plt.margins(0,0)
        plt.rcParams.update({'figure.max_open_warning': 1000})

        #plt.rc('xtick', labelsize=20)
        #plt.rc('ytick', labelsize=20)
        #plt.rcParams['font.size'] = 23
        #plt.rcParams.update({'font.size': 22})
        #plt.rcParams['legend.fontsize'] = 'large'
        #plt.rcParams['figure.titlesize'] = 'medium'

class ClusT(Baseset):
    def Pdecorator(_func):
        def wrapper(self, *args, **kargs):
            if (args[0].shape[1] >500) | (min(args[0].shape)<3):
                pass
            else:
                _func(self, *args, **kargs)
        return wrapper

    @Pdecorator
    def Plot_person(self, dfa, Xa, Xg):
        VIF_df = dfa[Xa].assign(const=1)
        VIF    = pd.Series( [ variance_inflation_factor(VIF_df.values, i)
                            for i in range(VIF_df.shape[1])
                            ], index=VIF_df.columns).fillna(0)

        cr = np.corrcoef(dfa[Xa].values,rowvar=False)
        cf = pd.DataFrame(cr, index=Xa,columns=Xa).fillna(0)
        cf['VIF'] = VIF.drop('const')
        Xg_cor = Xg.Group.unique()
        cor_dict = dict(zip(Xg_cor, self.color_[:len(Xg_cor)]))
        cf['Group'] = Xg.Group.map(cor_dict)

        vif_dict = {}
        for i in cf.VIF.unique():
            if i <= 5:
                vif_dict[i] = plt.cm.Greens(i/10)
            elif 5 <= i <10:
                vif_dict[i] = plt.cm.Blues(i/10)
            elif i >= 10:
                vif_dict[i] = plt.cm.Reds(i/cf.VIF.max())
        cf['VIFs'] = cf.VIF.map(vif_dict)

        cf.to_csv( self.out+'.xls', sep='\t' )

        linewidths= 0 if min(cr.shape) > 60  else 0.01
        figsize   = (20,20) if min(cr.shape) > 60  else (15,15)

        hm = sns.clustermap(cf[Xa],
                            method='complete',
                            metric='euclidean',
                            z_score=None,
                            figsize=figsize,
                            linewidths=linewidths,
                            cmap="coolwarm",
                            center=0,
                            #fmt='.2f',
                            #square=True, 
                            #cbar=True,
                            #yticklabels=Xa,
                            #xticklabels=Xa,
                            vmin=-1.1,
                            vmax=1.1,
                            annot=False,
                            row_colors=cf[['Group','VIFs']],
                            col_colors=cf[['Group','VIFs']],
                            )
        hm.savefig(self.out)
        #hm.fig.subplots_adjust(right=.2, top=.3, bottom=.2)
        plt.close()

    def Plot_pair(self, dfX):
        if (dfX.shape) <= (15, 15):
            #figsize   = (30,30) if min(dfX.shape) > 50  else (25,25)
            #plt.figure(figsize=figsize)
            #hm = sns.pairplot(dfX, height=6, plot_kws ={'edgecolor' : None},kind="reg")
            sns.set_style("whitegrid", {'axes.grid' : False})
            hm = sns.pairplot(dfX,height=6)
            hm.savefig(self.out)
            #plt.figure(figsize=(10,10))
            plt.close()

    @Pdecorator
    def Plot_hist(self, dfaA, dfa, Xa):
        with PdfPages(self.out) as pdf:
            for Xi in Xa:
                plt.figure()
                ax1 = plt.subplot(1,2,1)
                ax2 = plt.subplot(1,2,2)
                plt.sca(ax1)
                sns.distplot(dfaA[Xi],bins=100, label='before scale', hist=True, kde=True, rug=True,
                            rug_kws={"color": "g"},
                            kde_kws={"color": "b", "lw": 1.1, "label": "gaussian kernel"},
                            hist_kws={"alpha": 1,'lw':0.01, "color": "r"}
                            )
                plt.sca(ax2)
                sns.distplot(dfa[Xi],bins=100, label='after scale', hist=True, kde=True, rug=True,
                            rug_kws={"color": "g"},
                            kde_kws={"color": "b", "lw": 1.1, "label": "gaussian kernel"},
                            hist_kws={"alpha": 1, 'lw':0.01, "color": "r"}
                            )
                pdf.savefig()
                plt.close()

    @Pdecorator
    def Plot_heat(self, Xdf, Ydf, Xg, median=None, method='complete', metric='euclidean' ):
        if isinstance(Xdf, pd.DataFrame) & (len(Xdf.shape) >1):
            Xdf = Xdf.fillna(0)
            _Xa = Xdf.columns.tolist()
            row_dt = None
            col_dt = None

            if len(Xg) >0:
                Xg_cor = Xg.Group.unique()
                cor_dict = dict(zip(Xg_cor, self.color_[:len(Xg_cor)]))
                Xg['Colors'] = Xg.Group.map(cor_dict)
                Xg  = Xg.loc[_Xa]
                col_dt = Xg.Colors.values

            if len(Ydf) >0:
                if len(Ydf.shape)==1:
                    Ydf = Ydf.to_frame()
                row_dt = Ydf.copy()

                n1=0
                n2=0
                for i in Ydf.columns:
                    _cors = sorted(Ydf[i].unique())
                    if len(_cors) <=8:
                        cor_dict = dict(zip(_cors,  self.color_[:len(_cors)]))
                    else:
                        min_, max_ = Ydf[i].min(), Ydf[i].max()

                        if (median != None) | ( (max_>0) & (min_<0) ):
                            if median == None:
                                median = 0
                            cmap = self.Diverging[n2]
                            median_sd = max(max_- median, median-min_ )
                            max_n, min_n = median- median_sd, median + median_sd
                            colormap  = [ cmap( (i-min_n)*0.8/(max_n-min_n) )  for i in _cors ]
                            n2 += 1
                        else:
                            cmap = self.Sequential[n1]
                            colormap  = [ cmap( (i-min_)*0.8/(max_-min_) )  for i in _cors ]
                            n1 += 1
                        cor_dict = dict(zip( _cors, colormap ))
                    row_dt[i] = Ydf[i].map(cor_dict)

            linewidths= 0 if max(Xdf.shape) > 60  else 0.01
            figsize   = (25,25) if max(Xdf.shape) > 60 else (18,18)
            cmap = None if Xdf.values.min() >=0 else 'coolwarm'
            _min_ = Xdf.values.min()
            _max_ = Xdf.values.max()
            if _min_<0:
                _max_ = max(-_min_, _max_ )
                _min_ = -_max_

            hm = sns.clustermap(Xdf,
                                method=method,
                                metric=metric,
                                z_score=None,
                                figsize=figsize,
                                linewidths=linewidths,
                                cmap=cmap,
                                vmin = _min_,
                                vmax = _max_,
                                #yticklabels=_Xa,
                                #xticklabels=_AllDF.index.tolist(),
                                annot=False,
                                row_colors=row_dt,
                                col_colors=col_dt
                                )
            hm.savefig(self.out)
            plt.close()

class MPlot(Baseset):
    def Feature_Sorce(self, select_Sco, select_Fea, N, Y, cv):
        color_ = self.color_*select_Sco.shape[1]
        linestyle_ = self.linestyle_*select_Sco.shape[1]
        markertyle_ = self.markertyle_*select_Sco.shape[1]

        Sco_mean = select_Sco.mean(axis=1)
        Sco_Std  = select_Sco.std(axis=1)
        Fea_mean = select_Fea.mean()
        Fea_Std  = select_Fea.std()
        Fea_min  = (Fea_mean-1*Fea_Std) if (Fea_mean-1*Fea_Std) > 0 else 1
        Fea_max  = (Fea_mean + 1*Fea_Std) if (Fea_mean + 1*Fea_Std)<= select_Sco.shape[0] else select_Sco.shape[0]
        for i in range(select_Sco.shape[1]):
            plt.plot(select_Sco[i], marker=markertyle_[i], markersize=3.2, color=color_[i], linestyle=linestyle_[i], lw=1.0, label='')
            plt.vlines(select_Fea[i], 0, 1, color=color_[i], linestyle=linestyle_[i], lw=1.2, label='')
        plt.plot(Sco_mean , 'k-', lw=1.5, label='mean_score')
        plt.vlines(Fea_mean, 0, 1, lw=1.5, label='')
        plt.fill_between(Sco_mean.index, Sco_mean-1*Sco_Std, Sco_mean + 1*Sco_Std, color='grey', alpha=0.25, label=r'$\pm$ 1 std. dev.')
        plt.axvspan(Fea_min, Fea_max , alpha=0.3, color='grey', label='')

        plt.title('The %s features score of %s with %s estimator'%(cv, Y, N))
        plt.legend(loc='lower right')
        plt.ylabel('The %s features score'%cv)
        plt.xlabel('n featrues')
        plt.ylim([0.0, 1.0])
        plt.savefig( self.out )
        plt.close()

    def Feature_Import_line(self, All_import, N, Y):
        plt.figure(figsize=(13,10))
        color_ = self.color_*All_import.shape[1]
        linestyle_ = self.linestyle_*All_import.shape[1]
        markertyle_ = self.markertyle_*All_import.shape[1]

        All_raw = All_import.filter(regex=r'^[0-9]+$',axis=1)
        column = sorted(set(All_raw.columns))
        if len(column) == 1:
            for i in range(All_raw.shape[1]): 
                plt.plot(All_import.iloc[:,i], marker=markertyle_[i], markersize=3.2, color=color_[i], 
                         linestyle=linestyle_[i], lw=1.3)
            plt.plot(All_import['0_mean'], 'k-.', lw=1.5, label='mean_import',alpha= 0.9 )
            plt.fill_between(All_import.index, All_import['0_mean']-1*All_import['0_std'], All_import['0_std'] +
                             1*All_import['0_mean'], color='grey', alpha=0.3, label=r'$\pm$ 1 std. dev.')
        else:
            for n in column:
                for i in range(All_import[n].shape[1]):
                    plt.plot(All_import[n].iloc[:,i], marker=markertyle_[n], markersize=3.2, 
                             color=color_[n], linestyle=linestyle_[n], lw=1.1)
                plt.plot(All_import['%s_mean'%n], color=color_[n], lw=1.5, label='%s mean_import'%n)
                plt.fill_between(All_import.index, All_import['%s_mean'%n]-1*All_import['%s_std'%n], All_import['%s_std'%n] +
                                1*All_import['%s_mean'%n], color=color_[n], alpha=0.25, label=r'%s $\pm$ 1 std. dev.'%n)

        if (All_raw.values.min()>=0) & (All_raw.values.max()<=1):
            plt.ylim([0.0, 1.00])
        plt.title( Y + ' ' + N + '  featrue/coef importance ROC Accuracy')
        plt.legend(loc='best')
        plt.ylabel( Y + ' ' + N + ' featrue/coef importance values')
        plt.xticks( rotation='270')
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1,  title='ROCs and Accuracy')
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')

    def Feature_Import_box(self, All_import, Xg, Y_name, Model, sort_by_group=True):
        plt.figure(figsize=(13,10))
        color_ = self.color_*All_import.shape[1]
        linestyle_ = self.linestyle_*All_import.shape[1]
        markertyle_ = self.markertyle_*All_import.shape[1]

        All_import = pd.concat([All_import, Xg[['Group']]],axis=1, join='inner', sort=False)
        All_import.sort_values(by=['0_mean', '0_median'], ascending=[False, False], inplace=True, axis=0)
        if sort_by_group:
             All_import.sort_values(by=['Group', '0_mean', '0_median'], ascending=[True, False, False], inplace=True, axis=0)

        Xg_cor   = All_import.Group.unique()
        cor_dict = dict(zip(Xg_cor, color_[:len(Xg_cor)]))
        All_import['ColorsX'] = All_import.Group.map(cor_dict)


        color_a = ['red' if i >= 0 else 'blue' for i in All_import['0_mean']]
        color_b = ['red' if i <  0 else 'blue' for i in All_import['0_mean']]
        color_c  = All_import['ColorsX'].to_list()

        All_raw = All_import.filter(regex=r'^[0-9.]+$', axis=1)
        column  = sorted(set(All_raw.columns))
        X_labels = All_raw.index.to_list()

        Y_sd_min = All_import['0_mean'] - 1*All_import['0_std']
        Y_sd_max = All_import['0_mean'] + 1*All_import['0_std']
        #plt.plot(All_import['0_mean'], 'k-.', lw=1.5, label='mean_import', alpha=0.9)
        plt.fill_between( X_labels, Y_sd_min, Y_sd_max,
                        color='grey', alpha=0.3, label=r'$\pm$ 1 std. dev.')

        legend_elements=[ Patch(facecolor=cor_dict[g], edgecolor='r', label=g) for g in sorted(cor_dict.keys()) ]
        legend_elements.append(Line2D([0], [0], color='r', linestyle='-.', lw=1.5, alpha= 0.9, label='mean value') )
        legend_elements.append(Line2D([0], [0], color='r', linestyle='-',  lw=1.5, alpha= 0.9, label='median value') )
        legend_elements.append(Patch(facecolor='grey', edgecolor='black' , alpha=0.3, label=r'$\pm$ 1 std. dev.') )
        if All_import['0_mean'].min() <0:
            legend_elements.append(Patch(facecolor='white',  edgecolor='blue',  label=r'featrue/coef $\geq$0') )
            legend_elements.append(Patch(facecolor='white',  edgecolor='red' ,  label=r'featrue/coef <0') )

        ncol_ = 1 if len(legend_elements) <=6 else 2

        bplot =plt.boxplot(All_raw,
                        patch_artist=True,
                        vert=True,
                        labels=X_labels,
                        notch=0,
                        positions=range(len(X_labels)),
                        meanline=True,
                        showmeans=True,
                        meanprops={'linestyle':'-.'}, #'marker':'*'},
                        sym='+',
                        whis=1.5
                        )
        for i, patch in enumerate(bplot['boxes']):
            patch.set(color=color_b[i], linewidth=1.3)
            patch.set(facecolor = color_c[i])

        for element in [ 'means','medians','fliers']:     #['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
            for i, patch in enumerate(bplot[element]):
                patch.set(color=color_b[i], linewidth=1)
        #bplot1 =plt.violinplot(All_raw, positions=range(len(X_labels)),)
        #ax = sns.violinplot(data = All_raw.T, inner='box',scale = 'width',  cut=0, palette=color_a )

        plt.title(Y_name + ' ' + Model + '  featrue/coef importance ROC Accuracy')
        plt.legend(handles=legend_elements, ncol=ncol_, prop={'size':11}, loc='upper right')
        plt.ylabel(Y_name + ' ' + Model + ' featrue/coef importance values')
        plt.xticks(rotation='270')
        plt.savefig(self.out, bbox_inches='tight')

    def EvaluateCV(self, All_evluat, _group, _items):
        color_ = self.color_*len(All_evluat)

        melt = pd.melt(All_evluat, id_vars=[_group], value_name='value', var_name='variable', value_vars= _items)
        melt['value'] = melt['value'].astype(float)

        fig = sns.catplot(y="value",x=_group, hue='variable', 
                          data=melt, palette=color_, kind= 'box', 
                          legend_out =False, legend =True, height=9, aspect=1.3)
        fig.despine(right=False, top=False)
        fig.set_xticklabels(rotation=270)
        #plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig( self.out )
        plt.close()

    def Risk_Score(self, All_Score, All_risks,  _T_name, _S_name ):
        All_Score.index = [ 'S_' + str(i) for i in All_Score.index ]
        All_risks.index = [ 'S_' + str(i) for i in All_risks.index ]

        fsz = (17, 8) if All_Score.shape[0]<100 else (32, 15)
        fig = plt.figure(figsize = fsz ) #All_Score.shape[0]/4,All_Score.shape[0]/9) )
        fig.suptitle('The %s risk scores distribution'%(_T_name), x=0.5,y =0.90 )
        gs  = gridspec.GridSpec(24, 1)

        '''
        ax3 = plt.subplot(gs[0:1, :])
        ax3.pcolormesh(All_Score[[_S_name]].T, cmap=plt.cm.winter)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_ylabel(None)
        '''

        ax1 = plt.subplot(gs[0:9, :])
        ax1.grid(True)
        ax1.plot(All_Score[_T_name], 'c-' , marker='s', linewidth=1.5, markersize=4.2, label=_T_name)
        ax1.pcolor(All_Score[[_S_name]].T, cmap=plt.cm.winter)
        ax1.set_xticks([])
        ax1.set_ylabel(_T_name)
        ax1.legend()
        ax1.grid(which='both', axis='both')
        ax1.set_xlim(All_Score.index[0], All_Score.index[-1])

        ax2 = plt.subplot(gs[9:24, :] )
        ax2.plot(All_Score['risk_score_mean'], color=plt.cm.tab10(0), marker='*', linewidth=1.3, linestyle='--', markersize=4.2, label='risk_score_mean')
        ax2.plot(All_Score['risk_score_median'], color=plt.cm.tab10(1), marker='P', linewidth=1.3, linestyle=':', markersize=4.2, label='risk_score_median')
        ax2.plot(All_Score['risk_score_coefs_mean'], color=plt.cm.tab10(2) , marker='s', linewidth=1.3, linestyle='-.',markersize=4.2, label='risk_score_coefs_mean')
        ax2.scatter(All_risks.index, All_risks['risk_score'], color=plt.cm.tab10(3), marker='o', s=15.2, label='risk_score_all')
        ax2.set_xticklabels( All_Score.index, rotation='270')
        ax2.set_ylabel('risk_scores')
        ax2.set_xlim(All_Score.index[0], All_Score.index[-1])
        ax2.legend()
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

class ROCPR(Baseset):
    def ROC_CImport(self, y_true, y_pred, y_proba, model):
        y_proba = Check_proba(y_proba)
        y_trueb = Check_Binar(y_true)

        color_ = self.color_*y_proba.shape[1]
        linestyle_ = self.linestyle_*y_proba.shape[1]

        _accuracy = accuracy_score(y_true, y_pred)
        for i in range(y_proba.shape[1]):
            if y_proba.shape[1] ==2 and i == 0:
                continue
            fpr, tpr, threshold_ = roc_curve(y_trueb[:,i], y_proba[:, i])
            roc_auci = auc(fpr, tpr)
            w=tpr-fpr
            ks_score=w.max()
            ks_x=fpr[w.argmax()]
            ks_y=tpr[w.argmax()]

            label_i = 'Class %s: AUC (%0.4f), accuracy (%0.4f), KS (%0.4f) %s' %(i, roc_auci, _accuracy, ks_score, model)
            plt.plot(fpr, tpr, color=color_[i], lw=2.0, linestyle=linestyle_[i], label=label_i)
            plt.plot([ks_x,ks_x], [ks_x,ks_y],  lw=0.8, linestyle = '--', color='red', alpha=0.5)
            plt.text(ks_x,(ks_x+ks_y)/2,'  KS=%.5f'%ks_score)

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title( '%s %s ROC curve'%(y_true.name, model) )
        plt.legend(loc="lower right")
        plt.savefig( self.out )
        plt.close()

    def PR_CImport(self, y_true, y_pred, y_proba, model):
        y_proba = Check_proba(y_proba)
        y_trueb = Check_Binar(y_true)
    
        color_ = self.color_*y_proba.shape[1]
        linestyle_ = self.linestyle_*y_proba.shape[1]

        _accuracy = accuracy_score(y_true, y_pred)
        for i in range(y_proba.shape[1]):
            if y_proba.shape[1] ==2 and i == 0:
                continue
            precision, recall, threshold_pr = precision_recall_curve( y_trueb[:,i], y_proba[:,i] )
            average_precisioni = average_precision_score( y_trueb[:,i], y_proba[:,i] )

            label_i = 'Class %s: average precision (%0.4f), accuracy (%0.4f) %s' % (i, average_precisioni, _accuracy, model)
            plt.plot(recall, precision, color=color_[i], lw=2.0, linestyle=linestyle_[i], label=label_i)

        plt.plot([0, 1.05], [0, 1.05], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title( '%s %s PR curve'%(y_true.name, model) )
        plt.legend(loc="lower right")
        plt.savefig( self.out )
        plt.close()

    def ROC_MImport(self, ALL_Result, _model, _Y_name):
        color_ = self.Sequential2*len(_model)
        linestyle_ = self.linestyle_*len(_model)

        for _n, _m in enumerate(_model):
            y_true  = ALL_Result['TRUE']
            y_pred  = ALL_Result['Pred_%s_mean'%_m]
            y_proba = ALL_Result.filter(regex='.*_Prob_%s_mean'%_m, axis=1)

            y_proba = Check_proba(y_proba)
            y_trueb = Check_Binar(y_true)
            _mclass = y_proba.shape[1]
            _color_ = color_[_n]( np.linspace(0.2, 0.8, _mclass) )

            _accuracy = accuracy_score(y_true, y_pred)
            for i in range(_mclass):
                if _mclass ==2 and i == 0:
                    continue
                fpr, tpr, threshold_ = roc_curve(y_trueb[:,i], y_proba[:, i])
                roc_auci = auc(fpr, tpr)
                w=tpr-fpr
                ks_score=w.max()
                ks_x=fpr[w.argmax()]
                ks_y=tpr[w.argmax()]

                label_i = 'Class %s: AUC (%0.4f), accuracy (%0.4f), KS (%0.4f) %s' %(i, roc_auci, _accuracy, ks_score, _m)
                plt.plot(fpr, tpr, color=_color_[i], lw=1.8, linestyle=linestyle_[i], label=label_i)
                plt.plot([ks_x,ks_x], [ks_x,ks_y],   lw=0.5, linestyle=linestyle_[i], color=_color_[i], alpha=0.5)
                plt.text(ks_x,(ks_x+ks_y)/2,'  KS=%.5f'%ks_score)

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title( '%s %s ROC curve'%(_Y_name, ' '.join(_model)) )
        plt.legend(loc="lower right")
        plt.savefig( self.out )
        plt.close()

    def PR_MImport(self, ALL_Result, _model, _Y_name):
        color_ = self.Sequential2*len(_model)
        linestyle_ = self.linestyle_*len(_model)

        for _n, _m in enumerate(_model):
            y_true  = ALL_Result['TRUE']
            y_pred  = ALL_Result['Pred_%s_mean'%_m]
            y_proba = ALL_Result.filter(regex='.*_Prob_%s_mean'%_m, axis=1)

            y_proba = Check_proba(y_proba)
            y_trueb = Check_Binar(y_true)
            _mclass = y_proba.shape[1]
            _color_ = color_[_n]( np.linspace(0.2, 0.8, _mclass) )

            _accuracy = accuracy_score(y_true, y_pred)
            for i in range(_mclass):
                if _mclass ==2 and i == 0:
                    continue

                precision, recall, threshold_pr = precision_recall_curve(y_trueb[:,i], y_proba[:, i])
                average_precisioni = average_precision_score(y_trueb[:,i], y_proba[:, i])

                label_i = 'Class %s: average precision (%0.4f), accuracy (%0.4f) %s' %(i, average_precisioni, _accuracy, _m)
                plt.plot(recall, precision, color=_color_[i], lw=1.8, linestyle=linestyle_[i], label=label_i)

        plt.plot([0, 1.05], [0, 1.05], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title( '%s %s PR curve'%(_Y_name, ' '.join(_model)) )
        plt.legend(loc="lower right")
        plt.savefig( self.out )
        plt.close()

    def ROC_EImport(self, All_test_, _Y_name, _model):
        if len(All_test_[0]['TRUE']) <=2:
            return
        _times  = len(All_test_)
        colore_ = self.color_*_times
        colorm_ = self.Sequential2*_times
        linestyle_ = self.linestyle_*_times

        i = 0
        _mclass = 1
        while i < _mclass:
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            aucs = []
            accu = []
            colorm_i = colorm_[i]( np.linspace(0.1, 0.9, _times) )
            for _k, _df in enumerate(All_test_):
                y_true  = _df['TRUE']
                y_pred  = _df['Pred_%s'%_model]
                y_proba = _df.filter(regex='.*_Prob_%s'%_model, axis=1)

                y_proba = Check_proba(y_proba)
                y_trueb = Check_Binar(y_true)
                _mclass = y_proba.shape[1]
                if _mclass==2: i=1

                _accuracy = accuracy_score(y_true, y_pred)
                fpr, tpr, threshold_ = roc_curve(y_trueb[:,i], y_proba[:, i])
                roc_auci = auc(fpr, tpr)

                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auci)
                accu.append(_accuracy)

                label_i = 'Class %s: AUC (%0.4f), accuracy (%0.4f) %s' %(i, roc_auci, _accuracy, _model)
                plt.plot(fpr, tpr, color=colorm_i[_k], lw=0.75, linestyle=linestyle_[i], label=label_i)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            mean_accu = np.mean(accu)
            std_accu  = np.std(accu)

            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colore_[i], alpha=.65, label=r'Class %s: $\pm$ 1 std. dev.'%i)
            plt.plot(mean_fpr, mean_tpr,  color=colore_[i], label='Class %s: ROC (AUC = %0.4f $\pm$ %0.4f)\nClass %s: accuracy (%0.4f $\pm$ %0.4f)' % (i, mean_auc, std_auc, i, mean_accu, std_accu), lw=2, alpha=1)
            plt.plot(mean_fpr, tprs_lower, color='k', lw=1, alpha=.8)
            plt.plot(mean_fpr, tprs_upper, color='k', lw=1, alpha=.8)
            i += 1
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title( '%s %s ROC curve'%(_Y_name, _model) )
        leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.close()

    def PR_EImport(self, All_test_, _Y_name, _model):
        if len(All_test_[0]['TRUE']) <=2:
            return
        _times  = len(All_test_)
        colore_ = self.color_*_times
        colorm_ = self.Sequential2*_times
        linestyle_ = self.linestyle_*_times

        i = 0
        _mclass = 1
        while i < _mclass:
            mean_recall = np.linspace(0, 1, 100)
            prec = []
            Apre = []
            accu = []
            y_reals, y_probs= [], []
            colorm_i = colorm_[i]( np.linspace(0.1, 0.9, _times) )
            for _k, _df in enumerate(All_test_):
                y_true  = _df['TRUE']
                y_pred  = _df['Pred_%s'%_model]
                y_proba = _df.filter(regex='.*_Prob_%s'%_model, axis=1)

                y_proba = Check_proba(y_proba)
                y_trueb = Check_Binar(y_true)
                _mclass = y_proba.shape[1]
                if _mclass==2: i=1

                _accuracy = accuracy_score(y_true, y_pred)
                precision_, recall_, thresh_ = precision_recall_curve(y_trueb[:, i], y_proba[:, i])
                precision_, recall_, thresh_ = precision_[::-1], recall_[::-1], thresh_[::-1]
                average_precisioni = average_precision_score(y_trueb[:,i], y_proba[:, i])

                prec.append( np.interp(mean_recall, recall_, precision_) )
                Apre.append(average_precisioni)
                accu.append(_accuracy)

                y_reals.append(y_trueb[:,i])
                y_probs.append(y_proba[:,i])

                label_i = 'Class %s: average precision (%0.4f), accuracy (%0.4f) %s' %(i, average_precisioni, _accuracy, _model)
                plt.plot(recall_, precision_, color=colorm_i[_k], lw=0.75, linestyle=linestyle_[i], label=label_i)

            y_reals = np.concatenate(y_reals)
            y_probs = np.concatenate(y_probs)
            precision, recall, _ = precision_recall_curve(y_reals, y_probs)
            average_precision = average_precision_score(y_reals, y_probs)

            mean_prec = np.mean(prec, axis=0)
            std_prec = np.std(prec, axis=0)
            prec_upper = np.minimum(mean_prec + std_prec, 1)
            prec_lower = np.maximum(mean_prec - std_prec, 0)

            mean_Apre = np.mean(Apre)
            std_Apre  = np.std(Apre)
            mean_accu = np.mean(accu)
            std_accu  = np.std(accu)

            plt.plot(recall, precision, color='r', lw=1, alpha=1, label= 'Class %s: Overall average precision (%0.4f $\pm$ %0.4f)'%(i, average_precision, std_Apre) )
            plt.fill_between(mean_recall, prec_lower, prec_upper, color=colore_[i], alpha=.65, label=r'Class %s: $\pm$ 1 std. dev.'%i)
            plt.plot(mean_recall, mean_prec,  color=colore_[i], label='Class %s: average precision (%0.4f $\pm$ %0.4f)\nClass %s: accuracy (%0.4f $\pm$ %0.4f)' % (i, mean_Apre, std_Apre, i, mean_accu, std_accu), lw=2, alpha=1)
            plt.plot(mean_recall, prec_lower, color='k', lw=1, alpha=.8)
            plt.plot(mean_recall, prec_upper, color='k', lw=1, alpha=.8)
            i += 1

        plt.plot([0, 1.05], [0, 1.05], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title( '%s %s PR curve'%(_Y_name, _model) )
        leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.close()

class Plots(Baseset):
    def Plot_ROC(self):
        (ROC_data, Y) = self.array

        color_ = self.color_
        linestyle_ = self.linestyle_*len(ROC_data)
        for i,j in enumerate(ROC_data):
            fpr,tpr,roc_auc,Y_test_accuracy, N = j
            plt.plot(fpr, tpr, color=color_[i], lw=1.3, linestyle = linestyle_[i], label='ROC curve (area = %0.4f), accuracy (%0.4f)  %s '%(roc_auc,Y_test_accuracy, N))
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(Y + ' Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(self.out + Y + 'ROC.pdf')
        plt.close()

    def Plot_Test(self):
        (Y_tespred_df,) = self.array
        title    = Y_tespred_df.columns
        X_sampel = Y_tespred_df.index.tolist()
        Y = title[0]

        color_ = self.color_*len(title)
        linestyle_ = self.linestyle_*len(title)
        markertyle_ = self.markertyle_*len(title)

        plt.plot(Y_tespred_df[Y], marker ='o',markersize=3.5, color='k', label='true y',lw=1.5)
        for i,Yj  in enumerate(title[1:]): 
            plt.plot(Y_tespred_df[Yj], marker=markertyle_[i], markersize=3.2, color=color_[i], linestyle=linestyle_[i], lw=1.3, label=Yj)
        plt.title( Y + ' regression result comparison')
        plt.legend(loc='upper left')
        plt.ylabel( Y + ' real and predicted value')
        plt.xticks( rotation='270')
        plt.savefig(self.out +  Y + '_Predict.pdf')
        plt.close()

    def Feature_SorceCV(self):
        (select_Sco, select_Fea, N, Y, cv) = self.array

        color_ = self.color_*select_Sco.shape[1]
        linestyle_ = self.linestyle_*select_Sco.shape[1]
        markertyle_ = self.markertyle_*select_Sco.shape[1]

        Sco_mean = select_Sco.mean(axis=1)
        Sco_Std  = select_Sco.std(axis=1)
        Fea_mean = select_Fea.mean()
        Fea_Std  = select_Fea.std()
        Fea_min  = (Fea_mean-1*Fea_Std) if (Fea_mean-1*Fea_Std) > 0 else 1
        Fea_max  = (Fea_mean + 1*Fea_Std) if (Fea_mean + 1*Fea_Std)<= select_Sco.shape[0] else select_Sco.shape[0]
        for i in range(select_Sco.shape[1]):
            plt.plot(select_Sco[i], marker=markertyle_[i], markersize=3.2, color=color_[i], linestyle=linestyle_[i], lw=1.0, label='')
            plt.vlines(select_Fea[i], 0, 1, color=color_[i], linestyle=linestyle_[i], lw=1.2, label='')
        plt.plot(Sco_mean , 'k-', lw=1.5, label='mean_score')
        plt.vlines(Fea_mean, 0, 1, lw=1.5, label='')
        plt.fill_between(Sco_mean.index, Sco_mean-1*Sco_Std, Sco_mean + 1*Sco_Std, color='grey', alpha=0.25, label=r'$\pm$ 1 std. dev.')
        plt.axvspan(Fea_min, Fea_max , alpha=0.3, color='grey', label='')

        plt.title('The %s features score of %s with %s estimator'%(cv, Y, N))
        plt.legend(loc='lower right')
        plt.ylabel('The %s features score'%cv)
        plt.xlabel('n featrues')
        plt.ylim([0.0, 1.0])
        plt.savefig('%sClass.Regress_%s_%s_%s_features_score.pdf' % (self.out, Y, N ,cv))
        plt.close()

    def Feature_CoefSC(self):
        (select_Sco, select_Fea, N, Y, cv) = self.array

        Sco_mean = np.array(select_Sco).mean()
        Sco_Std  = np.array(select_Sco).std()
        Fea_mean = select_Fea.mean()
        Fea_Std  = select_Fea.std()

        Sco_min  = Sco_mean - Sco_Std
        Sco_max  = Sco_mean + Sco_Std

        Fea_min  = (Fea_mean-1*Fea_Std) if (Fea_mean-1*Fea_Std) > 0 else 1
        Fea_max  = (Fea_mean + 1*Fea_Std)

        plt.plot(select_Fea, select_Sco, 'g--', marker='*', linewidth=1.3, markersize=4.2, label='')
        plt.hlines(Sco_mean, 0, Fea_max+1, lw=1.5, label='mean_score')
        plt.vlines(Fea_mean, 0, 1, lw=1.5, label='mean_featrues')
        plt.axvspan(Fea_min, Fea_max , alpha=0.3, color='red', label='')
        plt.axhspan(Sco_min, Sco_max , alpha=0.3, color='blue', label='')

        plt.title('The %s features score of %s with %s estimator'%(cv, Y, N))
        plt.legend(loc='lower right')
        plt.ylabel('The %s features score'%cv)
        plt.xlabel('n featrues')
        plt.ylim([0.0, 1.0])
        plt.savefig('%sClass.Regress_%s_%s_%s_features_score.pdf' % (self.out, Y, N ,cv))
        plt.close()

    def ROC_Predict_Import(self):
        (All_ROC, N, Y) = self.array

        color_ = self.color_*len(All_ROC)
        linestyle_ = self.linestyle_*len(All_ROC)
        markertyle_ = self.markertyle_*len(All_ROC)

        for i in range(len(All_ROC)):
            fpr, tpr, roc_auc, Y_test_accuracy, N = All_ROC[i]
            label_i = 'ROC curve (area = %0.4f), accuracy (%0.4f) %s' % (roc_auc, Y_test_accuracy, N)
            plt.plot(fpr, tpr, color=color_[i], lw=1.3, linestyle=linestyle_[i], label=label_i)

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(Y + '_' + N + ' ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('%sClass_%s_%s_Predict_ROC_curve.pdf' % (self.out, Y, N))
        plt.close()
