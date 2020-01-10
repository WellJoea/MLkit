def _predict_proba_lr(self, X):
    """
    Probability estimation for OvR logistic regression.
    Positive class probabilities are computed as
    1. / (1. + np.exp(-self.decision_function(X)));
    multiclass is handled by normalizing that over all classes.
    """
    from scipy.special import expit
    prob = self.decision_function(X)
    expit(prob, out=prob)
    if prob.ndim == 1:
        return np.vstack([1 - prob, prob]).T
    else:
        # OvR normalization, like LibLinear's predict_probability
        prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
    return prob

def linestyle():
    import numpy as np
    import matplotlib.pyplot as plt

    linestyle_str = [
        ('solid', 'solid'),      # Same as (0, ()) or '-'
        ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
        ('dashed', 'dashed'),    # Same as '--'
        ('dashdot', 'dashdot')]  # Same as '-.'

    linestyle_tuple = [
        ('loosely dotted',        (0, (1, 10))),
        ('dotted',                (0, (1, 1))),
        ('densely dotted',        (0, (1, 1))),

        ('loosely dashed',        (0, (5, 10))),
        ('dashed',                (0, (5, 5))),
        ('densely dashed',        (0, (5, 1))),

        ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        ('densely dashdotted',    (0, (3, 1, 1, 1))),

        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


    def plot_linestyles(ax, linestyles):
        X, Y = np.linspace(0, 100, 10), np.zeros(10)
        yticklabels = []

        for i, (name, linestyle) in enumerate(linestyles):
            ax.plot(X, Y+i, linestyle=linestyle, linewidth=1.5, color='black')
            yticklabels.append(name)

        ax.set(xticks=[], ylim=(-0.5, len(linestyles)-0.5),
            yticks=np.arange(len(linestyles)), yticklabels=yticklabels)

        # For each line style, add a text annotation with a small offset from
        # the reference point (0 in Axes coords, y tick value in Data coords).
        for i, (name, linestyle) in enumerate(linestyles):
            ax.annotate(repr(linestyle),
                        xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
                        xytext=(-6, -12), textcoords='offset points',
                        color="blue", fontsize=8, ha="right", family="monospace")


    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]},
                                figsize=(10, 8))

    plot_linestyles(ax0, linestyle_str[::-1])
    plot_linestyles(ax1, linestyle_tuple[::-1])

    plt.tight_layout()
    plt.show()

class Featuring():
    def Clfing(self, Xdf, Ydf, SSS):
        return(Xdf, Ydf)
    def Coef_SC(self):
        (Xdf, Ydf) = self.array
        select_Sup = []
        select_Sco = []

        for _ in range(self.arg.SelectCV_rep):   #pool
            for _x in range( len(self.SPLIT) ) :
                SSS = self.SPLIT[_x]
                clf, df_import = self.Clfing(Xdf, Ydf, SSS)
                best_score_ = clf.best_score_
                if self.arg.removemix:
                    (best_score_, df_import) = self.Remove_mix(df_import, clf.best_score_, Xdf, Ydf, SSS)
                select_feat = pd.DataFrame([[1]]*df_import.shape[0], index=df_import.index)
                select_Sco.append(best_score_)
                select_Sup.append(select_feat)
                self.log.CIF('COEFCV: %s -> %s , split %.2f '%(self.arg.model, self.model, n))

        select_Sup = pd.concat(select_Sup,axis=1,sort=False).fillna(0).astype(int)
        select_Sup.columns =  [ '%s_%s'%(self.model, i) for i in range(select_Sup.shape[1]) ]
        Openf('%s%s_Class_Regress_SFSCV_%s_ranking.xls'%(self.arg.output, Ydf.name, self.model),select_Sup).openv()

        select_feature = select_Sup.sum(0)
        MPlot('%s%s_Class.Regress_COEFS_%s_Fscore.pdf'%(self.arg.output, Ydf.name, self.model)).Feature_Sorce(
            select_Sco, select_feature.values, self.model, Ydf.name, 'COEFS' )

        return( select_Sup )
    def Remove_mix(self, df_import_X, best_score_, Xdf, Ydf, SSS):
        df_import = df_import_X.copy()
        X_train   = Xdf.copy()
        All_score = [best_score_]
        All_import= [df_import]

        while (len(df_import) > self.arg.rmfisrt) or (round(min(df_import[0].abs()), 2) < self.arg.rmsecond):
            if (round(min(df_import[0].abs()), 2) < self.arg.rmthird):
                df_import_tmp = df_import[df_import[0].abs().round(2) >= self.arg.rmthird]
                X_train = X_train[df_import_tmp.index]
            elif (round(min(df_import[0].abs()), 2) < self.arg.rmsecond):
                df_import_tmp = df_import[ df_import[0].abs() != df_import[0].abs().min() ]
                X_train = X_train[df_import_tmp.index]
            df_import_last = df_import.copy()

            clf, df_import = self.Clfing(X_train, Ydf, SSS)

            All_score.append(clf.best_score_)
            All_import.append(df_import)

            if (len(df_import_last) == len(df_import)):
                improt_cat = pd.concat([df_import_last,df_import],sort=False,axis=1).fillna(0)
                if pearsonr(improt_cat.iloc[:,0], improt_cat.iloc[:,1])[0] >0.9:
                    break
        All_score_n = All_score.reverse()
        best_index  = len(All_score_n) - All_score_n.index(max(All_score_n)) -1
        return(All_score[best_index], All_import[best_index])

class MPlot(Baseset):
    def Feature_Import(self, All_score, Best_index, All_import, N, Y):
        plt.figure(figsize=(13,10))
        color_ = self.color_*len(All_score)
        linestyle_ = self.linestyle_*len(All_score)
        markertyle_ = self.markertyle_*len(All_score)

        All_raw = All_import.filter(regex=r'^[0-9]+$',axis=1)
        column = sorted(set(All_raw.columns))
        if len(column) == 1:
            for i in range(len(All_score)): 
                Score_name =  All_score[i]
                if i == Best_index:
                    Score_name += ' BEST'
                plt.plot(All_import.iloc[:,i], marker=markertyle_[i], markersize=3.2, color=color_[i], 
                         linestyle=linestyle_[i], lw=1.3, label=Score_name)
                #x_pos = All_import[i]
                #plt.text(text(0.5, 0.5, 'matplotlib', horizontalalignment='right',verticalalignment='center')
            plt.plot(All_import['0_mean'], 'k-.', lw=1.5, label='mean_import',alpha= 0.9 )
            plt.fill_between(All_import.index, All_import['0_mean']-1*All_import['0_std'], All_import['0_std'] +
                             1*All_import['0_mean'], color='grey', alpha=0.3, label=r'$\pm$ 1 std. dev.')
        else:
            for n in column:
                for i in range(len(All_score)):
                    Score_name =  All_score[i]
                    if i == Best_index:
                        Score_name += ' BEST'
                    plt.plot(All_import[n].iloc[:,i], marker=markertyle_[n], markersize=3.2, 
                             color=color_[n], linestyle=linestyle_[n], lw=1.1, label=str(n) + ' ' + Score_name)
                plt.plot(All_import['%s_mean'%n], color=color_[n], lw=1.5, label='%s mean_import'%n)
                plt.fill_between(All_import.index, All_import['%s_mean'%n]-1*All_import['%s_std'%n], All_import['%s_std'%n] +
                                1*All_import['%s_mean'%n], color=color_[n], alpha=0.25, label=r'%s $\pm$ 1 std. dev.'%n)

        if (All_raw.values.min()>=0) & (All_raw.values.max()<=1):
            plt.ylim([0.0, 1.00])
        plt.title( Y + ' ' + N + '  featrue/coef importance ROC Accuracy')
        plt.legend(loc='upper right')
        plt.ylabel( Y + ' ' + N + ' featrue/coef importance values')
        plt.xticks( rotation='270')
        leg = plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), numpoints=1,  title='ROCs and Accuracy')
        plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.close()

class ROCPR(Baseset):
    def ROC_KS(self, y_true, y_proba, y_pred, model):
        Y_name = y_true.name
        _accuracy  = accuracy_score( y_true , y_pred )
        fpr, tpr, thresholds=roc_curve(y_true,y_proba,pos_label=1)
        roc_auc =auc(fpr,tpr)
        w=tpr-fpr
        ks_score=w.max()
        ks_x=fpr[w.argmax()]
        ks_y=tpr[w.argmax()]

        plt.plot(fpr, tpr, lw=1.4, color= self.color_[0], linestyle = '-.', label='AUC (%0.2f), accuracy (%0.2f), KS (%0.2f)  %s'%(roc_auc, _accuracy, ks_score, model))
        plt.plot([ks_x,ks_x], [ks_x,ks_y], lw=2.0, linestyle = '--', color='red')
        plt.text(ks_x,(ks_x+ks_y)/2,'  KS=%.5f'%ks_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(Y_name + '_' + model + ' ROC curve')
        plt.legend(loc="lower right")
        plt.savefig( self.out )
        plt.close()

    def ROC_COMpare(self, All_dt, Y_name):
        color_ = self.color_*len(All_dt)
        linestyle_ = self.linestyle_*len(All_dt)
        markertyle_ = self.markertyle_*len(All_dt)

        for i, _modle_i  in enumerate(All_dt):
            M_name, Y_true, Y_proba, Y_predic = _modle_i
            _accuracy = accuracy_score(Y_true, Y_predic)
            fpr, tpr, threshold_roc = roc_curve(Y_true, Y_proba)
            roc_auc = auc(fpr, tpr)

            w=tpr-fpr
            ks_score=w.max()
            ks_x=fpr[w.argmax()]
            ks_y=tpr[w.argmax()]

            label_i = 'ROC AUC (%0.2f), accuracy (%0.2f), KS (%0.2f)  %s' % (roc_auc, _accuracy, ks_score, M_name)
            plt.plot(fpr, tpr, color=color_[i], lw=1.8, linestyle=linestyle_[i], label=label_i)
            plt.plot([ks_x,ks_x], [ks_x,ks_y],  lw=1.4, linestyle = '--', color=color_[i])
            plt.text(ks_x,(ks_x+ks_y)/2,'  KS=%.3f'%ks_score)

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title( Y_name + '_compare_ROC curve')
        plt.legend(loc="lower right")
        plt.savefig( self.out )
        plt.close()

    def PR_COMpare(self, All_dt, Y_name):
        color_ = self.color_*len(All_dt)
        linestyle_ = self.linestyle_*len(All_dt)
        markertyle_ = self.markertyle_*len(All_dt)

        for i, _modle_i  in enumerate(All_dt):
            M_name, Y_true, Y_proba, Y_predic = _modle_i
            _accuracy = accuracy_score(Y_true, Y_predic)

            precision, recall, threshold_pr = precision_recall_curve( Y_true, Y_proba)
            average_precisioni = average_precision_score( Y_true, Y_proba )

            label_i = 'mean PR (%0.2f), accuracy (%0.2f) %s' % ( average_precisioni, _accuracy, M_name)
            plt.plot(recall, precision, color=color_[i], lw=2.0, linestyle=linestyle_[i], label=label_i)

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title( Y_name + '_compare_PR curve')
        plt.legend(loc="lower left")
        plt.savefig( self.out )
        plt.close()

    def ROC_Import(self, All_test_, Best_index, N, Y ):
        color_ = self.color_*len(All_test_)
        linestyle_ = self.linestyle_*len(All_test_)
        markertyle_ = self.markertyle_*len(All_test_)

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        accu = []
        for i in range(len(All_test_)):
            _test_df = All_test_[i]
            Y_test  = _test_df['True']
            Y_tespred_pro = _test_df.filter(regex=("^Prob_"))
            Y_tespred = _test_df['Predict']

            Y_test_accuracy = accuracy_score(Y_test, Y_tespred)
            fpr, tpr, threshold_roc = roc_curve(Y_test, Y_tespred_pro['Prob_1'])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            accu.append(Y_test_accuracy)
            label_i = 'ROC AUC (%0.2f), accuracy (%0.2f) %s' % (roc_auc, Y_test_accuracy, N)
            if i == Best_index:
                label_i += ' BEST'
            plt.plot(fpr, tpr, color=color_[i], lw=1.3, linestyle=linestyle_[i], label=label_i)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_accu = np.mean(accu)
        std_accu  = np.std(accu)
        plt.plot(mean_fpr, mean_tpr, color='k', label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)\nMean accuracy (%0.2f $\pm$ %0.2f)' % ( mean_auc, std_auc, mean_accu, std_accu ), lw=2, alpha=.9)
        #plt.plot(mean_fpr, tprs_lower, color='k', lw=1, alpha=.8)
        #plt.plot(mean_fpr, tprs_upper, color='k', lw=1, alpha=.8)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'$\pm$ 1 std. dev.')

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(Y + '_' + N + ' ROC curve')
        plt.legend(loc="lower right")
        plt.savefig( self.out )
        plt.close()

    def PR_Import(self, All_test_, Best_index, N, Y):
        color_ = self.color_*len(All_test_)
        linestyle_ = self.linestyle_*len(All_test_)
        markertyle_ = self.markertyle_*len(All_test_)
        for i in range(len(All_test_)):
            _test_df = All_test_[i]
            Y_test  = _test_df['True']
            Y_tespred_pro = _test_df.filter(regex=("^Prob_"))
            Y_tespred = _test_df['Predict']

            Y_test_accuracy = accuracy_score(Y_test, Y_tespred)
            precision, recall, threshold_pr = precision_recall_curve(Y_test, Y_tespred_pro['Prob_1'])
            average_precisioni = average_precision_score( Y_test, Y_tespred_pro['Prob_1'])
            label_i = 'mean PR (%0.2f), accuracy (%0.2f) %s' % (average_precisioni, Y_test_accuracy, N)
            if i == Best_index:
                label_i += ' BEST'
            plt.plot(recall, precision, color=color_[i], lw=1.0, linestyle=linestyle_[i], label=label_i)

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(Y + '_' + N + ' PR curve')
        plt.legend(loc="lower left")
        plt.savefig( self.out )
        plt.close()
