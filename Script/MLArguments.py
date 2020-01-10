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

import argparse
import os

def Args():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawDescriptionHelpFormatter,
                prefix_chars='-+',
                conflict_handler='resolve',
                description="\nThe traditional machine learning analysis based on sklearn package:\n",
                epilog='''\
Example:
1. python MLkit.py Auto    -i data.traintest.txt -g group.new.txt -p data.predict.txt -o testdt/ -m DT
2. python MLkit.py Auto    -i data.traintest.txt -g group.new.txt -o testdt/ -m DT -pc -s S M
2. python MLkit.py Common  -i data.traintest.txt -g group.new.txt -o testdt/ -m DT
3. python MLkit.py Fselect -i data.traintest.txt -g group.new.txt -o testdt/ -m DT -s S
4. python MLkit.py Predict -p data.predict.txt   -g group.new.txt -o testdt/ -m DT.''')

    parser.add_argument('-V','--version',action ='version',
                version='MLkit version 0.1')

    subparsers = parser.add_subparsers(dest="commands",
                    help='machine learning models help.')
    P_Common   = subparsers.add_parser('Common',conflict_handler='resolve', #add_help=False,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    help='The common parameters used for other models.')
    P_Common.add_argument("-i", "--input",type=str,
                    help='''the input train and test data file with dataframe format  by row(samples) x columns (features and Y). the sample column name must be Sample.
''')
    P_Common.add_argument("-g", "--group",type=str,#required=True,
                    help='''the group file tell the featues, groups and variable type, which has Variables, Group, Type columns. Only continuous and discrete variables are supported in variable type. Onehot variables is coming.''')
    P_Common.add_argument("-o", "--outdir",type=str,default=os.getcwd(),
                    help="output file dir, default=current dir.")
    P_Common.add_argument("-m", "--model",type=str, default='XGB',
                    help='''the model you can used for ML.
You can choose the models as follows:
classification:......++++++++++++++++++++++
**RF.................RandomForestClassifier
**GBDT...............GradientBoostingClassifier
**XGB................XGBClassifier(+LR/LRCV)
**MLP................MLPClassifier
**DT.................DecisionTreeClassifier
**AdaB_DT............AdaBoostClassifier(DT)
**LinearSVM..........LinearSVC(penalty='l1')
**LinearSVMil2.......LinearSVC(penalty='l2')
**SVMlinear..........SVC(kernel="linear")
**SVM................SVC(no linear)
**nuSVMrbf...........NuSVC(kernel='rbf')
**SGD................SGDClassifier
**KNN................KNeighborsClassifier
**RNN................RadiusNeighborsClassifier
**MNB................MultinomialNB
**CNB................ComplementNB
**BNB................BernoulliNB
**GNB................GaussianNB
**LR.................LogisticRegression
**LRCV...............LogisticRegressionCV
Regressioin:.........+++++++++++++++++++++
**RF.................RandomForestRegressor
**GBDT...............GradientBoostingRegressor
**XGB................XGBRegressor
**MLP................MLPRegressor
**DT.................DecisionTreeRegressor
**AdaB_DT............AdaBoostRegressor(DT)
**LinearSVM..........LinearSVR
**SVMlinear..........SVR(kernel="linear")
**SVMrbf.............SVR(kernel="rbf")
**nuSVMrbf...........NuSVC(kernel='rbf')
**SGD................SGDRegressor
**KNN................KNeighborsRegressor
**RNN................RadiusNeighborsRegressor
**LG.................LogisticRegression
**LassCV.............LassoCV
**Lasso..............Lasso
**ENet...............ElasticNet
**ENetCV.............ElasticNetCV
''')
    P_Common.add_argument("-am", "--Addmode", type=str, default='LinearSVM',
                    help='''use additional model to adjust modeling in RF, GBDT and XGB estimators. N means no addition.''')
    P_Common.add_argument("-t", "--pool",type=int,default=20,
                    help="the CPU numbers that can be used.")
    P_Common.add_argument("-nt", "--n_iter", type=int, default= 2500,
                    help="Number of parameter settings that are sampled in RSCV. n_iter trades off runtime vs quality of the solution.")
    P_Common.add_argument("-mr", "--MissRow",type=float,default=0.8,
                    help="The rows missvalues rate, if high than the 1-value, the rows will be removed.")
    P_Common.add_argument("-mc", "--MissCol",type=float,default=0.8,
                    help="The columns missvalues rate, if high than the 1-value, the columns will be removed.")
    P_Common.add_argument("-mq", "--Mtfreq",type=float, default=0.98,
                    help="The columns mode number frequency, if high than the 1-value, the columns will be removed."
                         "If the feature matrix is sparse, plase set a higher vales, such as the maximum 1.")
    P_Common.add_argument("-mv", "--MissValue", type=str,default='median', choices=['mean', 'median', 'most_frequent', 'constant'],
                    help="Imputation transformer for completing missing values.")
    P_Common.add_argument("-fv", "--FillValue", default=None,
                    help='''When MissValue == 'constant, fill_value is used to replace all occurrences of missing_values. Idefault fill_value will be 0 when imputing numerical data and missing_value for strings or object data types.''')
    P_Common.add_argument("-pp", "--PairPlot", action='store_true', default=False,
                    help='''the pairplot of pairwise features, it is not recommended when the number of features is large 20.''')
    P_Common.add_argument("-nj", "--n_job", type=int,default=-1,
                    help="Number of cores to run in parallel while fitting across folds.")
    P_Common.add_argument("-s", "--scaler", nargs='+', default = ['S'], #['S','M'],
                    help='''the feature standardization, you can chose: RobustScaler(R), StandardScaler(S),MinMaxScaler(M) or not do (N).''')
    P_Common.add_argument("-rs", "--refset", type=str, default = 'train', choices=['train','predict','all'], 
                        help='''the data set used for feature standardization, you can chose: train(train data), predict(predict data), all (train + predict data)''')
    P_Common.add_argument("-qr", "--QuantileRange", type=int, nargs='+', default=[10,90],
                    help="Quantile range used to calculate scale when use the RobustScaler method.")
    P_Common.add_argument("-ps", "--pcathreshold", type=float, default=0.95,
                    help='''the threshold value of sum of explained variance ratio use for pca plot and ML training and testing.''')

    P_fselect  = subparsers.add_parser('Fselect', conflict_handler='resolve', add_help=False)
    P_fselect.add_argument("-sb", "--SelectB", nargs='*', default=['VTh', 'MI', 'AUC', 'ANVF', 'FISH', 'Chi2C', 'RKs', 'MWU', 'TTI', 'PEAS', 'SPM', 'KDT', 'LR' ],
                    help='''use statistic method to select the top highest scores features with SelectKBest.
You can choose the models as follows:
classification:......++++++++++++++++++++++
**VTh..................VarianceThreshold (score)
**Chi2.................chi2 (P)
**MI...................mutual_info_classif (score)
**ANVF.................f_classif (P)
**AUC..................AUC (score)
**FISH.................fisher_exact (P, removed)
**Chi2C................chi2_contingency (P)
**WC...................wilcoxon (P)
**RS...................ranksums (P)
**MWU..................mannwhitneyu (P)
**TTI..................ttest_ind (P)
Regressioin:.........++++++++++++++++++++++
**VTh..................VarianceThreshold
**ANVF.................f_regression
**PEAS  ...............pearsonr
**MI...................mutual_info_classif
''')
    P_fselect.add_argument("-kb", "--SelectK", type=float, nargs=4, default=[0.75, 0.70, 1, 1],
                    help="the SelectKBest feature selection K best number, you can use int or float in 'VTh', 'MI', 'AUC' , P.")
    P_fselect.add_argument("-st", "--SelectS", type=float, nargs=4, default=[ 0.0005, 0.0005, 0.59, 0.1 ],
                    help="the SelectKBest feature selection threshold in 'VTh', 'MI', 'AUC', P.")
    P_fselect.add_argument("-pt", "--pvaluetype", type=str, default='P', choices=['P','Padj'],
                    help="the SelectKBest feature selection Type in P. you can use P values or P adjust pdrbh values.")
    P_fselect.add_argument("-ft", "--Fisher", type=int, default=600,
                    help="Beacuse of slow computing speed, fisher exact test in scipy module will be discarded if features number is larger than the default value.")
    P_fselect.add_argument("-kr", "--SelectR", type=float, default=0.4,
                    help="the SelectKBest True Ratio in all statistic methods in SelectB paremetors.")
    P_fselect.add_argument("-se", "--SelectE", nargs='*', default=['RFECV'],
                    help='''whether use machine learning estimators to select features with some of RFECV(RFECV), SFSCV(SequentialFeatureSelector), CoefSelect(gridsearch + coef) method.
the estimator in RFECV and CoefSelect is insteaded as follows: 
GNB, KNN, RNN, SVM, SVMrbf -> SVMlinear 
MLP, AdaB_DT -> XGB
MNB, CNB, BNB, GNB -> MNB.
RFECV(RFECV) is based on RFECV(feature ranking with recursive feature elimination and cross-validated selection of the best number of features).
SFSCV is based on mlxtend SequentialFeatureSelector package.
CoefSC(gridsearch + coef) removes features that cannot meet the contition by step.
''')
    P_fselect.add_argument("-gt", "--CVtime", type=int, default=20,
                    help="the repeat time in each model in feature selection in SelectE.")
    P_fselect.add_argument("-vm", "--CVmodel", type=str,default='SSS', choices=['SSS','LOU'],
                    help='''the cross validation model in GridsearchCV, RFECV and SFSCV.''')
    P_fselect.add_argument("-sm", "--specifM", type=str, nargs='*', default=[], #['XGB','LinearSVM'],
                    help="the specifed models use for feature selection instead of RFECV or/and SFSCV default model, such as you can use XGB and LinearSVM simultaneously. ")
    P_fselect.add_argument("-set", "--set", type=str, default='parallel', choices=['parallel','serial'],
                    help="use serial or parallel set decese multiple specifed models. if parallel, the final the features threshold ratio is deceded by the max values in all rates.")
    P_fselect.add_argument("-sp", "--SelectCV_rep",type=int,default=1,
                    help="the repetition times for using RFECV or SFSCV in one split set.")
    P_fselect.add_argument("-rr", "--RFE_rate", type=float, default=25,
                    help="the features threshold ratio selected by RFECV models. value >1: features number; 0 <value< = 1: features number rate; value=0: only remove the non-zero features.")
    P_fselect.add_argument("-sr", "--SFS_rate", type=float, default=0.3,
                    help="the features threshold ratio selected by SFSCV models. value >1: features number; 0 <value< = 1: features number rate; value=0: only remove the non-zero features.")
    P_fselect.add_argument("-cr", "--CFS_rate", type=float, default=0.6,
                    help="the features threshold ratio selected by CFSCV models. value >1: features number; 0 <value< = 1: features number rate; value=0: only remove the non-zero features.")
    P_fselect.add_argument("-kf", "--k_features", nargs='+', default = ['best'],
                    help='''Number of features to select. the string 'best' or 'parsimonious' and a tuple containing a min and max ratio can be provided, sush as [0.2, 0.75]. "best":  the feature subset with the best cross-validation performance. "parsimonious" : the smallest feature subset that is within one standard error of the cross-validation performance will be selected.''')
    P_fselect.add_argument("-rm", "--removemix", action='store_true' , default=False,
                    help='''whether remove the mix improtance/coef for trainning again under Coef_Select_C method.''')
    P_fselect.add_argument("-rmf", "--rmfisrt", type=float,default=15,
                    help="the final features number under choosing removemix  under CoefSelect model. ")
    P_fselect.add_argument("-rms", "--rmsecond",type=float,default=0.05,
                    help='''remove one the importance features lower the argment by step under choosing removemix under CoefSelect model.''')
    P_fselect.add_argument("-rmt", "--rmthird",type=float,default=0.01,
                    help='''remove all the importance features lower the argment by step under choosing removemix under CoefSelect model.''')
    P_Fselect  = subparsers.add_parser('Fselect',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_fselect],
                    help='Feature selection from standardized data.')

    P_fitting  = subparsers.add_parser('Fitting',conflict_handler='resolve', add_help=False)
    P_fitting.add_argument("-cm", "--CVfit", type=str,default='SSA',
                    help='''the cross validation model:
**SSA..................StratifiedShuffleSplit() + StratifiedKFold()
**SSS..................StratifiedShuffleSplit()
**SKF..................StratifiedKFold()
**RSKF.................RepeatedStratifiedKFold()
**RKF..................RepeatedKFold()
**LPO..................LeavePOut()
**LOU..................LeaveOneOut()
''')
    P_fitting.add_argument("-tz", "--testS",type=float,default=0.3,
                    help="the test size for cross validation when using StratifiedShuffleSplit.")
    P_fitting.add_argument("-sc", "--SearchCV",type=str, nargs='?', default='GSCV',choices=['GSCV','RSCV',],
                    help="the hyperparameters optimization method. You also can set it none to discard the method.")
    P_fitting.add_argument("-rt", "--Repeatime", type=int,default=1,
                    help='''the repeat time of modeling. Suggest change to a lagger value with a none value of SearchCV.''')
    P_fitting.add_argument("-cz", "--GStestS",type=float,default=0.4,
                    help="the test size for cross validation when using StratifiedShuffleSplit in gridsearchCV.")
    P_fitting.add_argument("-cv", "--crossV",type=int,default=10,
                    help="the cross validation times when using StratifiedShuffleSplit.")
    P_fitting.add_argument("-lp", "--leavP",type=int,default=1,
                    help="the size of the test sets in Leave-P-Out cross-validator.")
    P_fitting.add_argument("-pc", "--pca", action='store_true' , default=False,
                    help='''whether use pca matrix as final set for trainning as testing.''')
    P_fitting.add_argument("-cc", "--calib", action='store_true' , default=False,
                    help='''whether use CalibratedClassifierCV to calibrate probability with isotonic regression or sigmoid.''')
    P_fitting.add_argument("-si", "--calibme", type=str, default='sigmoid', choices=['isotonic','sigmoid'],
                    help="the CalibratedClassifierCV method, you can choose isotonic and sigmoid.")
    P_Fitting  = subparsers.add_parser('Fitting',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_fitting],
                    help='Fitting and predicting the training and testing set from estimators.')

    P_predict  = subparsers.add_parser('Predict', conflict_handler='resolve',add_help=False,)
    P_predict.add_argument("-p", "--predict",type=str,
                    help="the predict matrix file.")
    P_predict.add_argument("-x", "--modelpath",type=str,
                    help="the model path used for prediction.")
    P_predict.add_argument("-y", "--out_predict",type=str,
                    help="the predict result file path.")
    P_predict.add_argument("-z", "--out_header",type=str, default='',
                    help="the predict result file header.")
    P_Predict  = subparsers.add_parser('Predict',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fitting, P_predict],
                    help='predict new data from fittting process model.')

    P_scoring  = subparsers.add_parser('Score', conflict_handler='resolve',add_help=False,)
    P_scoring.add_argument("-mi", "--max_intervals", type=int,  default=15,
                    help="Max intervals of boxing in chimerge.")
    P_scoring.add_argument("-bp", "--basepoints", type=int,  default=60,
                    help="The basepoints.")
    P_scoring.add_argument("-bd", "--baseodds", type=float,  default=0.05,
                    help="The good bad rate, such as 0.05 = 1:20.")
    P_scoring.add_argument("-pd", "--PDO", type=float,  default=3,
                    help="The PDO.")
    P_scoring.add_argument("-ms", "--modscore", type=str, default='XGB',
                    help="The score model: you can  use LRCV, LR, GBDT, XGB, SVMlinear, .....")
    P_Scoring  = subparsers.add_parser('Score',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fselect,P_fitting, P_scoring, P_predict],
                    help='Scoring the samples.')

    P_Autopipe = subparsers.add_parser('Auto', conflict_handler='resolve', prefix_chars='-+',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fselect, P_fitting, P_scoring, P_predict],
                    help='the auto-processing for all: standardization, feature selection, Fitting and/or Prediction.')
    P_Autopipe.add_argument("+P", "++pipeline",nargs='+',
                    help="the auto-processing: standardization, feature selection, Fitting and/or Prediction.")
    P_Autopipe.add_argument('+M','++MODEL' , nargs='+', type=str, default=['Standard'],
                    help='''Chose more the one models from Standard, Fselect,Fitting and Predict used for DIY pipline.''')

    '''
    #for i in dir(parser): print(i, eval('parser.%s'%i))
    #AA = copy.deepcopy(parser)
    P_Diy      = subparsers.add_parser('DIY', conflict_handler='resolve',
                    parents= [P_Common] + [eval('P_'+ i.lower()) for i in copy.deepcopy(parser).parse_args().MODEL],
                    help='DIY pipline')
    if parser.parse_args().MODEL:
        Parents += [ eval('P_'+ i.lower())
                      for i in parser.parse_args().ALLMODEL
                      for j in parser.parse_args().MODEL
                      if j==i ]'''

    args  = parser.parse_args()
    return args
