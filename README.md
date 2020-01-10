# Machine Learning Kit
The [**MLkit**](https://github.com/WellJoea/MLkit.git) is based on traditional machine learning based on sklearn.<br/>
You can ues most of the estimator in sklearn in MLkit and implement auto standardization, feature selection, Fitting and Prediction.<br/>
___
## Installation
### Dependencies
<pre><code>        'joblib >= 0.13.2 ',
        'matplotlib >= 3.0.3',
        'mlxtend >= 0.16.0',
        'numpy >= 1.16.4',
        'pandas >= 0.24.2',
        'scikit-learn >= 0.21.2',
        'scikit-plot >= 0.3.7',
        'scipy >= 1.3.0',
        'seaborn >= 0.9.0',
        'sklearn-pandas >= 1.8.0',
</code></pre>
### User installation
- download: https://github.com/WellJoea/MLkit.git
- cd MLkit
- python setup.py install
___
## useage
**MLkit.py -h**<br/>
**usage:** MLkit.py [-h] [-V] {Common,Fselect,Fitting,Predict,Score,Auto} ...<br/>

The traditional machine learning analysis is based on sklearn package:<br/>
### **1. positional arguments:**
<p>{Common,Fselect,Fitting,Predict,Score,Auto}</p>
<pre><code>                        machine learning models help.
    Common              The common parameters used for other models.
    Fselect             Feature selection from standardized data.
    Fitting             Fitting and predicting the training and testing
                        set from estimators.
    Predict             predict new data from fittting process model.
    Score               Scoring the samples.
    Auto                the auto-processing: standardization, feature
                        selection, Scoring, Fitting and/or Prediction.
</code></pre>        

### **2. optional arguments:**
<pre><code>-h, --help            show this help message and exit
-V, --version         show program's version number and exit
</code></pre>

### **3. Example:**
<p>MLkit.py Auto -h</p>
<pre><code>usage: MLkit.py Auto 
		[-h] [-i INPUT] [-g GROUP] [-o OUTDIR] [-m MODEL]
		[-t POOL] [-sc {GSCV,RSCV}] [-nt N_ITER] [-mr MISSROW]
		[-mc MISSCOL] [-mv {mean,median,most_frequent,constant}]
		[-fv FILLVALUE] [-pp] [-nj N_JOB] [-vm CVMODEL]
		[-cm CVFIT] [-s SCALER [SCALER ...]]
		[-qr QUANTILERANGE [QUANTILERANGE ...]]
		[-pt PCATHRESHOLD] [-sb [SELECTB [SELECTB ...]]]
		[-kb SELECTK] [-rf] [-sf] [-cs]
		[-sm [SPECIFM [SPECIFM ...]]] [-st {parallel,serial}]
		[-sp SELECTCV_REP] [-rr RFE_RATE] [-sr SFS_RATE]
		[-cr CFS_RATE] [-kf K_FEATURES [K_FEATURES ...]] [-rm]
		[-rmf RMFISRT] [-rms RMSECOND] [-rmt RMTHIRD] [-tz TESTS]
		[-cv CROSSV] [-pc] [-lr LRMODE] [-mi MAX_INTERVALS]
		[-bp BASEPOINTS] [-bd BASEODDS] [-pd PDO] [-ms MODSCORE]
		[-p PREDICT] [+P PIPELINE [PIPELINE ...]]
		[+M MODEL [MODEL ...]]

</code></pre>
<pre><code>Examples:
	MLkit.py Auto -i data.traintest.txt -g group.new.txt -p data.predict.txt -o testdt/ -m DT
	MLkit.py Auto -i data.traintest.txt -g group.new.txt -o testdt/ -m DT -pc -s S M
	MLkit.py Common -i data.traintest.txt -g group.new.txt -o testdt/ -m DT
	MLkit.py Fselect -i data.traintest.txt -g group.new.txt -o testdt/ -m DT -s S
	MLkit.py Predict -p data.predict.txt   -g group.new.txt -o testdt/ -m DT.
</code></pre>       

### **4. abbreviation:**
<p>All of the estimators you can use as follows (default: XGB):</p>
<pre><code> classification:.++++++++++++++++++++++
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
Regressioin:....+++++++++++++++++++++
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
</code></pre>

<p>Feature selection with tradional statistic method:</p>
<pre><code>-sb --SelectB:
          (default: ['ANVF', 'MI', 'RS', 'MWU', 'TTI', 'PS'])
classification:......+++++++++++++++++
                **VTh.................VarianceThreshold
                **ANVF................f_classif
                **Chi2................chi2
                **MI..................mutual_info_classif
                **WC..................wilcoxon
                **RS..................ranksums
                **MWU.................mannwhitneyu
                **TTI.................ttest_ind
Regressioin:.........+++++++++++++++++
                **VTh................VarianceThreshold
                **ANVF...............f_regression
                **PS.................pearsonr
                **MI.................mutual_info_classif
</code></pre>

