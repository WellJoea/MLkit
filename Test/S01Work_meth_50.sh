IN=/lustre/rde/user/zhouw/00software/MLkit_dev/Test/00DataNew
OP=/lustre/rde/user/zhouw/00software/MLkit_dev/Test/02ResultAll

Type=$1
Sc=$2
OU=$OP
GR=$IN/group.txt

IP=$IN/TT.data.txt
PR=$IN/PR.data.txt

Python=/lustre/rde/user/zhouw/00software/anaconda3/bin/python
MLkit=/lustre/rde/user/zhouw/00software/anaconda3/bin/MLkit.py

#$Python $MLkit Auto -i $IP -g $GR  -p $PR -o $OU -m $Type -s $Sc -se RFECV SFSCV -sm LinearSVM XGB -vm LOU -sp 1 #
$Python $MLkit Auto -i $IP -g $GR  -p $PR -o $OU -m $Type -s $Sc -se RFECV  -vm LOU -sp 5 -sc -rt 10 -cm LOU #
#$Python $MLkit Common -i $IP -g $GR  -o $OU -m $Type -s $Sc #-p $PR
#$Python $MLkit Fselect -i $IP -g $GR  -o $OU -m $Type -s $Sc -se RFECV SFSCV -sm LinearSVM XGB #-p $PR
#$Python $MLkit Fitting -i $IP -g $GR  -o $OU -m $Type -s $Sc 
#$Python $MLkit Predict -i $IP -g $GR  -o $OU -m $Type -s $Sc -p $PR #-rs 'all'
