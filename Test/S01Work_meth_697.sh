IN=/lustre/rde/user/zhouw/03ML/06DAIJIGANGnew/00DataNew
OP=/lustre/rde/user/zhouw/03ML/06DAIJIGANGnew/02ResultAll

Type=$1
Sc=$2
OU=$OP/Classify_MethL_Region_697
GR=$IN/DJG_meth_group.697.txt

IP=$IN/DJG_meth_snv_pro_clin_98.txt

Python=/lustre/rde/user/zhouw/00software/anaconda3/bin/python
MLkit=/lustre/rde/user/zhouw/00software/anaconda3/bin/MLkit.py
$Python $MLkit Auto -i $IP -g $GR -o $OU -m $Type -s $Sc -kr 0.5 -rr 30
