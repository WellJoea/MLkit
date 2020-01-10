import pandas as pd
import numpy as np
import re
import os

class Openf():
    def __init__(self, file, *array,compression='infer',comment='#', index_label=None, sep="\t",index = True, header=0,index_col=None):
        self.file  = file.strip()
        self.array = array
        self.compr = compression
        self.comme = comment
        self.sep   = sep
        self.header= header
        self.index_col= index_col
        self.index = index 
        self.index_label=index_label

    def openb(self):
        if re.search(r'.*xlsx$|.*xlss$', self.file, re.I):
            outp = pd.read_excel(self.file, header=self.header, index_col=self.index_col, encoding='utf-8', sheet_name=0, comment=self.comme, compression=self.compr).fillna(np.nan)
        else:
            outp = pd.read_csv(self.file, header=self.header, index_col=self.index_col, encoding='utf-8', sep=self.sep, comment=self.comme,compression=self.compr).fillna(np.nan)
        return(pd.DataFrame(outp))

    def openw(self):
        os.makedirs( os.path.dirname(self.file), exist_ok=True )
        f= open(self.file, 'w',encoding='utf-8')
        f.write(self.array[0])
        f.close()
    def openv(self):
        os.makedirs( os.path.dirname(self.file), exist_ok=True )
        df = self.array[0]
        df.to_csv(self.file, sep=self.sep, index=self.index, index_label=self.index_label, header=True, line_terminator='\n')

class OpenM():
    def __init__(self, arg, log):
        self.arg = arg
        self.log = log

    def openi(self):
        dfall = Openf(self.arg.input, index_col=0).openb()
        return (dfall)
    def openg(self):
        group = Openf(self.arg.group).openb()
        group.columns = ['Variables', 'Group', 'Type']

        RYa = group[(group.Group == 'Y') & (group.Type =='R') ].Variables.tolist()
        CYa = group[(group.Group == 'Y') & (group.Type =='C') ].Variables.tolist()
        Xa  = group[(group.Group != 'Y')].Variables.tolist()
        Xg  = group[(group.Group != 'Y')][['Variables','Group']]
        Xg.set_index('Variables', inplace=True)

        return (group, RYa, CYa, Xa, Xg)

    def openp(self):
        dfpred = Openf(self.arg.predict, index_col=0).openb()
        return (dfpred)
