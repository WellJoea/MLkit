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

#from .MLNoDaemonProcess import NoDaemonProcess
from .MLPlots import Plots
from .MLEstimators import ML
from .MLPreprocessing import Engineering
from .MLFeaturing import Feature_selection
from .MLSupervising import Modeling
from .MLScoring import CRDScore
from .MLPredicting import Prediction

class Pipeline():
    'The pipeline used for machine learning models'
    def __init__(self, arg, log,  *array, **dicts):
        self.arg = arg
        self.log = log
        self.array  = array
        self.dicts  = dicts

    def Pipe(self):

        if self.arg.commands in ['Common', 'Score', 'Auto']:
            Engineering(self.arg, self.log).Common()

        if self.arg.commands in ['Fselect','Score','Auto']:
            Feature_selection(self.arg, self.log).Fselect()

        if self.arg.commands in ['Fitting','Auto']:
            Modeling(self.arg, self.log).Fitting()

        if self.arg.commands in ['Score']:
            CRDScore(self.arg, self.log).Scoring()

        if ('predict' in  self.arg) and (self.arg.predict):
            if self.arg.commands in ['Score', 'Predict', 'Auto']:
                Prediction(self.arg, self.log).SpvPredicting()
