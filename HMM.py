#!usr/bin/env python
#coding:utf-8

import numpy as np
import scipy.io as sio

class CHMMModel:
    '''
    A Class for improved HMM
    '''
    def __init__(self, HiddenStatesNum, ObservationStatesNum):
        self.HiddenStatesNum = HiddenStatesNum      # The number of hidden states
        self.ObservationStatesNum = ObservationStatesNum    # The number of observation states

        #Pi: size: N; The probability of initial vector.
        self.InitProbs = np.zeros(self.HiddenStatesNum, np.float)

        #A: size: N*N; The probability of transition matrix.
        # a[i][j] represents the probability from state i to state j
        self.TransProbs = np.zeros((self.HiddenStatesNum, self.HiddenStatesNum), np.float)

        #Theta: size: K*N; The probability of emission matrix.
        #Theta[k][i] represents the probability of observed state i at hidden state k.
        self.Theta = np.zeros((self.HiddenStatesNum, self.ObservationStatesNum), np.float)

        #a, b: size: K;a,b represent the parameters of negative binomial distribution.
        self.a = np.zeros(self.HiddenStatesNum, np.float)
        self.b = np.zeros(self.HiddenStatesNum, np.float)

    def copy(self):
        copy_HMM = CHMMModel(self.HiddenStatesNum, self.ObservationStatesNum)
        copy_HMM.InitProbs = self.InitProbs
        copy_HMM.TransProbs = self.TransProbs
        copy_HMM.Theta = self.Theta
        copy_HMM.a = self.a
        copy_HMM.b = self.b
        return copy_HMM

