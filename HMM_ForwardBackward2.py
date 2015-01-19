#!usr/bin/env python
#coding:utf-8

import numpy as np
from numpy import log, exp, newaxis
from HMM import CHMMModel
from scipy.special import gammaln
import os
import scipy.io as sio
import time
from multiprocessing import Process, Pool, Lock
from scipy.special import polygamma
from CommonFunc import  normalizeVec, normalizeMatrix, loadLastData, loadPickle, calRec, dumpPickle
from ReadConfig import CReadConfig
from fileTxt import CRecords

ITERMAX = 100

def logsumexp(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.

       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is the default), logsumexp is
       computed along the last dimension.
    """
    if len(x.shape) < 2:
        xmax = x.max()
        return xmax + log(exp(x-xmax).sum())
    elif dim == -1:
        xmax = x.max()
        if xmax == -np.inf:
            return x[0]
        return xmax + log(exp(x-xmax).sum())
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim+1, len(x.shape)) + [dim])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        mask = (xmax == -np.inf)
        xmax[-mask] += log(exp(x[-mask, :] - xmax[-mask][..., newaxis]).sum(axis=lastdim))
        xmax[mask] = -np.inf
        return xmax

def calLogMultinomial(train, mHMM):
    '''
    Calculating the multinomial matrix.
    '''
    NutArray = train.sum(axis=1)
    #Calculate the log value of multinomial distribution
    # value = log(Nut!) - sum(log(x_i!)) + sum(x_i * log(prob_i))
    fact_part = gammaln(NutArray+1) - gammaln(train+1).sum(axis=1)
    prob_part = np.dot(train, mHMM.Theta.transpose())
    temp = np.repeat([fact_part], repeats=mHMM.HiddenStatesNum, axis=0)
    return temp.transpose() + prob_part

def calLogNegBio(train, mHMM):
    NutArray = train.sum(axis=1)
    temp = gammaln(NutArray[..., newaxis] + mHMM.a[newaxis, ...]) \
           - gammaln(mHMM.a)[newaxis, ...] - gammaln(NutArray+1)[..., newaxis] \
           + NutArray[..., newaxis] * log(mHMM.b)[newaxis, ...]  \
           - (NutArray[..., newaxis] +mHMM.a[newaxis, ...]) * log(mHMM.b+1)[newaxis, ...]
    return temp

def calLogEmssProb(train, mHMM):
    ''''''
    logMultiMatrix = calLogMultinomial(train, mHMM)
    logNegBio = calLogNegBio(train, mHMM)
    logEmssProbs = logNegBio + logMultiMatrix
    return logEmssProbs

def calLogAlphaBeta(mHMM, logEmssProbs):
    ''''''
    Tu = logEmssProbs.shape[0]
    logAlpha = np.zeros((Tu, mHMM.HiddenStatesNum), np.float)
    logBeta = np.zeros((Tu, mHMM.HiddenStatesNum), np.float)
    logNormalVector = np.zeros(Tu) #P(I(u, t) | I(u,1:t-1))

    logAlpha[0, :] = mHMM.InitProbs + logEmssProbs[0, :]
    logNormalVector[0] = logsumexp(logAlpha[0, :])
    logAlpha[0, :] -= logNormalVector[0]

    for t in range(1, Tu):
        for k in range(mHMM.HiddenStatesNum):
            logAlpha[t, k] = logsumexp(logAlpha[t-1, :] + mHMM.TransProbs[:, k]) + logEmssProbs[t, k]
        logNormalVector[t] = logsumexp(logAlpha[t, :])
        logAlpha[t, :] -= logNormalVector[t]

    #beta
    logBeta[-1, :] = 0
    for t in range(Tu-2, -1, -1):
        for k in range(mHMM.HiddenStatesNum):
            logBeta[t, k] = logsumexp(logBeta[t+1, :] + mHMM.TransProbs[k, :] + logEmssProbs[t+1, :])
        logBeta[t, :] -= logNormalVector[t+1]

    return logAlpha, logBeta, logNormalVector

def calSubLogRho(mHMM, logEmssProb, logAlpha, logBeta, logNormalVector):
    Tu = logEmssProb.shape[0]
    logRho = np.zeros((Tu-1, mHMM.HiddenStatesNum, mHMM.HiddenStatesNum))
    temp = logEmssProb + logBeta
    for t in range(Tu-1):
        for i in range(mHMM.HiddenStatesNum):
            logRho[t, i, :] = logAlpha[t, i] + mHMM.TransProbs[i, :] + temp[t+1, :]
        logRho[t, :, :] -= logNormalVector[t+1]
    return logRho

def calLogGammaRho(train, mHMM):
    '''
    Calculating the gamma and rho.
    '''
    logEmssProb = calLogEmssProb(train, mHMM)
    logAlpha, logBeta, logNormalVector = calLogAlphaBeta(mHMM, logEmssProb)
    logGamma = logAlpha + logBeta
    logRho = calSubLogRho(mHMM, logEmssProb, logAlpha, logBeta, logNormalVector)
    return logGamma, logRho

def isZeros(matrix):
    index = np.nonzero(matrix == 0)
    matrix[index] = NearZero
    return matrix

class CForwardBackward:
    def __init__(self):
        self.logGamma = {}   #P(Z(u,t)|I(u,1:T)) for each user. size:|U|*(|Tu|*|hidden_num|)
        self.logRho = {}      #P(Z(u,t-1), Z(u,t) | I(u, 1:T)) for each user. size:|U|*(|Tu|*hidden_num*hidden_num)
        self.coreNum = 4

    def calNegBion(self, mHMM, mUserModels, UserGamma=None):
        '''
        Initialize the parameter a, b of HMM
        '''
        next_psi_numerator = 0.0
        next_nut_numerator = 0.0
        next_denominator = 0.0
        mNutArray = {}
        mUserGamma = {}
        if UserGamma is None:
            for user_id in mUserModels:
                mUserGamma[user_id] = np.ones((mUserModels[user_id].train.shape[0], mHMM.HiddenStatesNum))
        else:
            for user_id in UserGamma:
                mUserGamma[user_id] = exp(UserGamma[user_id])
        for user_id in mUserModels:
            mNutArray[user_id] = mUserModels[user_id].train.sum(axis=1)
            next_psi_numerator += (mUserGamma[user_id] * polygamma(0, mNutArray[user_id])[..., newaxis]).sum(axis=0)
            next_nut_numerator += (mUserGamma[user_id] * mNutArray[user_id][..., newaxis]).sum(axis=0)
            next_denominator += mUserGamma[user_id].sum(axis=0)
        avrNut = next_nut_numerator / next_denominator
        avrPsi = next_psi_numerator / next_denominator

        a0 = 0.5/(log(avrNut) - avrPsi)
        for i in range(100):
            f, df = 0.0, 0.0
            for user_id in mUserModels:
                f += (mUserGamma[user_id] * (polygamma(0, mNutArray[user_id][..., newaxis]+a0[newaxis, ...]) - polygamma(0, a0)[newaxis, ...]
                                             - log(avrNut/a0+1)[newaxis, ...])).sum(axis=0)
                df += (mUserGamma[user_id] * (polygamma(1, mNutArray[user_id][..., newaxis]+a0[newaxis, ...]) - polygamma(1, a0)[newaxis, ...]
                                              - (1.0/(avrNut+a0))[newaxis, ...] + (1.0/a0)[newaxis, ...])).sum(axis=0)
            temp = f/df
            a1 = a0 - temp
            if abs(f).sum() < 0.000001:
                break
            else:
                a0 = a1
        b = next_nut_numerator / (a1*next_denominator)
        return a1, b

    def initTheta(self, mHMM, mUserModels):
        '''
        Initialize the parameter delta of HMM
        '''
        next_theta_numerator = 0.0
        next_theta_denominator = 0.0
        for user_id in mUserModels:
            next_theta_denominator += mUserModels[user_id].train.shape[0]
            next_theta_numerator += mUserModels[user_id].train.sum(axis=0)
        theta = np.array([next_theta_numerator/next_theta_denominator])
        return np.repeat(theta, mHMM.Theta.shape[0], axis=0)

    def initParameters(self, mHMM, mUserModels):
        '''
        Initialize the parameters of HMM
        '''
        #normalize InitProbs
        mHMM.InitProbs += np.random.dirichlet(np.array([alphaInitDir/mHMM.HiddenStatesNum
                                               for i in range(mHMM.HiddenStatesNum)]))
        normalizeVec(mHMM.InitProbs)
        mHMM.InitProbs = log(mHMM.InitProbs)

        #TransitionProbs
        mHMM.TransProbs += np.random.dirichlet([alphaTransDir/mHMM.HiddenStatesNum
                                               for i in range(mHMM.HiddenStatesNum)],
                                              size=mHMM.HiddenStatesNum)
        normalizeMatrix(mHMM.TransProbs)
        mHMM.TransProbs = log(mHMM.TransProbs)

        #Theta
        mHMM.Theta += self.initTheta(mHMM, mUserModels)
        isZeros(mHMM.Theta)
        normalizeMatrix(mHMM.Theta)
        mHMM.Theta = log(mHMM.Theta)

        #a and b
        mHMM.a, mHMM.b = self.calNegBion(mHMM, mUserModels)


    def learn(self, mUserModels, mHMM):
        '''
        Learn the parameters.
        '''
        old_likelihood = -np.inf

        for iteration in range(ITERMAX):
            pool = Pool(processes=self.coreNum)
            result = {}
            for user_id in mUserModels:
                result[user_id] = pool.apply_async(calLogGammaRho, (mUserModels[user_id].train, mHMM, ))
            pool.close()
            pool.join()

            for user_id in result:
                logGamma, logRho = result[user_id].get()
                self.logGamma[user_id] = logGamma.copy()
                self.logRho[user_id] = logRho.copy()

            next_init_numerator = np.zeros(mHMM.InitProbs.shape, np.float)
            next_init_denominator = np.zeros(1, np.float)
            next_trans_numerator = np.zeros(mHMM.TransProbs.shape, np.float)
            next_trans_denominator = np.zeros(mHMM.TransProbs.shape[0], np.float)
            next_theta_numerator = np.zeros(mHMM.Theta.shape, np.float)
            next_theta_denominator = np.zeros(mHMM.Theta.shape[0], np.float)

            for user_id in mUserModels:
                #print 'M ', user_id
                gamma = exp(self.logGamma[user_id])
                rho = exp(self.logRho[user_id])
                next_init_numerator += gamma[0, :]
                next_init_denominator += gamma[0, :].sum()
                next_trans_numerator += rho[:, :, :].sum(axis=0)
                next_trans_denominator += (rho[:, :, :].sum(axis=2)).sum(axis=0)

                NutArray = mUserModels[user_id].train.sum(axis=1)
                next_theta_numerator += np.dot(gamma.transpose(), mUserModels[user_id].train)
                temp = np.dot(NutArray, gamma)
                next_theta_denominator += temp


            mHMM.InitProbs = (next_init_numerator + float(alphaInitDir)/mHMM.HiddenStatesNum - 1)\
                             /(next_init_denominator + alphaInitDir - mHMM.HiddenStatesNum)
            mHMM.InitProbs = log(mHMM.InitProbs)

            next_trans_denominator = np.repeat(np.array([next_trans_denominator]), mHMM.TransProbs.shape[1], axis=0)
            next_trans_denominator = next_trans_denominator.transpose()
            mHMM.TransProbs = (next_trans_numerator + float(alphaTransDir)/mHMM.HiddenStatesNum - 1)\
                            /(next_trans_denominator + alphaTransDir - mHMM.HiddenStatesNum)
            mHMM.TransProbs = log(mHMM.TransProbs)

            next_theta_denominator = np.repeat(np.array([next_theta_denominator]), mHMM.Theta.shape[1], axis=0)
            next_theta_denominator = next_theta_denominator.transpose()
            mHMM.Theta = (next_theta_numerator + float(alphaThetaDir)/mHMM.ObservationStatesNum - 1)\
                            /(next_theta_denominator + alphaThetaDir - mHMM.ObservationStatesNum)
            mHMM.Theta = log(mHMM.Theta)

            mHMM.a, mHMM.b = self.calNegBion(mHMM, mUserModels, self.logGamma)

            #calculate the expected likelihood.
            new_likelihood = 0.0
            for user_id in mUserModels:
                Tu = mUserModels[user_id].train.shape[0]
                gamma = exp(self.logGamma[user_id])
                rho = exp(self.logRho[user_id])

                init = (gamma[0, :]*mHMM.InitProbs).sum()

                #Trans_part
                trans = (rho*np.repeat([mHMM.TransProbs], repeats=Tu-1, axis=0)).sum()

                #Theta_part
                mult_pmf = calLogMultinomial(mUserModels[user_id].train, mHMM)
                emss = (gamma*mult_pmf).sum()

                #Delta_part
                negative_pmf = calLogNegBio(mUserModels[user_id].train, mHMM)
                negative = (gamma*negative_pmf).sum()

                new_likelihood += init + trans + emss + negative

            print 'Iteration %d: %f' % (iteration, new_likelihood)
            if new_likelihood - old_likelihood <0.1:
                self.logGamma = old_gamma.copy()
                mHMM = old_HMM.copy()
                dumpPickle(mHMM, 'HMM/mHMM.txt')
                dumpPickle(self.logGamma, 'HMM/gamma.txt')
                mHMM.savemat('HMM', 0)
                break
            else:
                old_likelihood = new_likelihood
                old_gamma = self.logGamma.copy()
                old_HMM = mHMM.copy()

    def calSubRecommend(self, mUserModels, active_id, mHMM):
        '''
        Calculating the recommendation for the active_id.
        '''
        recommendation = np.zeros(mHMM.ObservationStatesNum, np.float)
        for k in range(mHMM.HiddenStatesNum):
            logRec = logsumexp(self.logGamma[active_id][-1, :] + mHMM.TransProbs[:, k]) - mHMM.a[k]*log(1+mHMM.b[k]*exp(mHMM.Theta[k,:]))
            recommendation += exp(logRec)
        recommendation = 1-recommendation
        recDict = {}
        for i in range(mHMM.ObservationStatesNum):
            if mUserModels[active_id].train[:, i].sum()>0:
                continue
            if recommendation[i] >= 0:
                recDict[i] = recommendation[i]
        return recDict

    def calRecommend(self, mUserModels, hidden_num, mArtistsList, top_num):
        '''
        Calculating the recommendation for each user.
        '''
        mHMM = CHMMModel(hidden_num, len(mArtistsList))
        self.initParameters(mHMM, mUserModels)
        self.learn(mUserModels, mHMM)

        recommendation = {}
        for user_id in mUserModels:
            temp = self.calSubRecommend(mUserModels, user_id, mHMM)
            temp = sorted(temp.iteritems(), key=lambda x:x[1], reverse=True)
            recommendation[user_id] = [item_id for item_id, prob in temp[:top_num]]
        return recommendation, mHMM

    def clearVar(self):
        self.gamma = {}
        self.rho = {}

def main(mUserModels, mArtistsList, parameters):
    '''
    Test the data by week.
    '''
    hitNum10, recNum10, preNum10 = 0, 0, 0
    hitNum5, recNum5, preNum5 = 0, 0, 0

    mCRecords10 = CRecords(r'recommendation/HMMForCFIm2%s_t%d_a%d.txt' % (parameters['date_flag'], 10, parameters['items_thr']))
    mCRecords5 = CRecords(r'recommendation/HMMForCFIm2%s_t%d_a%d.txt' % (parameters['date_flag'], 5, parameters['items_thr']))
    mCRecords10.writeDescription(parameters['description'])
    mCRecords5.writeDescription(parameters['description'])
    mCRecords10.writeParameters(parameters)
    mCRecords5.writeParameters(parameters)

    copy_UserModels = mUserModels.copy()
    mCForwardBackward = CForwardBackward()

    dayStamp = 1104508800.0
    date_num = parameters['date_num']
    dayStr = time.localtime(dayStamp)
    year, month, day = dayStr.tm_year, dayStr.tm_mon, dayStr.tm_mday
    while date_num < parameters['date_thr']:
        print '*'*100
        print 'DateNum: %d.' % date_num

        mUserModels = copy_UserModels.copy()
        for user_id in mUserModels.keys():
            flag = mUserModels[user_id].splitDate(_year=year, _month=month, _day=day, _dateNum = date_num)
            if flag == False:  del mUserModels[user_id]
        if len(mUserModels) > 1:
            recommendation, mHMM = mCForwardBackward.calRecommend(mUserModels, parameters['hidden_num'], mArtistsList,  parameters['top_num'])
            mCForwardBackward.clearVar()
        else:
            recommendation = {}

        h, r, p = calRec(recommendation, mUserModels, 10)
        recNum10 += r
        preNum10 += p
        hitNum10 += h
        if h > 0.0:
            print 'Hit: %d, Recall: %d, Precision: %d, UsersNum: %d.' % (h, r, p, len(mUserModels.keys()))
            print 'Recall: %f, Precision: %f.' % (float(h)/r, float(h)/p)
            mCRecords10.writeHRP(date_num, h, r, p)
        else:
            print 'Hit: %d, Recall: %d, Precision: %d, UsersNum: %d.' % (h, r, p, len(mUserModels.keys()))
            #fp.write('Hit: %d, Recall: %d, Precision: %d, UsersNum: %d.\n' % (h, r, p, len(mUserModels.keys())))

        h, r, p = calRec(recommendation, mUserModels, 5)
        recNum5 += r
        preNum5 += p
        hitNum5 += h
        if h > 0.0:
            mCRecords5.writeHRP(date_num, h, r, p)

        date_num += parameters['date_interval']

    print '*'*100
    print 'artists_thr: %d.' % parameters['items_thr']
    recall = float(hitNum10)/recNum10
    precision = float(hitNum10)/preNum10
    f1 = 2*recall*precision/(recall+precision)
    print 'HitNum: ', hitNum10
    print 'Recall: %d, %f.' % (recNum10, recall)
    print 'Precision: %d, %f.' % (preNum10, precision)
    print 'F1: ', f1
    mCRecords10.writeDescription('artists_thr: %d.\n' % parameters['items_thr'])
    mCRecords10.writeDescription('Recall: %d, %f.\n' % (recNum10, recall))
    mCRecords10.writeDescription('Precision: %d, %f.\n' % (preNum10, precision))
    mCRecords10.writeDescription('F1: %f\n' % f1)
    mCRecords10.close()

    recall = float(hitNum5)/recNum5
    precision = float(hitNum5)/preNum5
    f1 = 2*recall*precision/(recall+precision)
    print 'HitNum: ', hitNum5
    print 'Recall: %d, %f.' % (recNum5, recall)
    print 'Precision: %d, %f.' % (preNum5, precision)
    print 'F1: ', f1
    mCRecords5.writeDescription('artists_thr: %d.\n' % parameters['items_thr'])
    mCRecords5.writeDescription('Recall: %d, %f.\n' % (recNum5, recall))
    mCRecords5.writeDescription('Precision: %d, %f.\n' % (preNum5, precision))
    mCRecords5.writeDescription('F1: %f\n' % f1)
    mCRecords5.close()

if __name__ == '__main__':

    NearZero = 10**(-290)
    mCConfig = CReadConfig("config.ini")
    parameters = mCConfig.getHMM()

    mUserModels, mArtistsList = loadLastData(parameters['items_thr'], parameters['date_flag'])
    #
    # mUserModels = loadPickle('Data/mUserModelsIm%d_%s.txt' % (parameters['items_thr'], parameters['date_flag']))
    # mArtistsList = loadPickle('Data/mArtistsListIm%d_%s.txt' % (parameters['items_thr'], parameters['date_flag']))


    hiddenStates = [20, 30, 40, 50]
    for i in range(len(hiddenStates)):
        hidden_num = hiddenStates[i]
        parameters['hidden_num'] = hidden_num
        alphaInitDir = (1+parameters['init']/hidden_num) * hidden_num
        alphaTransDir = (1+parameters['trans']/(hidden_num*hidden_num)) * hidden_num
        alphaThetaDir = (1+parameters['theta']/(hidden_num*len(mArtistsList))) * len(mArtistsList)
        main(mUserModels, mArtistsList, parameters)
