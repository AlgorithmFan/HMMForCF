#!usr/bin/env python
#coding:utf-8

import os
import time
import numpy as np
from numpy import log, exp, newaxis
from scipy.special import gammaln
from multiprocessing import Process, Pool
from scipy.special import polygamma
from HMM import CHMMModel
from UserModel import CUserModel
from CommonFunc import normalizeMatrix, normalizeVec, loadLastData, calRec

ITERMAX = 100
alphaInitDir = 100.0
alphaTransDir = 100.0
alphaThetaDir = 2000.0
NearZero = 10**(-290)

def logsumexp(x, dim=-1):
    '''
    Compute log(sum(exp(x))) in a numerically stable way.
    '''
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

def calLogMultinomial(data, mHMM):
    '''
    Calculating the multinomial matrix.
    '''
    NutArray = data.sum(axis=1)
    #Calculate the log value of multinomial distribution
    # value = log(Nut!) - sum(log(x_i!)) + sum(x_i * log(prob_i))
    first_part = gammaln(NutArray+1)
    second_part = gammaln(data+1).sum(axis=1)
    third_part = np.dot(data, mHMM.Theta.transpose())
    return (first_part - second_part)[..., newaxis] + third_part

def calLogNegBio(data, mHMM):
    '''
    Calculating the negative binomial matrix.
    '''
    NutArray = data.sum(axis=1)
    temp = gammaln(NutArray[..., newaxis] + mHMM.a[newaxis, ...]) \
           - gammaln(mHMM.a)[newaxis, ...] - gammaln(NutArray+1)[..., newaxis] \
           + NutArray[..., newaxis] * log(mHMM.b)[newaxis, ...]  \
           - (NutArray[..., newaxis] +mHMM.a[newaxis, ...]) * log(mHMM.b+1)[newaxis, ...]
    return temp


def calLogEmssProb(data, mHMM):
    '''
    Calculating the emssion matrix.
    '''
    logMultiMatrix = calLogMultinomial(data, mHMM)
    logNegBio = calLogNegBio(data, mHMM)
    logEmssProbs = logNegBio + logMultiMatrix
    return logEmssProbs

def calLogAlphaBeta(mHMM, logEmssProbs):
    '''
    Calculating the alpha and beta. #Formula 1.2 - Formula 1.4
    '''
    Tu = logEmssProbs.shape[0]
    logAlpha = np.zeros((Tu, mHMM.HiddenStatesNum), np.float)
    logBeta = np.zeros((Tu, mHMM.HiddenStatesNum), np.float)
    logNormalVector = np.zeros(Tu) #P(I(u, t) | I(u,1:t-1))

    #Formula 1.2
    logAlpha[0, :] = mHMM.InitProbs + logEmssProbs[0, :]
    logNormalVector[0] = logsumexp(logAlpha[0, :])
    logAlpha[0, :] -= logNormalVector[0]

    #Formula 1.3
    for t in range(1, Tu):
        for k in range(mHMM.HiddenStatesNum):
            logAlpha[t, k] = logsumexp(logAlpha[t-1, :] + mHMM.TransProbs[:, k]) + logEmssProbs[t, k]
        logNormalVector[t] = logsumexp(logAlpha[t, :])
        logAlpha[t, :] -= logNormalVector[t]

    #beta
    logBeta[-1, :] = 0
     #Formula 1.4
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
    #Formula 1.1
    logEmssProb = calLogEmssProb(train, mHMM)
    #Formula 1.2 - 1.4
    logAlpha, logBeta, logNormalVector = calLogAlphaBeta(mHMM, logEmssProb)
    logGamma = logAlpha + logBeta    #Formula 1.5
    logRho = calSubLogRho(mHMM, logEmssProb, logAlpha, logBeta, logNormalVector)     #Formula 1.6
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
        Calculate the parameter a, b of HMM
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

            a1 = a0 - f/df
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
        Initialize the parameters of HMM. #Step 2.
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
        # mHMM.Theta += np.random.dirichlet([alphaThetaDir/mHMM.ObservationStatesNum
        #                                        for i in range(mHMM.ObservationStatesNum)],
        #                                       size=mHMM.HiddenStatesNum)
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

        #print 'E Step.'
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

            #print 'M Step.'
            next_init_numerator = np.zeros(mHMM.InitProbs.shape, np.float)
            next_init_denominator = np.zeros(1, np.float)
            next_trans_numerator = np.zeros(mHMM.TransProbs.shape, np.float)
            next_trans_denominator = np.zeros(mHMM.TransProbs.shape[0], np.float)
            next_theta_numerator = np.zeros(mHMM.Theta.shape, np.float)
            next_theta_denominator = np.zeros(mHMM.Theta.shape[0], np.float)
            for user_id in mUserModels:
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

            #Formula 1.7
            mHMM.InitProbs = (next_init_numerator + float(alphaInitDir)/mHMM.HiddenStatesNum - 1)\
                             /(next_init_denominator + alphaInitDir - mHMM.HiddenStatesNum)
            mHMM.InitProbs = log(mHMM.InitProbs)

            #Formula 1.8
            mHMM.TransProbs = (next_trans_numerator + float(alphaTransDir)/mHMM.HiddenStatesNum - 1)\
                            /(next_trans_denominator[..., newaxis] + alphaTransDir - mHMM.HiddenStatesNum)
            mHMM.TransProbs = log(mHMM.TransProbs)

            #Formula 1.9
            mHMM.Theta = (next_theta_numerator + float(alphaThetaDir)/mHMM.ObservationStatesNum - 1)\
                            /(next_theta_denominator[..., newaxis] + alphaThetaDir - mHMM.ObservationStatesNum)
            mHMM.Theta = log(mHMM.Theta)

            #Formula 1.10 and #Formula 1.11
            mHMM.a, mHMM.b = self.calNegBion(mHMM, mUserModels, self.logGamma)

            #calculate the expected likelihood.
            #print 'Calculate the expected likelihood.'

            #Formula 1.12
            new_likelihood = 0.0
            for user_id in mUserModels:
                Tu = mUserModels[user_id].train.shape[0]
                gamma = exp(self.logGamma[user_id])
                rho = exp(self.logRho[user_id])

                init = (gamma[0, :]*mHMM.InitProbs).sum()

                #Trans_part
                trans = (rho*mHMM.TransProbs[newaxis, ...]).sum()

                #Theta_part
                mult_pmf = calLogMultinomial(mUserModels[user_id].train, mHMM)
                emss = (gamma*mult_pmf).sum()

                #negative_part
                negative_pmf = calLogNegBio(mUserModels[user_id].train, mHMM)
                negative = (gamma*negative_pmf).sum()

                new_likelihood += init + trans + emss + negative

            print 'Iteration %d: %f' % (iteration, new_likelihood)
            if new_likelihood - old_likelihood <0.0001:
                self.logGamma = old_gamma.copy()
                mHMM = old_HMM.copy()
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
        #Formula 1.13 and Formula 1.14
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

def main(mUserModels, mArtistsList, artists_thr, top_num, hidden_num, dateNum, dateFlag):
    '''
    Test the data by week.
    '''
    hitNum10, recNum10, preNum10 = 0, 0, 0
    hitNum5, recNum5, preNum5 = 0, 0, 0

    copy_UserModels = mUserModels.copy()
    mCForwardBackward = CForwardBackward()

    dayStamp = 1104508800.0                 #2015-1-1
    thr = 1293724800.0                      #2010-12-31
    while dayStamp < thr:
        dayStr = time.localtime(dayStamp)
        year, month, day = dayStr.tm_year, dayStr.tm_mon, dayStr.tm_mday

        print '*'*100
        print 'Date range from %d-%d-%d: dateNum: %d.' % (year, month, day, dateNum)
        mUserModels = copy_UserModels.copy()
        for user_id in mUserModels.keys():
            flag = mUserModels[user_id].splitDate(_year=year, _month=month, _day=day, _dateNum = dateNum)
            if flag == False:  del mUserModels[user_id]
        if len(mUserModels) > 1:
            recommendation, mHMM = mCForwardBackward.calRecommend(mUserModels, hidden_num, mArtistsList, top_num)
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
        else:
            print 'Hit: %d, Recall: %d, Precision: %d, UsersNum: %d.' % (h, r, p, len(mUserModels.keys()))

        h, r, p = calRec(recommendation, mUserModels, 5)
        recNum5 += r
        preNum5 += p
        hitNum5 += h

        if dateFlag == 'm':
            month += 1
            if month > 12:
                year += 1
                month = 1
            dayStamp = time.mktime(time.strptime('%d-%d-%d' % (year, month, day),'%Y-%m-%d'))
        elif dateFlag == 'w':
            dayStamp += 604800*4
        elif dateFlag == 'd':
            dayStamp += 86400*30

    print '*'*100
    print 'artists_thr: %d.' % artists_thr
    recall = float(hitNum10)/recNum10
    precision = float(hitNum10)/preNum10
    print 'HitNum: ', hitNum10
    print 'Recall: %d, %f.' % (recNum10, recall)
    print 'Precision: %d, %f.' % (preNum10, precision)
    print 'F1: ', 2*recall*precision/(recall+precision)

    recall = float(hitNum5)/recNum5
    precision = float(hitNum5)/preNum5
    print 'HitNum: ', hitNum5
    print 'Recall: %d, %f.' % (recNum5, recall)
    print 'Precision: %d, %f.' % (preNum5, precision)
    print 'F1: ', 2*recall*precision/(recall+precision)

if __name__ == '__main__':
    artists_thr = 100           # Artists listened to by >100 users included
    top_num = 10                # The number of recommendation
    hidden_num = 30             # The number of hidden states
    flag = 'm'                  # The length of the time peroid was set to 1 month.
    if flag == 'm':
        dateNum = 30            # 30 months for training, and 1 month for testing.
    elif flag == 'w':
        dateNum = 130           # 130 weeks for training, and 1 week for testing.
    elif flag == 'd':
        dateNum = 500           # 500 days for training, and 1 day for testing.
    else:
        exit(0)
    mUserModels, mArtistsList = loadLastData(artists_thr, flag) #Load data
    main(mUserModels, mArtistsList, artists_thr, top_num, hidden_num, dateNum, flag)    # Training and prediction
