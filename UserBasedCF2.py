#!usr/bin/env python
#coding:utf-8

import numpy as np
import time
from CommonFunc import loadLastData, calRec
from ReadConfig import CReadConfig
from fileTxt import CRecords

class CUserBasedCF:
    def __init__(self):
        pass

    def calSimilarity(self, user_A, user_B):
        '''
        Calculate the similarity between user A and user B.
        '''
        user_A_avr = float(np.sum(user_A))/len(user_A)
        user_B_avr = float(np.sum(user_B))/len(user_B)
        temp_A = user_A-user_A_avr
        temp_B = user_B-user_B_avr
        return float(np.sum(temp_A*temp_B))/np.linalg.norm(temp_A)/np.linalg.norm(temp_B)*temp_B

    def calSubRecommend(self, users_map_items, artistsList, active_user_id):
        '''
        Calculate the recommendation for the active user.
        '''
        recommendation = np.zeros(len(artistsList), 'float')
        for user_id in users_map_items:
            if user_id == active_user_id:
                continue
            recommendation += self.calSimilarity(users_map_items[active_user_id], users_map_items[user_id])

        temp = {}
        for index in range(len(artistsList)):
            if users_map_items[active_user_id][index] > 0: continue
            if recommendation[index]>0:
                temp[index] = recommendation[index]
        return temp

    def preProcess(self, mUserModels, artistsList):
        '''
        Get the average rate of items.
        '''
        artists = np.zeros(len(artistsList), 'float')
        users_map_items = {}
        for user_id in mUserModels:
            temp = np.zeros(len(artistsList), 'int')
            for artist_id in range(artists.shape[0]):
                if mUserModels[user_id].train[:, artist_id].sum()>0:
                    temp[artist_id] = 1.0
            users_map_items[user_id] = temp
            artists += temp
        return users_map_items, artists/len(mUserModels)

    def calRecommend(self, mUserModels, artistsList, top_num):
        '''
        Calculate the recommendation for all users
        '''
        users_map_items, items_avrRate = self.preProcess(mUserModels, artistsList)
        recommendation = {}
        for user_id in mUserModels:
            temp = self.calSubRecommend(users_map_items, artistsList, user_id)
            temp = sorted(temp.iteritems(), key=lambda x:x[1], reverse=True)
            recommendation[user_id] = [item_id for item_id, sim in temp[:top_num]]
        return recommendation

def main(mUserModels, mArtistsList, parameters):
    hitNum10, recNum10, preNum10 = 0, 0, 0
    hitNum5, recNum5, preNum5 = 0, 0, 0
    mCRecords10 = CRecords('recommendation/UserBasedCF2%s_t%d_a%d.txt' % (parameters['date_flag'], 10, parameters['items_thr']))
    mCRecords5 = CRecords('recommendation/UserBasedCF2%s_t%d_a%d.txt' % (parameters['date_flag'], 5, parameters['items_thr']))
    mCRecords10.writeDescription(parameters['description'])
    mCRecords5.writeDescription(parameters['description'])

    copy_UserModels = mUserModels.copy()
    mCUserBasedCF = CUserBasedCF()

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

        if len(mUserModels) != 0:
            recommendation = mCUserBasedCF.calRecommend(mUserModels, mArtistsList, parameters['top_num'])
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
    mCRecords10.writeDescription.write('Precision: %d, %f.\n' % (preNum10, precision))
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
    mCRecords5.writeDescription.write('Precision: %d, %f.\n' % (preNum5, precision))
    mCRecords5.writeDescription('F1: %f\n' % f1)
    mCRecords5.close()


if __name__ == '__main__':
    from CommonFunc import loadPickle
    mCConfig = CReadConfig("config.ini")
    parameters = mCConfig.getUBCF()

    # mUserModels, mArtistsList = loadLastData(artists_thr, flag)
    #
    mUserModels = loadPickle('Data/mUserModelsIm%d_%s.txt' % (parameters['items_thr'], parameters['date_flag']))
    mArtistsList = loadPickle('Data/mArtistsListIm%d_%s.txt' % (parameters['items_thr'], parameters['date_flag']))
    #

    main(mUserModels, mArtistsList, parameters)

