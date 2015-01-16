#!usr/bin/env python
#coding:utf-8

import numpy as np
import time
from CommonFunc import loadLastData, calRec

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
            # if users_map_items[active_user_id][index] > 0: continue
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


def main(mUserModels, mArtistsList, artists_thr, top_num, dateNum, dateFlag):
    '''
    Test the data by week.
    '''
    hitNum10, recNum10, preNum10 = 0, 0, 0
    hitNum5, recNum5, preNum5 = 0, 0, 0

    copy_UserModels = mUserModels.copy()
    mCUserBasedCF = CUserBasedCF()

    dayStamp = 1104508800.0
    thr = 1293724800.0
    while dayStamp < thr:
        dayStr = time.localtime(dayStamp)
        year, month, day = dayStr.tm_year, dayStr.tm_mon, dayStr.tm_mday

        print '*'*100
        print 'Date range from %d-%d-%d: dateNum: %d.' % (year, month, day, dateNum)
        mUserModels = copy_UserModels.copy()
        for user_id in mUserModels.keys():
            flag = mUserModels[user_id].splitDate(_year=year, _month=month, _day=day, _dateNum = dateNum)
            if flag == False:  del mUserModels[user_id]
        if len(mUserModels) != 0:
            recommendation = mCUserBasedCF.calRecommend(mUserModels, mArtistsList, top_num)
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
            #fp.write('Hit: %d, Recall: %d, Precision: %d, UsersNum: %d.\n' % (h, r, p, len(mUserModels.keys())))

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
    artists_thr = 100

    top_num = 10
    flag = 'd'

    if flag == 'm':
        dateNum = 30
    elif flag == 'w':
        dateNum = 130
    elif flag == 'd':
        dateNum = 200
    else:
        exit(0)

    mUserModels, mArtistsList = loadLastData(artists_thr, flag)
    main(mUserModels, mArtistsList, artists_thr, top_num, dateNum, flag)

