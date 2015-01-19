#!usr/bin/env python
#coding:utf-8
import numpy as np
from numpy import newaxis
from UserModel import CUserModel
import codecs
import cPickle

def normalizeVec(vector):
    '''Normalize a vector'''
    v_sum = vector.sum()
    assert(v_sum != 0)
    vector /= v_sum

def normalizeMatrix(matrix):
    '''Normalize each row of this matrix '''
    row_sum = matrix.sum(axis=1)
    matrix /= row_sum[..., newaxis]

def calRec(recommendation, mUserModels, top_num):
    hitNum, recall, precision = 0, 0, 0
    for user_id in recommendation:
        index = np.nonzero(mUserModels[user_id].test>0)
        recall += len(index[0])
        precision += len(recommendation[user_id][:top_num])
        for item_index in recommendation[user_id][:top_num]:
            if sum(mUserModels[user_id].test[:, item_index])>0:
                hitNum += 1
    return hitNum, recall, precision

def loadArtists(filename, thr_num):
    fp = codecs.open(filename, 'rb', 'utf-8')
    artists = set()
    while True:
        line = fp.readline()
        if line == '': break
        temp = line.split('\t')
        if int(temp[2])<=thr_num:
            continue
        artists.add(int(temp[1]))
    return artists

def loadUserActions(filename, artists):
    fp = codecs.open(filename, 'rb', 'utf-8')
    mUserModels = {}
    while True:
        line = fp.readline()
        if line == '': break
        temp = line.split()
        user_id, artist_id, timestamp = int(temp[0]), int(temp[1]), int(temp[2])
        if user_id not in mUserModels:
            mUserModels[user_id] = CUserModel(user_id)
        if artist_id not in artists:
            continue
        mUserModels[user_id].addItem((artist_id, timestamp))
    for user_id in mUserModels.keys():
        if len(mUserModels[user_id].Items_Time)<=0:
            del mUserModels[user_id]
    return mUserModels

def loadLastData(artists_thr, flag):
    '''
    flag = 'd', 'w' or 'm'
    '''
    print '*'*100
    print 'Read Artists.txt.'
    artists = loadArtists('artists.txt', artists_thr)
    print 'The length of artists is %d.' % len(artists)

    print '*'*100
    print 'Read users_artists_timestamp.txt.'
    mUserModels = loadUserActions('users_artists_timestamp.txt', artists)

    print '*'*100
    artistsList = list(artists)
    print 'Divide the data into train and test.'
    for user_id in mUserModels.keys():
        print 'user %d: %d' % (user_id, len(mUserModels[user_id].Items_Time))

        mUserModels[user_id].staticData(artistsList, flag)
        if len(mUserModels[user_id].dataDict)<=1:
            del mUserModels[user_id]

    print 'The number of users is %d.' % len(mUserModels)
    return mUserModels, artistsList

def loadPickle(filename):
    print 'Read data from %s.' % filename
    fp = open(filename)
    data = cPickle.load(fp)
    fp.close()
    return data

def dumpPickle(data, filename):
    fp = open(filename, 'w')
    cPickle.dump(data, fp)
    fp.close()

if __name__ == '__main__':
    test = np.random.randint(0, 10, (5,4))
    test = np.array(test, np.float)
    normalizeMatrix(test)
    print test
    print test.sum(axis=1)