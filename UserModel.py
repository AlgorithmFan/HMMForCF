#!usr/bin/env python
#coding:utf-8

import types
import time
import datetime
import numpy as np
from GetTime import getMonth, getWeek, getDay

class CUserModel:
    def __init__(self, _user_id):
        self.user_id = _user_id
        self.Items_Time = []
        self.flag = 'm'
        self.dataDict = None
        self.train = None
        self.test = None

    def addItem(self, item_time):
        if type(item_time) is not types.TupleType:
            assert(0)
        self.Items_Time.append(item_time)

    def sortItemsByTime(self):
        '''Sort the items order by time asc'''
        self.Items_Time.sort(key=lambda x:x[1])

    def staticData(self, artistsList, flag):
        '''
        Divide the data according to interval.
        '''
        self.flag = flag
        #artistsDict = {item:artistsList.index(item) for item in artistsList}
        artistsDict = {artistsList[i]: i for i in range(len(artistsList))}
        self.sortItemsByTime()
        artistsLen = len(artistsList)
        dataDict = {}
        dateFuncDict = {'m': getMonth, 'w': getWeek, 'd': getDay}
        dateFunc = dateFuncDict[self.flag]
        for item_id, timestamp in self.Items_Time:
            index = int(dateFunc(timestamp))
            if index not in dataDict:
                dataDict[index] = np.zeros(artistsLen, 'int')
            dataDict[index][artistsDict[item_id]] += 1
        self.dataDict = dataDict

    def splitDate(self, **parameters):
        '''
        Split the data into train dataset and test dataset according to interval.
        '''
        dateFuncDict = {'m': self.splitDataByMonth, 'w':self.splitDataByWeek, 'd':self.splitDataByDay}
        dateFunc = dateFuncDict[self.flag]
        train, test = dateFunc(**parameters)

        if len(test) == 0 or len(train)==0:
            return False
        else:
            self.train = np.array(train)
            self.test = np.array(test)
            index = np.nonzero(self.train.sum(axis=0)>0)
            #self.test[:, index] = np.zeros(self.test[:, index].shape)
            return True

    def splitDataByMonth(self, _year, _month, _day, _dateNum = 12):
        '''
        Spliting the data according to month.
        '''
        train, test = [], []
        year, month, dateNum = _year, _month, _dateNum
        for i in range(dateNum):
            index = datetime.datetime(year=year, month=month, day=1)
            index = int(time.mktime(index.timetuple()))
            if index in self.dataDict:
                train.append(self.dataDict[index])
            month += 1
            if month>12:
                year += 1
                month = 1

        for i in range(1):
            index = datetime.datetime(year=year, month=month, day=1)
            index = int(time.mktime(index.timetuple()))
            if index in self.dataDict:
                test.append(self.dataDict[index])
            month += 1
            if month > 12:
                year += 1
                month -= 12

        return train, test

    def splitDataByWeek(self, _year, _month, _day,  _dateNum = 10):
        '''
        Spliting the data according to week.
        '''
        dayStamp = int(time.mktime(time.strptime('%d-%d-%d' % (_year, _month, _day),'%Y-%m-%d')))
        weekStamp = getWeek(dayStamp)
        train, test = [], []
        dateNum = _dateNum
        for i in range(dateNum):
            if weekStamp in self.dataDict:
                train.append(self.dataDict[weekStamp])
            weekStamp += 604800

        for i in range(1):
            if weekStamp in self.dataDict:
                test.append(self.dataDict[weekStamp])
            weekStamp += 604800

        return train, test


    def splitDataByDay(self, _year, _month, _day, _dateNum=30):
        '''
        Spliting the data according to day.
        '''
        dayStamp = int(time.mktime(time.strptime('%d-%d-%d' % (_year, _month, _day),'%Y-%m-%d')))
        train, test = [], []
        dateNum = _dateNum
        for i in range(dateNum):
            if dayStamp in self.dataDict:
                train.append(self.dataDict[dayStamp])
            dayStamp += 86400

        for i in range(1):
            if dayStamp in self.dataDict:
                test.append(self.dataDict[dayStamp])
            dayStamp += 86400

        return train, test

