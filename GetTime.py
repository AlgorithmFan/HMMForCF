#!usr/bin/env python
#coding:utf-8

import datetime
import time

def getMonth(timestamp):
    '''
    Get the month of timestamp, and transform it to timestamp
    '''
    temp = datetime.datetime.fromtimestamp(timestamp)
    temp = datetime.datetime(year=temp.year, month=temp.month, day=1)
    monthStamp = time.mktime(temp.timetuple())
    return monthStamp

def getWeek(timestamp):
    '''
    Get the week of timestamp, and transform it to timestamp
    '''
    weekth = int(time.strftime('%w', time.localtime(timestamp)))
    MondayStamp = timestamp - (weekth-1)*86400
    MondayStr = time.localtime(MondayStamp)
    return time.mktime(time.strptime(time.strftime('%Y-%m-%d', MondayStr), '%Y-%m-%d'))

def getDay(timestamp):
    '''
    Get the day of timestamp, and transform it to timestamp
    '''
    temp = datetime.datetime.fromtimestamp(timestamp)
    temp = datetime.datetime(year=temp.year, month=temp.month, day=temp.day)
    dayStamp = time.mktime(temp.timetuple())
    return dayStamp
