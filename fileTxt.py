#!usr/bin/env python
#coding:utf-8
import time

class CRecords():
    def __init__(self, filename):
        self.fp = open(filename, 'a')
        self.begin()

    def begin(self):
        self.fp.write('#'*100+'\n')
        dayStr = time.localtime()
        self.fp.write('%d-%d-%d %d:%d:%d\n' %
                (dayStr.tm_year, dayStr.tm_mon, dayStr.tm_mday,
                 dayStr.tm_hour, dayStr.tm_min, dayStr.tm_sec))
        self.fp.flush()

    def writeDescription(self, description=None):
        self.fp.write('*'*100 + '\n')
        if description is not None:
            self.fp.write(description)
            self.fp.write('\n')
        self.fp.flush()

    def writeParameters(self, parameters):
        for key in parameters:
            if key == 'description':
                continue
            self.fp.write('%s: %s\t' % (key, parameters[key]))
        self.fp.write('\n')
        self.fp.write('DateNum\tHit\tRecall\tPrecision\n')
        self.fp.flush()

    def writeHRP(self, date_num, hit_num, recall_num, precision_num):
        self.fp.write('%s\t%s\t%s\t%s\n' % (date_num, hit_num, recall_num, precision_num))
        self.fp.flush()

    def close(self):
        self.fp.close()

