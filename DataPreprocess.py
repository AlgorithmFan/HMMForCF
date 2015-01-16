#!usr/vin/env python
#coding:utf-8
import time
import codecs

def getListFromString(line):
    strList = line.split('\t')
    filterList = []
    for t in strList:
        if t!='':
            filterList.append(t)
    return filterList

def transformData(readFile, writeFile, artistFile, user_num=1000):
    '''Read data from readFile, and write data to writeFile, artistFile.'''
    read_fp = codecs.open(readFile, 'rb', 'utf-8')
    write_fp = codecs.open(writeFile, 'wb', 'utf-8')

    artist_num, artists = 0, {}
    read_num = 0
    timeFormat = "%Y-%m-%dT%H:%M:%SZ"
    while True:
        line = read_fp.readline()
        if line == '': break
        temp = getListFromString(line)     #Split the string by \t
        read_num += 1
        if len(temp)<3: continue        #If the length of splited list is below 6, this string is not available.
        print read_num, temp[0]
        try:
            user_id = int(temp[0][6:])      #Transform the user_id from string to int
            if user_id > user_num:   break
            timestamp = int(time.mktime(time.strptime(temp[1], timeFormat)))    #Transform the time string to int
            artist_id = temp[2]                         #the artist_id
        except:
            continue
        if artist_id not in artists:
            artists[artist_id] = {'index':artist_num, 'users':set()}
            artist_num += 1
        artists[artist_id]['users'].add(user_id)
        write_fp.write('%d\t%d\t%d\n' % (user_id, artists[artist_id]['index'], timestamp))
    read_fp.close()
    write_fp.close()
    fp = codecs.open(artistFile, 'wb', 'utf-8')
    for artist_id in artists:
        if len(artists[artist_id]['users']) <= 20:
            continue
        fp.write('%s\t%d\t%d\n' % (artist_id, artists[artist_id]['index'], len(artists[artist_id]['users'])))
    fp.close()


if __name__=='__main__':
    filename = r"E:\DataSet\lastfm-dataset-1K\userid-timestamp-artid-artname-traid-traname.tsv"
    users_map_artists = transformData(filename, 'users_artists_timestamp.txt', 'artists.txt')