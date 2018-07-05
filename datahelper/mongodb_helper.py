__author__ = 'ck'

import sys
import pymongo
from enum import Enum
import json
import os
#from ..mongodb_interface import mongodb_interface
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from mongodb_interface import mongodb_interface

uri = 'mongodb://BaseAdmin:qwerty123@ds161016.mlab.com:61016/base_melange'

class mongodb_helper():

    def insert_points_data(filename):
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        raw_pt = [i.strip().split('\\')[-1][:-5] for i in open(filename).readlines()]
        points_db = db[filename+"_db"]

        #Insert the points into db
        for points in raw_pt:
            #print points
            columns = points.split(' ')

            s=""
            for index,column in enumerate(columns):
                s+='"' + 'Column'+str(index+1) + '":"'+column+'",'
            s=s[:-1]
            s="{"+s+"}"
            json_obj = json.loads(s)
            points_db.insert(json_obj)
        client.close()

    def get_points_data(self,filename): 
        db_Output = []
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        dbName = filename+"_db"
        points_db = db[dbName]
        #Get data from mongo DB
        points_db_cursor = points_db.find()

        for points in points_db_cursor:
            s=""
            length = len(points)
            for index in range(1,length):
                column_name = "Column"+str(index)
                s = s + points[column_name] + " "
            s = s[:-1]
            db_Output.append(s)
        return db_Output

    def insert_timeseries_data(self,filePath):
        print filePath
        filename = filePath.split("\\")[-1][:-4]
        print filename
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        
        raw_pt = [i.strip().split('\n') for i in open(filePath).readlines()]
        #print raw_pt
        db_name = db[filename+"_db"]

        #Set up index
        #db.db_name.ensureIndex({KEY:1})

        #Insert the points into db
        for points in raw_pt:
            columns = points[0].split(',')
            
            time_date = columns[0]
            value = columns[1]
            
            s = '{"DateTime":"' + time_date + '","Value":"'+value+'"}'
            json_obj = json.loads(s)
            db_name.insert(json_obj)
        client.close()

    def get_timeseries_data(self,filename):
        print filename
        db_Output = []
        #filename = filePath.split("\\")[-1][:-4]
        #print filename
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        
        db_name = db[filename+"_db"]

        #Get data from mongo DB
        timeseries_db_cursor = db_name.find()

        for points in timeseries_db_cursor:
            s=""
            length = len(points)
            for index in range(1,length):
                db_Output.append(points["Value"])
        return db_Output


    def recursive_file_read(self,path):
        print path
        for root, subdirs, files in os.walk(path):
            for file in os.listdir(root):
                filePath = os.path.join(root, file)
                if os.path.isdir(filePath):
                    pass
                else:
                    print "Ck"
                    self.insert_timeseries_data(filePath)

if __name__ == '__main__':
    #main(sys.argv[1:])
    #insert_into_db("rice_pt_sdh")
    #print get_from_db("rice_pt_soda")
    #insert_timeseries_data("AHU1 Final Filter DP.csv")
    mdb_helper = mongodb_helper()
    print mdb_helper.get_points_data("rice_pt_soda")
    #mdb_helper.recursive_file_read("pressure")
    print mdb_helper.get_timeseries_data("2 Mag CHW Return Temp")

