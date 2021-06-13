#!/usr/bin/python
import datetime
import json
import os
import sys
from pymongo import MongoClient
from bson import ObjectId


def converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()
    if isinstance(o, ObjectId):
        return str(o)


# 1- get the connection parameters
directory = sys.argv[1]
serverHost = sys.argv[2]
serverPort = sys.argv[3]
serverUser = sys.argv[4]
serverPassword = sys.argv[5]
serverDb = sys.argv[6]

# 2- Connects with the database and the PedagogicalSoftwareData collection
client = MongoClient('mongodb://' + serverUser + ':' + serverPassword + '@' + serverHost + ':' + serverPort + '/' +
                     serverDb)
db = client[serverDb]
pedagogicalsoftwaredata = db.PedagogicalSoftwareData

# 3- If the directory does not exists, it creates the directory
if not os.path.exists(directory):
    os.makedirs(directory)

# 4- Opens the file to write the database information
with open(directory + '/pedagogicalinterventions.json', 'a') as outfile:
    # 5- Retrieve all the information
    for data in pedagogicalsoftwaredata.find():

        # 4.1- Deleting the elements that will not be used in machine learning
        del data['_id']
        del data['_class']
        if 'elements' in data:
            del data['elements']
        if 'binary' in data:
            del data['binary']
        if 'screenShot' in data:
            del data['screenShot']
        if 'exerciseId' in data:
            del data['exerciseId']
        if 'student' in data:
            if 'institutionId' in data['student']:
                del data['student']['institutionId']
            if 'userId' in data['student']:
                del data['student']['userId']
            if 'studentNumber' in data['student']:
                del data['student']['studentNumber']
        if 'exercise' in data:
            if 'name' in data['exercise']:
                del data['exercise']['name']
            if 'description' in data['exercise']:
                del data['exercise']['description']
        if 'solutionDistance' in data:
            if 'nextSteps' in data['solutionDistance']:
                del data['solutionDistance']['nextSteps']


        # 4.2- Write the information in the file defined
        json.dump(data, outfile, default=converter)
        outfile.write('\n')
