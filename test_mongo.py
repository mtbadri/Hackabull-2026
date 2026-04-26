from pymongo import MongoClient
import certifi

client = MongoClient("mongodb+srv://keko:africa786@secondbrain.fmdobsv.mongodb.net/?appName=secondbrain", tlsCAFile=certifi.where())

try:
    client.admin.command('ping')
    print("Connected successfully!")
except Exception as e:
    print("Connection failed:", e)
