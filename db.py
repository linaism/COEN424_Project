from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import os


class MongoDB:
    def __init__(self, app=None):
        if app is not None:
            self.init_app(app)
        self.client = None

    def init_app(self, app):
        MONGO_URI = os.environ.get("MONGO_URI")
        self.client = MongoClient(MONGO_URI)
        print("MongoDB initialized")

    def get_client(self):
        return self.client
    
    def save_results(self, results):
        try:
            collection = self.client.results.entries
            result = collection.insert_one(results)
            return result
        except Exception as e:
            print(e)

    def get_all_results(self):
        try:
            collection = self.client.results.entries
            result = collection.find({}, {'_id': False})
            return result
        except Exception as e:
            print(e)