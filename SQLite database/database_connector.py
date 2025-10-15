import sqlite3
from dotenv import load_dotenv
import os
import logging

class SQLite_database_connector:
    def __init__(self, database_path):
        load_dotenv()
        self.__database_path = os.getenv(database_path)
        self.__connection = None
        self.__cursor = None

    def connect(self, database_name):
        self.__connection = sqlite3.connect(self.__database_path + "/" + database_name)
        self.__cursor = self.__connection.cursor()
        logging.info("✅ Database connected")
        logging.error("Database connection failed")
        logging.error("Database connection is not found")

    def close(self):
        self.__connection.close()
        logging.info("✅ Database disconnected")
        logging.error("Database connection failed")
        logging.error("Database connection is not found")