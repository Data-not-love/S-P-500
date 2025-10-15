# from logGenerator import log_generator
# from dotenv import load_dotenv
import os
import logging
import pandas as pd
# import sqlite3
# from SQLite_database_connector import SQLite_database_connector
# class load:
#     def __init__(self, log_file_path, list_500, base_dir, database_path, database_name):
#         load_dotenv()
#         self.__logGenerator = log_generator(log_file_path)
#         self.__logGenerator.log_config()  # Initialize logging
#         self.__list_500 = os.getenv(list_500)
#         self.__base_dir = os.getenv(base_dir)
#         self.__database_path = os.getenv(database_path)
#         self.__database_connector = SQLite_database_connector.connect(self.__database_path, database_name) 

#     def load(self):
#         logging.info("Loading data...")
#         logging.info(f"Loading data from {self.__list_500} to {self.__base_dir}")
#         logging.info(f"Loading data completed")
#         logging.error(f"Data can't be loaded")
#         logging.error(f"Data is not found")

#     def __load_smp_500(self):
#         pass

#     def __load_smp_500_data(self):
#         pass

# load = load("LOG_FILE_PATH", "SMP_500", "RAW_DATA_PATH", "DATABASE_PATH","ok.db")
import pandas as pd
class load_dataset:
    def __init__(self, filename):
        self.__df = None

    def load(self):
        self.__df = pd.read_csv(self.__filename)
        print(f"✅ Loaded {len(self.__df)} companies from {self.__list_500}")
        logging.info(f"Loaded {self.__filename} with {len(self.__df)} companies data")
        