from logGenerator import log_generator
from dotenv import load_dotenv
import os
import logging

class extract:
    def __init__(self, log_file_path, list_500, base_dir):
        load_dotenv()
        self.__logGenerator = log_generator(log_file_path)
        self.__logGenerator.log_config()  # Initialize logging
        self.__list_500 = os.getenv(list_500)
        self.__base_dir = os.getenv(base_dir)

    def extract(self):
        logging.info("Extracting data...")
        logging.info(f"Extracting data from {self.__list_500} to {self.__base_dir}")
        logging.info(f"Extracting data successfully")
        logging.error(f"Data can't be extracted")
        logging.error(f"Data is not found")

    def __extract_smp_500(self):
        pass

    def __extract_smp_500_data(self):
        pass
