import logging
# handlers is a sub module. So TimedRotatingFileHandler cant be used only using import logging
from dotenv import load_dotenv
import os
from datetime import datetime

class log_generator:
    def __init__(self, logFilePath):
        # load .env
        load_dotenv()
        self.__logFilePath = os.getenv(logFilePath)
        self.__logFileForEachDay = self.generate_log_file_for_each_day()

    def generate_log_file_for_each_day(self):
        todayLog = datetime.now().strftime('%Y-%m-%d')
        combinedLogFileName = self.__logFilePath + "/" + todayLog + ".log"
        return combinedLogFileName


    def log_config(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.__logFileForEachDay), 
                logging.StreamHandler()
            ])

    
    def get_log_file_path(self):
        return self.__logFilePath



    def generateLog(self):
       pass


logGenerator = log_generator("LOG_FILE_PATH")
logGenerator.log_config()
logGenerator.generateLog()