import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from dotenv import load_dotenv
from logGenerator import log_generator

# find a day that have the most volume for each company
# each company will have a folder with data visualizations
# run commnad : python -m "Visualization.test"
# fix it later
# add more functions later
# add more data later
class Visualization:
    def __init__(self,log_file_path, raw_data_path, raw_date_file, chart_path, chart_name):
        load_dotenv()
        self.__logGenerator = log_generator(log_file_path)
        self.__logGenerator.log_config()
        self.__raw_data_path = os.getenv(raw_data_path)
        self.__raw_date_file = os.getenv(raw_date_file)
        self.__chart_path = os.getenv(chart_path)
        self.__chart_name = os.getenv(chart_name)


        self.__successful_visualizations = 0
        self.__failed_visualizations = 0
    
    def visualize(self):
        pass

    def __visualize_smp_500(self):
        pass

    def __visualize_smp_500_data(self):
        pass