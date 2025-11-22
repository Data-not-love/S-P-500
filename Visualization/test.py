import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from dotenv import load_dotenv
from logGenerator import log_generator
from Pipeline.Calculations.basicCalculations import basicCalculations
# find a day that have the most volume for each company
# each company will have a folder with data visualizations
# run commnad : python -m "Visualization.test"
# fix it later
# add more functions later
# add more data later
class Visualization:
    def __init__(self,log_file_path, raw_data_path, raw_date_file, chart_path, chart_name):
        load_dotenv()
        self.__basicCalculations = basicCalculations(log_file_path, raw_date_file)        
        

        
    
   