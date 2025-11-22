import pandas as pd
import logging
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('../../')
from logGenerator import log_generator

# basic calculations
# run command : python -m "Pipeline.Calculations.basicCalculations"
# for each company and then create a chart for it

class basicCalculations:
    def __init__(self, log_file_path, raw_date_file):
        load_dotenv()
        self.__logGenerator = log_generator(log_file_path)
        self.__logGenerator.log_config()
        self.__raw_date_file = raw_date_file
        self.__df = None
        
    def load_data(self):
        self.__df = pd.read_csv(self.__raw_date_file)
        print(f"✅ Loaded {len(self.__df)} companies from {self.__raw_date_file}")
        logging.info(f"Loaded {len(self.__df)} companies from {self.__raw_date_file}. Loaded successfully.")
    
    def price_change_calculation(self):
    # price = close - open
        self.__df['Price Change'] = self.__df['Close'] - self.__df['Open']
        print(self.__df['Price Change'])
        logging.info(f"Price Change was calculated successfully.")


    def price_change_in_percentage(self):
    # P% = ( closed -open ) / open * 100
        self.__df['Price Change Percentage'] = (self.__df['Close'] - self.__df['Open']) / self.__df['Open'] * 100
        print(self.__df['Price Change Percentage'])
        logging.info(f"Price Change Percentage was calculated successfully.")

    def daily_range(self):
    # range = high - low
        self.__df ['Daily Range'] = self.__df['High'] - self.__df['Low']
        logging.info(f"Daily Range was calculated successfully.")
        
        print("✅ All calculations completed successfully!")
        logging.info("Basic calculations pipeline completed")
        
        
    def create_chart(self):
        """Create interactive TradingView-style chart"""
        # Create subplots for multiple charts
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price Change Over Time', 'Price Change Percentage', 'Daily Range'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Convert Date to datetime if it's not already
        self.__df['Date'] = pd.to_datetime(self.__df['Date'])
        
        # Chart 1: Price Change
        fig.add_trace(
            go.Scatter(
                x=self.__df['Date'],
                y=self.__df['Price Change'],
                mode='lines',
                name='Price Change',
                line=dict(color='turquoise', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price Change:</b> $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Chart 2: Price Change Percentage
        fig.add_trace(
            go.Scatter(
                x=self.__df['Date'],
                y=self.__df['Price Change Percentage'],
                mode='lines',
                name='Price Change %',
                line=dict(color='green', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price Change %:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Chart 3: Daily Range
        fig.add_trace(
            go.Scatter(
                x=self.__df['Date'],
                y=self.__df['Daily Range'],
                mode='lines',
                name='Daily Range',
                line=dict(color='red', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Daily Range:</b> $%{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout for TradingView-like appearance
        fig.update_layout(
            title={
                'text': 'Interactive S&P500 Stock Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=1000,
            showlegend=True,
            hovermode='x unified',
            # Enable zoom and pan functionality
            dragmode='zoom',
            # TradingView-like color scheme
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            # Grid styling
            xaxis=dict(
                gridcolor='#404040',
                showgrid=True,
                gridwidth=1,
                # Enable zoom on x-axis
                fixedrange=False
            ),
            yaxis=dict(
                gridcolor='#404040',
                showgrid=True,
                gridwidth=1,
                # Enable zoom on y-axis
                fixedrange=False
            )
        )
        
        # Update all subplots with TradingView styling
        for i in range(1, 4):
            fig.update_xaxes(
                gridcolor='#404040',
                showgrid=True,
                gridwidth=1,
                # Enable zoom on all x-axes
                fixedrange=False,
                row=i, col=1
            )
            fig.update_yaxes(
                gridcolor='#404040',
                showgrid=True,
                gridwidth=1,
                # Enable zoom on all y-axes
                fixedrange=False,
                row=i, col=1
            )
        

        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]),
                    font=dict(color='black', size=12),
                    bgcolor='white',
                    bordercolor='gray',
                    borderwidth=1
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Show the interactive chart
        fig.show()
        
        # Save as HTML for sharing
        fig.write_html("interactive_stock_chart.html")
        print("📊 Interactive chart saved as 'interactive_stock_chart.html'")
        logging.info("Interactive TradingView-style chart created")
        

        
basicCalculations = basicCalculations("LOG_FILE_PATH","/home/lana-ddos/Downloads/cd1/raw data/3M/MMM 1y.csv")
basicCalculations.load_data()
basicCalculations.price_change_calculation()
basicCalculations.price_change_in_percentage()
basicCalculations.daily_range()
basicCalculations.create_chart()