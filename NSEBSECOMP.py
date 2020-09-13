# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:11:37 2020

@author: sjacob
"""

import urllib.request
import csv
import requests
import bs4
from datetime import datetime
import lxml
from lxml import html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import datetime
import pandas_datareader.data as web
from scipy.stats import norm



     



st.title("Enter the Tickers of the company")
symb = st.text_input("Enter company tickers for comparison followed by , example: SUYOG.BO,MEGH.BO")
symb = symb.upper()
tickers = symb.split(",")

def comp_comp(tickers):
    df = pd.DataFrame(columns = ['Ticker', 'Cov_mar','Market_var','Beta','Volatility%','Return%','Sharp Ratio','CAGR','MAXDD%']) 
    data = pd.DataFrame()
    data['^BSESN'] = web.DataReader('^BSESN', data_source='yahoo', start='2015-1-1')['Adj Close'] 

    for t in tickers:
        data[t] = web.DataReader(t, data_source='yahoo', start='2015-1-1')['Adj Close']  
   
        sec_returns = np.log( data / data.shift(1) )
        cov = sec_returns.cov() * 250
        cov_with_market = cov[t][0]
 
      
        dr = data[t].pct_change()
        dcum = (1 + dr).cumprod()
        n = len(data)/252
        CAGR = ((dcum[-1])**(1/n) - 1)*100
      
 
    
        droll = dcum.cummax()
        ddw = droll - dcum
        ddmax = ddw/droll
        max_dd = (ddmax.max())*100
   
 

        market_var = sec_returns['^BSESN'].var() * 250


        MSFT_beta = cov_with_market / market_var
        MSFT_er = 0.06 + MSFT_beta * 0.05
        returns = np.log(data[t] / data[t].shift(1))
        annual_returns = returns.mean() * 250 * 100
        vols = returns.std() * 250 ** 0.5 *100
    #Sharp = (MSFT_er - 0.005) / (sec_returns[t].std() * 250 ** 0.5)
        Sharp = annual_returns/vols
 
    
 
        df = df.append(pd.Series([t,cov_with_market,market_var,MSFT_beta,vols,annual_returns,Sharp,CAGR,max_dd], index=df.columns ), ignore_index=True)
        df.sort_values(by='Return%', ascending=False)   
         
    return df
                        

if len(tickers) >=2:
    df1 = comp_comp(tickers)
    st.dataframe(df1,900,900)
