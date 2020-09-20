# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:05:16 2020

@author: sjacob
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:37:26 2020

@author: sjacob
"""

# -*- coding: utf-8 -*-


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
from nsepy import get_history
import matplotlib
from nsepy import get_history
from datetime import date
import numpy as np
from io import StringIO
import pandas as pd


def MACD(DF,a,b,c):
    
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""

    df = DF.copy()
    df["MA_Fast"]=df["Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return df

# Visualization - plotting MACD/signal along with close price and volume for last 100 data points



# Visualization - Using object orient approach
# Get the figure and the axes

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2


def BollBnd(DF,n):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MA"] = df['Close'].rolling(n).mean()
    df["BB_up"] = df["MA"] + 2*df['Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] - 2*df['Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    df.dropna(inplace=True)
    return df



def RSI(DF,n):
    "function to calculate RSI"
    df = DF.copy()
    df['delta']=df['Close'] - df['Close'].shift(1)
    df['gain']=np.where(df['delta']>=0,df['delta'],0)
    df['loss']=np.where(df['delta']<0,abs(df['delta']),0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n-1)*avg_gain[i-1] + gain[i])/n)
            avg_loss.append(((n-1)*avg_loss[i-1] + loss[i])/n)
    df['avg_gain']=np.array(avg_gain)
    df['avg_loss']=np.array(avg_loss)
    df['RS'] = df['avg_gain']/df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    return df['RSI']

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2


def ADX(DF,n):
    "function to calculate ADX"
    df2 = DF.copy()
    df2['TR'] = ATR(df2,n)['TR'] #the period parameter of ATR function does not matter because period does not influence TR calculation
    df2['DMplus']=np.where((df2['High']-df2['High'].shift(1))>(df2['Low'].shift(1)-df2['Low']),df2['High']-df2['High'].shift(1),0)
    df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
    df2['DMminus']=np.where((df2['Low'].shift(1)-df2['Low'])>(df2['High']-df2['High'].shift(1)),df2['Low'].shift(1)-df2['Low'],0)
    df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for i in range(len(df2)):
        if i < n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == n:
            TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
    df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
    df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
    df2['DIsum']=df2['DIplusN']+df2['DIminusN']
    df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2*n-1:
            ADX.append(np.NaN)
        elif j == 2*n-1:
            ADX.append(df2['DX'][j-n+1:j+1].mean())
        elif j > 2*n-1:
            ADX.append(((n-1)*ADX[j-1] + DX[j])/n)
    df2['ADX']=np.array(ADX)
    return df2['ADX']

def OBV(DF):
    """function to calculate On Balance Volume"""
    df = DF.copy()
    df['daily_ret'] = df['Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']


def stochOscltr(DF,a=20,b=3):
    """function to calculate Stochastics
       a = lookback period
       b = moving average window for %D"""
    df = DF.copy()
    df['C-L'] = df['Close'] - df['Low'].rolling(a).min()
    df['H-L'] = df['High'].rolling(a).max() - df['Low'].rolling(a).min()
    df['%K'] = df['C-L']/df['H-L']*100
    
    df['%D'] = df['%K'].ewm(span=b,min_periods=b).mean()
    return df['%K'].rolling(b).mean(),df['%K'].rolling(b).mean()    


def plot_band(df,text):
    df.reset_index(level=0, inplace=True)
    fig = go.Figure()
 
#Set up traces
    fig.add_trace(go.Scatter(x=df['Date'], y= df['MA'],line=dict(color='blue', width=.7), name = 'Middle Band'))
    fig.add_trace(go.Scatter(x=df['Date'], y= df['BB_up'],line=dict(color='red', width=1.5), name = 'Upper Band (Sell)'))
    fig.add_trace(go.Scatter(x=df['Date'], y= df['BB_dn'],line=dict(color='green', width=1.5), name = 'Lower Band (Buy)'))
    fig.add_trace(go.Scatter(x=df['Date'], y= df['Signal'],line=dict(color='Orange', width=1.5), name = 'Singal'))
    fig.add_trace(go.Scatter(x=df['Date'], y= df['MACD'],line=dict(color='black', width=1.5), name = 'Slow'))
    #fig.add_trace(go.Scatter(x=df['Date'], y= df['Volume'],line=dict(color='violet', width=1.5), name = 'Slow'))
    #fig.add_trace(go.Scatter(x=df['Date'], y= df['MACD'],line=dict(color='black', width=1.5))
                  
                  
    
    fig.add_trace(go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name = 'market data'))
    # Add titles
    fig.update_layout(
        title='Bollinger Band Strategy for'+text,
        yaxis_title='Index Price')
    
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    #Show
    fig.show()


    



st.title("NSE Sector Analysis")
symb = st.text_input("Enter End date in yyyy-MM-DD format","2020-9-18")
end_dat =  symb.split("-")
year = int(end_dat[0])
month = int(end_dat[1])
day = int(end_dat[2])

nifty = get_history('NIFTY', date(2015, 1, 1), date(year, month, day), index=True)
niftyauto = get_history('NIFTY AUTO', date(2015, 1, 1), date(year, month, day), index=True)
niftyrealty = get_history('NIFTY REALTY', date(2015, 1, 1), date(year, month, day), index=True)
niftymnc = get_history('NIFTY MNC', date(2015, 1, 1), date(year, month, day), index=True)
niftyfmcg = get_history('NIFTY FMCG', date(2015, 1, 1), date(year, month, day), index=True)
niftypharma = get_history('NIFTY PHARMA', date(2015, 1, 1), date(year, month, day), index=True)
niftymetal = get_history('NIFTY METAL', date(2015, 1, 1),date(year, month, day), index=True)
niftyit = get_history('NIFTY IT', date(2015, 1, 1), date(year, month, day), index=True)

nifty = MACD(nifty, 12, 26, 9)
nifty = BollBnd(nifty,20)    
niftyauto = MACD(niftyauto, 12, 26, 9)
niftyauto = BollBnd(niftyauto,20)   
niftyrealty = MACD(niftyrealty, 12, 26, 9)
niftyrealty = BollBnd(niftyrealty,20)  
niftymnc = MACD(niftymnc, 12, 26, 9)
niftymnc = BollBnd(niftymnc,20)  
niftyfmcg = MACD(niftyfmcg, 12, 26, 9)
niftyfmcg = BollBnd(niftyfmcg,20)  
niftypharma = MACD(niftypharma, 12, 26, 9)
niftypharma = BollBnd(niftypharma,20)  
niftyit = MACD(niftyit, 12, 26, 9)
niftyit = BollBnd(niftyit,20)  
niftymetal = MACD(niftymetal, 12, 26, 9)
niftymetal = BollBnd(niftymetal,20)  


plot_band(nifty,"Nifty 50")
plot_band(niftyauto,"Nifty Auto")
plot_band(niftyrealty,"Nifty Realty ")
plot_band(niftyfmcg,"Nifty FMCG ")
plot_band(niftypharma,"Nifty Pharma ")
plot_band(niftymnc,"Nifty MNC ")
plot_band(niftyit,"Nifty IT ")
plot_band(niftymetal,"Nifty Metal ")



tickers = ['NIFTY', 'NIFTY AUTO','NIFTY MNC','NIFTY PHARMA','NIFTY IT','NIFTY METAL','NIFTY REALTY', 'NIFTY FMCG','NIFTY ENERGY'
          ,'NIFTY BANK','NIFTY INFRA','NIFTY MEDIA','NIFTY SERV SECTOR','NIFTY CONSUMPTION','NIFTY OIL & GAS',
          'NIFTY PVT BANK','NIFTY FIN SERVICE']
data = pd.DataFrame()

for t in tickers:

    try:
        data[t] =get_history(t, date(2015, 1, 1), date(year, month, day), index=True)['Close']
    except:
        continue
    
returns = np.log(data / data.shift(1))
vols = returns.std() * 252 ** 0.5 *100
annual_returns = returns.mean() * 252 * 100

df = pd.DataFrame(columns = ['Ticker', 'Cov_mar','Market_var','Beta','Volatility%','Return%','Sharp Ratio','CAGR','MAXDD%']) 
data = pd.DataFrame()
data['^BSESN'] = web.DataReader('^BSESN', data_source='yahoo', start='2015-1-1')['Close'] 

for t in tickers:
 
    data[t] = get_history(t, date(2015, 1, 1), date(2020, 9, 18), index=True)['Close']
    #data['^BSESN'] = web.DataReader(t, data_source='yahoo', start='2010-1-1')['Adj Close']  
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
                                    
                        
st.dataframe(df,800,800)