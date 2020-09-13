# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
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
import pandas_datareader
import datetime
import pandas_datareader.data as web
from scipy.stats import norm
start = datetime.datetime(2005, 1, 1)
from nsetools import Nse
from nsepy import get_history
from datetime import date
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima                              # for determining ARIMA orders
from statsmodels.tsa.arima_model import ARIMA
# For non-seasonal data
import statsmodels.api as sm
# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}


def data_read(symbol,start,end):
   
    bse = '^BSESN'
    tickers = [symbol, bse]
    data = pd.DataFrame()
    for t in tickers:
        data[t] = web.DataReader(t, data_source='yahoo', start='2012-1-1')['Adj Close']  
     
    sec_returns = np.log( data / data.shift(1) )
    cov = sec_returns.cov() * 250
    cov_with_market = cov.iloc[0,1]
    market_var = sec_returns[bse].var() * 250
    MSFT_beta = cov_with_market / market_var
    MSFT_er = 0.06 + MSFT_beta * 0.05
    st.header("The expected Return using CAPM")
    MSFT_er*100
    st.header("The Beta of the stock "+symbol+" is:")
    MSFT_beta
    
    
    returns = np.log(data / data.shift(1))
    vols = returns.std() * 250 ** 0.5 *100
    st.header("The Volatility of the Stock is:")
    vols
    annual_returns = returns.mean() * 250 * 100
    st.header("The Annual Return of the Stock is:")
    annual_returns
    sharpe_ratio = annual_returns[symbol]/vols[symbol]
    st.header("The Sharpe Ratio of the Stock is:")
    sharpe_ratio
    

    

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod()
    n = len(df)/252
    CAGR = (df["cum_return"][-1])**(1/n) - 1
    return CAGR
def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    vol = df["daily_ret"].std() * np.sqrt(252)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    
def sortino(DF,rf):
    "function to calculate sortino ratio ; rf is the risk free rate"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    neg_vol = df[df["daily_ret"]<0]["daily_ret"].std() * np.sqrt(252)
    sr = (CAGR(df) - rf)/neg_vol
    return sr

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd
    
def calmar(DF):
    "function to calculate calmar ratio"
    df = DF.copy()
    clmr = CAGR(df)/max_dd(df)
    return clmr

def monta_carlo(data):
  

    log_returns = np.log(1 + data.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()

    #drift.values
    #stdev.values

    t_intervals = 250
    iterations = 1000

    daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))
    S0 = data.iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]
    st.header("MONTE CARLO SIMULATION:")
    st.line_chart(price_list)
    st.header("The MAXIMUM PRICE OBTAINED FROM MONTE CARLO SIMULATION IS:")
    a= price_list.max()
    a
    st.header("The MINIMUM PRICE OBTAINED FROM MONTE CARLO SIMULATION IS:")
  
    b = price_list.min()
    b
    st.header("The MEAN PRICE OBTAINED FROM MONTE CARLO SIMULATION IS:")
    c = price_list.mean()
    c
    
    
def holt_time(df):


    lt = len(df)

    train_data = df.iloc[:lt-1]# Goes up to but not including 108
    test_data = df.iloc[lt-1:]
    st.header("The Latest Stock Information")
    dft = df.tail()
    st.dataframe(dft,800,800)
    fitted_model = ExponentialSmoothing(train_data['Adj Close'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
    test_predictions = fitted_model.forecast(12).rename('HW Forecast')
    st.header("The Time Series Forcast Prediction Testing")
    test_predictions
   
    final_model = ExponentialSmoothing(df['Adj Close'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
    forecast_predictions = fitted_model.forecast(36).rename('HW Forecast')
    st.header("The Time Series Forcast Prediction for next 3 years")
    st.dataframe(forecast_predictions,700,700)
   
def file_write(symbol):
    df = pd.DataFrame()
 
    df = web.DataReader(symbol, data_source='yahoo', start='2012-1-1')
    df.to_csv('test1.csv')

def read_file():
    df = pd.read_csv("test1.csv")
#df.set_index('Date', inplace=True)
    df['Date'] = df['Date'].apply(pd.to_datetime)
    df.set_index('Date', inplace=True)
    df =df.resample(rule='M').mean()
    df =df.dropna()
    return df
def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""

    df = DF.copy()
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
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
    df["MA"] = df['Adj Close'].rolling(n).mean()
    df["BB_up"] = df["MA"] + 2*df['Adj Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] - 2*df['Adj Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    df.dropna(inplace=True)
    return df



def RSI(DF,n):
    "function to calculate RSI"
    df = DF.copy()
    df['delta']=df['Adj Close'] - df['Adj Close'].shift(1)
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
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
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

def sarima(df):
    train = df.iloc[:len(df)-12]
    test = df.iloc[len(df)-12:]
    model = SARIMAX(train['Close'],order=(2,1,0),seasonal_order=(2,0,0,12))
    results = model.fit()
    start=len(train)
    end=len(train)+len(test)-1
    #predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA(0,1,0)(2,0,0,12) Predictions')
    fcast202 = results.predict(start = len(df)-1, end = len(df)+12, dynamic = True )
    st.header("The SARIMA Prediction is:")
    st.dataframe(fcast202,500,500)
    
def arima(df):
    model = ARIMA(df['Close'], order=(2, 0, 0))  
    results_AR = model.fit()
    fcast202 = results_AR.predict(start = len(df)-1, end = len(df)+12, dynamic = True )
    st.header("The ARIMA Prediction is:")
    st.dataframe(fcast202,500,500)
    
def financial_report(ticker):
    url = 'https://in.finance.yahoo.com/quote/'+ ticker+ '/financials?p='+ticker
    


# Fetch the page that we're going to parse, using the request headers
# defined above
    page = requests.get(url)
    
    # Parse the page with LXML, so that we can start doing some XPATH queries
    # to extract the data that we want
    tree = html.fromstring(page.content)
    
    # Smoke test that we fetched the page by fetching and displaying the H1 element
    tree.xpath("//h1/text()")
    table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")
    
    # Ensure that some table rows are found; if none are found, then it's possible
    # that Yahoo Finance has changed their page layout, or have detected
    # that you're scraping the page.
    assert len(table_rows) > 0
    
    parsed_rows= []
    
    for table_row in table_rows:
        parsed_row = []
        el = table_row.xpath("./div")
        none_count = 0
        for rs in el:
            try:
                (text,) = rs.xpath('.//span/text()[1]')
                parsed_row.append(text)
            except ValueError:
                
                parsed_row.append(np.NaN)
                none_count += 1
    
        if (none_count < 4):
            parsed_rows.append(parsed_row)
    dfn = pd.DataFrame(parsed_rows)
    
        
    url = 'https://in.finance.yahoo.com/quote/'+ ticker +'/balance-sheet?p='+ticker
        
    
    
    # Fetch the page that we're going to parse, using the request headers
    # defined above
    page = requests.get(url)
    
    # Parse the page with LXML, so that we can start doing some XPATH queries
    # to extract the data that we want
    tree = html.fromstring(page.content)
    
    # Smoke test that we fetched the page by fetching and displaying the H1 element
    tree.xpath("//h1/text()")
    table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")
    
    # Ensure that some table rows are found; if none are found, then it's possible
    # that Yahoo Finance has changed their page layout, or have detected
    # that you're scraping the page.
    assert len(table_rows) > 0
    
    parsed_rows= []
    
    for table_row in table_rows:
        parsed_row = []
        el = table_row.xpath("./div")
        
        none_count = 0
        for rs in el:
            
            try:
                (text,) = rs.xpath('.//span/text()[1]')
                parsed_row.append(text)
            except ValueError:
                
                parsed_row.append(np.NaN)
                none_count += 1
    
        if (none_count < 4):
            
            parsed_rows.append(parsed_row)
    dfn = dfn.append(parsed_rows)
    
    url = 'https://in.finance.yahoo.com/quote/'+ ticker+ '/cash-flow?p='+ ticker
        
    
    
    # Fetch the page that we're going to parse, using the request headers
    # defined above
    page = requests.get(url)
    
    # Parse the page with LXML, so that we can start doing some XPATH queries
    # to extract the data that we want
    tree = html.fromstring(page.content)
    
    # Smoke test that we fetched the page by fetching and displaying the H1 element
    tree.xpath("//h1/text()")
    table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")
    
    # Ensure that some table rows are found; if none are found, then it's possible
    # that Yahoo Finance has changed their page layout, or have detected
    # that you're scraping the page.
    assert len(table_rows) > 0
    
    parsed_rows= []
    
    for table_row in table_rows:
        parsed_row = []
        el = table_row.xpath("./div")
        none_count = 0
        for rs in el:
            try:
                  
                (text,) = rs.xpath('.//span/text()[1]')
                parsed_row.append(text)
            except ValueError:
                parsed_row.append(np.NaN)
                none_count += 1
    
        if (none_count < 4):
            parsed_rows.append(parsed_row)
    dfn = dfn.append(parsed_rows)
    dfn.to_csv(ticker+'.csv')
    st.header("The Copany Financial Report %:")
    st.dataframe(dfn,800,800)
    #dfn
    
def fetch_results(search_term, number_results, language_code):
    
    assert isinstance(search_term, str), 'Search term must be a string'
    assert isinstance(number_results, int), 'Number of results must be an integer'
    escaped_search_term = search_term.replace(' ', '+')
 
    google_url = 'https://www.google.com/search?q={}&num={}&hl={}'.format(escaped_search_term, number_results, language_code)
    response = requests.get(google_url, headers=USER_AGENT)
    response.raise_for_status()
 
    return search_term, response.text


def parse_results(html, keyword):
    soup = BeautifulSoup(html, 'html.parser')
 
    found_results = []
    rank = 1
    result_block = soup.find_all('div', attrs={'class': 'g'})
    for result in result_block:
 
        link = result.find('a', href=True)

   
        title = result.find('h3')
        description = result.find('span', attrs={'class': 'st'})
        if link and title:
            link = link['href']
            title = title.get_text()
            if description:
                description = description.get_text()
            if link != '#':
                found_results.append({'keyword': keyword, 'rank': rank, 'title': title, 'description': description})
                rank += 1
    return found_results      

def get_key_stats(tgt_website):
 
    # The web page is make up of several html table. By calling read_html function.
    # all the tables are retrieved in dataframe format.
    # Next is to append all the table and transpose it to give a nice one row data.
    df_list = pd.read_html(tgt_website)
    result_df = df_list[0]
 
    for df in df_list[1:]:
        result_df = result_df.append(df)
 
    # The data is in column format.
    # Transpose the result to make all data in single row
    return result_df.set_index(0).T 
     

dfticker = pd.read_csv("C:\\STOCK\\LATSTOCK.csv",index_col=False,encoding= 'unicode_escape')


#MSFT_beta


symbol = " "
#select = st.sidebar.text_input('Share symbol:',lsta)
st.sidebar.title("Tickers")
symbol = st.sidebar.text_input("Ticker in capital")
st.sidebar.title("User Inputs")
start = st.sidebar.text_input("Start Date","2012-1-1")
end = st.sidebar.text_input("End Date","2020-8-31")


ticker =symbol.upper()

Company = symbol

st.title("Share Price analysis of   "+Company)
#st.markdown("This application is a Share Price Analysis")
#st.sidebar.markdown("This application is a Share Price Anaalysis")

if symbol != " ":
    symbol = symbol+'.BO'
    data_read(symbol,start,end)
   
    df = web.DataReader(symbol, data_source='yahoo', start='2012-1-1')
   
    cg = CAGR(df)
    vola = volatility(df)
    shar = sharpe(df,0.06)
    sor = sortino(df,0.06)
    md = max_dd(df)
    calm = calmar(df)

    st.header("The Compunded Annual Groth of the stock %:")
    cg*100
    st.header("The Volatility of the stock is  %:")
    vola*100
    st.header(" The Sharp Ratio is:")
    shar
    st.header(" The Sortino Ratio is: ")
    sor
    st.header(" The MAX Draw Down it shows loss for the investment % ")
    md*100
 
    st.header(" The Calamr Ratio is(It shows the Risk Adjusted Return): ")
    calm
    data = pd.DataFrame()

    
    data[symbol] = web.DataReader(symbol, data_source='yahoo', start='2012-1-1')['Adj Close']
    monta_carlo(data)
    df['EWMA12'] = df['Close'].ewm(span=12,adjust=False).mean()
    df['12-month-SMA'] = df['Close'].rolling(window=12).mean()
    df[['Close','EWMA12','12-month-SMA']].plot(figsize=(12,8)).autoscale(axis='x',tight=True);
    st.header("EWMA, 12 month SMA")
    st.line_chart( df[['Close','EWMA12','12-month-SMA']])
    file_write(symbol)
    df1 = read_file()
    holt_time(df1)
    sarima(df1)
    arima(df1)
    
    df = MACD(df1, 12, 26, 9)

    st.header("MACD Chart")
    st.line_chart(df[["MACD","Signal"]].tail(150))
    
    
    ohlcv = df.tail(70)
    # Visualizing Bollinger Band of the stocks for last 100 data points
    
    #BollBnd(ohlcv,15).iloc[-100:,[-4,-3,-2]].plot(title="Bollinger Band",figsize=(20,8))
    st.header("RSI")
    temp1 =RSI(df1,30)
    st.line_chart(temp1.tail(200))
    st.header("OBV")
    temp3 = OBV(df1)
    st.line_chart(temp3.tail(100))
    st.header("TR and ATR")
    temp= ATR(df1,20)
    st.line_chart(temp[["TR","ATR"]].tail(100))
    tgt_website = r'https://sg.finance.yahoo.com/quote/'+symbol+'/key-statistics?p='+symbol
    result_df = get_key_stats(tgt_website)
    st.dataframe(result_df.T,800,800)
   
    financial_report(symbol)
    keyword, html = fetch_results(Company + ' is a good company to buy stock', 100, 'en')
    results = parse_results(html,keyword)
    dfnews =pd.DataFrame(results)
    st.header("News about the company")
    st.dataframe(dfnews['description'],700,700)




  
   
   
 
    
    
    
    
 



