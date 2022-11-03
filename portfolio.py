# import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import datetime
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import datetime
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm
import requests
from bs4 import BeautifulSoup
import cufflinks as cf
cf.go_offline()
from lxml import html
import requests
import numpy as np
import pandas as pd
import os
import talib

import datetime
import pandas as pd
import yfinance as yf

patterns = {
    'CDL2CROWS':'Two Crows',
    'CDL3BLACKCROWS':'Three Black Crows',
    'CDL3INSIDE':'Three Inside Up/Down',
    'CDL3LINESTRIKE':'Three-Line Strike',
    'CDL3OUTSIDE':'Three Outside Up/Down',
    'CDL3STARSINSOUTH':'Three Stars In The South',
    'CDL3WHITESOLDIERS':'Three Advancing White Soldiers',
    'CDLABANDONEDBABY':'Abandoned Baby',
    'CDLADVANCEBLOCK':'Advance Block',
    'CDLBELTHOLD':'Belt-hold',
    'CDLBREAKAWAY':'Breakaway',
    'CDLCLOSINGMARUBOZU':'Closing Marubozu',
    'CDLCONCEALBABYSWALL':'Concealing Baby Swallow',
    'CDLCOUNTERATTACK':'Counterattack',
    'CDLDARKCLOUDCOVER':'Dark Cloud Cover',
    'CDLDOJI':'Doji',
    'CDLDOJISTAR':'Doji Star',
    'CDLDRAGONFLYDOJI':'Dragonfly Doji',
    'CDLENGULFING':'Engulfing Pattern',
    'CDLEVENINGDOJISTAR':'Evening Doji Star',
    'CDLEVENINGSTAR':'Evening Star',
    'CDLGAPSIDESIDEWHITE':'Up/Down-gap side-by-side white lines',
    'CDLGRAVESTONEDOJI':'Gravestone Doji',
    'CDLHAMMER':'Hammer',
    'CDLHANGINGMAN':'Hanging Man',
    'CDLHARAMI':'Harami Pattern',
    'CDLHARAMICROSS':'Harami Cross Pattern',
    'CDLHIGHWAVE':'High-Wave Candle',
    'CDLHIKKAKE':'Hikkake Pattern',
    'CDLHIKKAKEMOD':'Modified Hikkake Pattern',
    'CDLHOMINGPIGEON':'Homing Pigeon',
    'CDLIDENTICAL3CROWS':'Identical Three Crows',
    'CDLINNECK':'In-Neck Pattern',
    'CDLINVERTEDHAMMER':'Inverted Hammer',
    'CDLKICKING':'Kicking',
    'CDLKICKINGBYLENGTH':'Kicking - bull/bear determined by the longer marubozu',
    'CDLLADDERBOTTOM':'Ladder Bottom',
    'CDLLONGLEGGEDDOJI':'Long Legged Doji',
    'CDLLONGLINE':'Long Line Candle',
    'CDLMARUBOZU':'Marubozu',
    'CDLMATCHINGLOW':'Matching Low',
    'CDLMATHOLD':'Mat Hold',
    'CDLMORNINGDOJISTAR':'Morning Doji Star',
    'CDLMORNINGSTAR':'Morning Star',
    'CDLONNECK':'On-Neck Pattern',
    'CDLPIERCING':'Piercing Pattern',
    'CDLRICKSHAWMAN':'Rickshaw Man',
    'CDLRISEFALL3METHODS':'Rising/Falling Three Methods',
    'CDLSEPARATINGLINES':'Separating Lines',
    'CDLSHOOTINGSTAR':'Shooting Star',
    'CDLSHORTLINE':'Short Line Candle',
    'CDLSPINNINGTOP':'Spinning Top',
    'CDLSTALLEDPATTERN':'Stalled Pattern',
    'CDLSTICKSANDWICH':'Stick Sandwich',
    'CDLTAKURI':'Takuri (Dragonfly Doji with very long lower shadow)',
    'CDLTASUKIGAP':'Tasuki Gap',
    'CDLTHRUSTING':'Thrusting Pattern',
    'CDLTRISTAR':'Tristar Pattern',
    'CDLUNIQUE3RIVER':'Unique 3 River',
    'CDLUPSIDEGAP2CROWS':'Upside Gap Two Crows',
    'CDLXSIDEGAP3METHODS':'Upside/Downside Gap Three Methods'
}
st.title("Market Dashboard Application")

st.sidebar.header("User Input")


class ValueAtRiskMonteCarlo:

    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        stock_data = np.zeros([self.iterations, 1])
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price
        stock_price = self.S * np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.n) * rand)
        # we have to sort the stock prices to determine the percentile
        stock_price = np.sort(stock_price)

        # it depends on the confidence level: 95% -> 5 and 99% -> 1
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        return self.S - percentile

def get_input():
    symbol = st.sidebar.text_input("Symbol", "BTC-USD")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2021,1,1))
    end_date = st.sidebar.date_input("End Date",datetime.date(2021,12,31))
    return symbol, start_date, end_date



def cal_data(tickers,start_date,end_date):

    df = pd.DataFrame(columns=['Ticker', 'Cov_mar', 'Market_var', 'Beta', 'Volatility%', 'Return%', 'CV', 'Sharp Ratio', 'CAGR',
                 'MAXDD%', 'Value at Risk'])
    #tickers = symbols.split(",")

    data = pd.DataFrame()
    data['^BSESN'] = yf.download('^BSESN', start=start_date, end=end_date)['Adj Close']
    for t in tickers:
       

        t = t.upper()
        t.strip()
        t= t+".NS"
        data[t] = yf.download(t, start=start_date, end=end_date)['Adj Close']


        sec_returns = np.log(data / data.shift(1))
        cov = sec_returns.cov() * 252
        cov_with_market = cov[t][0]

        dr = data[t].pct_change()
        dcum = (1 + dr).cumprod()
        n = len(data[t]) / 252
        CAGR = ((dcum[-1]) ** (1 / n) - 1) * 100

        droll = dcum.cummax()
        ddw = droll - dcum
        ddmax = ddw / droll
        max_dd = (ddmax.max()) * 100

        market_var = sec_returns['^BSESN'].var() * 252

        returns = np.log(data[t] / data[t].shift(1))
        MSFT_beta = cov_with_market / market_var
        MSFT_er = returns.mean() * 252
        Sharp = (MSFT_er - 0.05) / (sec_returns[t].std() * 252 ** 0.5)

        # returns = np.log(data[t] / data[t].shift(1))
        vols = returns.std() * 252 ** 0.5 * 100
        annual_returns = returns.mean() * 252 * 100
        CV = (vols / annual_returns) * 100
        S = 100000  # this is the investment (stocks or whatever)
        c = 0.95  # condifence level: this time it is 99%
        n = 730  # 1 day
        iterations = 100000  # number of paths in the Monte-Carlo simulation
        mu = np.mean(returns)
        sigma = np.mean(returns.std())

        model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)
        val = model.simulation()
        
        df = df.append(
            pd.Series([t, cov_with_market, market_var, MSFT_beta, vols, annual_returns, CV, Sharp, CAGR, max_dd, val],
                      index=df.columns), ignore_index=True)
        df.to_csv('QUALSTOCKS12022.csv')
    return df,data
def plot_data(df,a,b,t1,t2):
    plt.figure(figsize=[20, 15])
    x = df[a]
    y = df[b]
    annotations = df["Ticker"]
    plt.grid(True)
    plt.title(a+"-------"+b, fontsize=18, color='black')
    plt.xlabel(a, fontsize=18, color='black')
    plt.ylabel(b, fontsize=18, color='black')
    plt.scatter(x, y, s=200)
    for i, label in enumerate(annotations):
        plt.annotate(label, (x[i], y[i]))
    plt.scatter(x, y, s=300)

    plt.axvline(x=t1, color='r', label='axvline - full height')
    plt.axhline(y=t2, color='r', linestyle='-')
    plt.show()

    fig = plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)

def port_opt(df1,start_date,end_date):
    pf_data = pd.DataFrame()
    for t in df1["Ticker"]:
        pf_data[t] = yf.download(t, start=start_date, end=end_date)['Adj Close']

    log_returns = np.log(pf_data / pf_data.shift(1))
    num_assets = len(df1["Ticker"])
    num_assets
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    np.sum(weights * log_returns.mean()) * 252
    np.dot(weights.T, np.dot(log_returns.cov() * 252, weights))
    np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
    pfolio_returns = []
    pfolio_volatilities = []
    wts = []
    for x in range(1000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        pfolio_returns.append(np.sum(weights * log_returns.mean()) * 252)
        pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights))))
        wts.append(weights)
    pfolio_returns = np.array(pfolio_returns)
    pfolio_volatilities = np.array(pfolio_volatilities)
    wts = np.array(wts)
    # pfolio_returns, pfolio_volatilities
    portfolios = pd.DataFrame({'Return': pfolio_returns, 'Volatility': pfolio_volatilities})
    portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(15, 10));
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    fig = plt.show()
    st.pyplot(fig)
    df = pd.DataFrame(data=wts)
    df.columns = df1["Ticker"]
    df_col_merged = pd.concat([portfolios, df], axis=1)
    dft = df_col_merged.loc[df_col_merged['Return'].argmax()]
    dft.columns = ["Portfolio","% to be Invested"]
    
    st.dataframe(dft*100, 900, 600)


def daily_return(df):
    df_daily_return = df.copy()

    # Loop through each stock (while ignoring time columns with index 0)
    for i in df.columns[1:]:
        # Loop through each row belonging to the stock
        for j in range(1, len(df)):
            # Calculate the percentage of change from the previous day
            df_daily_return[i][j] = ((df[i][j] - df[i][j - 1]) / df[i][j - 1]) * 100

    # set the value of first row to zero since the previous value is not available
    df_daily_return[i][0] = 0

    return df_daily_return

def getdata():
 
    with open('datasets/NIFTY1.csv') as f:
        companies = f.read().splitlines()

        for company in companies:
            symbol = company.split(',')[1]+".NS"

            df = yf.download(symbol, period="4y")  
            df.to_csv('datasets/daily/{}.csv'.format(symbol))   




def candle_trend(patterns,tickers):
    
    data = pd.DataFrame(columns=['Ticker','Candle-Pattern','Definition','Signal'])
    for t in tickers:
        t = t.upper()
        t.strip()
        t= t+".NS"
        df = yf.download(t, period="4y") 
        for pattern in patterns:
            ta_function = getattr(talib, pattern)
            result = ta_function(df['Open'], df['High'], df['Low'], df['Close'])
            last_pattern = result.tail(1).values[0]
            if last_pattern > 0:
                data = data.append(pd.Series([t,pattern,patterns[pattern],"BULLISH"],index=data.columns), ignore_index=True)
                
                
            elif last_pattern < 0:
                data = data.append(pd.Series([t,pattern,patterns[pattern],"BEAR"],index=data.columns), ignore_index=True)
          

    return data


filename = st.text_input('Enter a file path:')

try:
    with open(filename) as input:
        tic = input.read()
except FileNotFoundError:
    st.error('File not found.')



symbols, start_date, end_date = get_input()
tickers = tic.split("\n")

df,data = cal_data(tickers, start_date, end_date)


st.subheader("Ticker Data Comparison:")
st.dataframe(df,900,300)
df1 = df[(df['CAGR']>10 ) & (df['Return%']>10)]
df1 = df1.sort_values(by='Return%', ascending=False)
df1 =df1.reset_index()
st.subheader("Sorted Data Based on Returns:")
st.dataframe(df1,900,300)
st.subheader("Candle Stick Patterns:")
p1 = candle_trend(patterns,tickers)
st.dataframe(p1,300,300)
plot_data(df1,'Return%','Value at Risk',50,50000)

plot_data(df1,'Return%','Volatility%',50,50)

plot_data(df1,'Volatility%','Value at Risk',50,50000)

plot_data(df1,'Return%','CV',50,100)

st.subheader("Markowitz Portfolio Optimization:")

port_opt(df1,start_date,end_date)
data = data.reset_index()
stocks_daily_return = daily_return(data)
st.subheader("Heatmap of the Stocks:")
cm = stocks_daily_return.drop(columns = ['Date']).corr()
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax);
st.pyplot(fig)

df2 = df1[(df1['Volatility%']<53 ) & (df1['Value at Risk']<=50000)]
df2 = df2.sort_values(by='CAGR', ascending=False)
df2 = df2.reset_index()

plot_data(df2,'Return%','Value at Risk',50,50000)

plot_data(df2,'Return%','Volatility%',50,50)

plot_data(df2,'Volatility%','Value at Risk',50,50000)

plot_data(df2,'Return%','CV',50,100)


st.subheader("Candle Stick Pattern:")
d = candle_trend(patterns,tickers)
st.dataframe(d,300,500)