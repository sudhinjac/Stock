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
#import talib
import datetime
import pandas as pd
import yfinance as yf

from arch import arch_model
#from arch.__future__ import reindexing
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

import pandas_datareader
import datetime
import pandas_datareader.data as web
from scipy.stats import norm




st.title("Welcome To Stock Ticker Analyzer")




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



ticker = st.text_input('Enter Ticker EX:"HCLTECH.NS"',value="AARTIIND.NS")
bse = '^BSESN'
tickers = [ticker,bse]

def cal_data(tickers):
    data = pd.DataFrame()
    for t in tickers:
        t = t.upper()
        data[t] = yf.download(t, period='4y')['Adj Close']
    sec_returns = np.log(data / data.shift(1))
    cov = sec_returns.cov() * 250
    cov_with_market = cov.iloc[0, 1]
    market_var = sec_returns[bse].var() * 250
    MSFT_beta = cov_with_market / market_var
    st.write("Beta is : {}".format(MSFT_beta))
    returns = np.log(data / data.shift(1))

    vols = returns.std() * 250 ** 0.5 * 100
    st.write("Volatility is : {}".format(vols))
    annual_returns = returns.mean() * 250 * 100
    st.write("Return is : {}".format(annual_returns))
    sharpe_ratio = annual_returns[ticker] / vols[ticker]
    st.write("Shape Ratio : {}".format(sharpe_ratio))
    CV = (vols / annual_returns) * 100
    st.write("CV : {}".format(CV))



def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod()
    n = len(df) / 252
    CAGR = (df["cum_return"][-1]) ** (1 / n) - 1
    return CAGR


def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    vol = df["daily_ret"].std() * np.sqrt(252)
    return vol


def sharpe(DF, rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf) / volatility(df)
    return sr


def sortino(DF, rf):
    "function to calculate sortino ratio ; rf is the risk free rate"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    neg_vol = df[df["daily_ret"] < 0]["daily_ret"].std() * np.sqrt(252)
    sr = (CAGR(df) - rf) / neg_vol
    return sr


def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


def calmar(DF):
    "function to calculate calmar ratio"
    df = DF.copy()
    clmr = CAGR(df) / max_dd(df)
    return clmr

def signals(dfnew):
    dfnew['50DMA'] = dfnew['Close'].rolling(window=50).mean()
    dfnew['200DMA'] = dfnew['Close'].rolling(window=200).mean()
    dfnew['crit1'] = dfnew['Close'] >= dfnew['200DMA']
    dfnew['cr12'] = (dfnew['50DMA'] >= dfnew['200DMA']) | dfnew['crit1'] == True
    st.dataframe(dfnew.tail(100),800,800)


df = pd.DataFrame()
df = yf.download(ticker, period='4y')



cal_data(tickers)

cg = CAGR(df)
vola = volatility(df)
shar = sharpe(df,0.06)
sor = sortino(df,0.06)
md = max_dd(df)
calm = calmar(df)

st.write("\n The Compunded Annual Groth of the stock is {}%".format(cg*100))
st.write("\n The Volatility of the stock is  {}%".format(vola*100))
st.write("\n The Sharp Ratio is: {}".format(shar))
st.write("\n The Sortino Ratio is: {}".format(sor))
st.write("\n The MAX Draw Down it shows loss for the investment {}%".format(md*100))
st.write("\n The Calamr Ratio is(It shows the Risk Adjusted Return) : {}".format(calm))

st.header("Trading Signals")
signals(df)
def monte_carlo(ticker):
    data = pd.DataFrame()
    data[ticker] = yf.download(ticker, period='4y')['Adj Close']
    log_returns = np.log(1 + data.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()



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
    a = price_list.max()
    a
    st.header("The MINIMUM PRICE OBTAINED FROM MONTE CARLO SIMULATION IS:")

    b = price_list.min()
    b
    st.header("The MEAN PRICE OBTAINED FROM MONTE CARLO SIMULATION IS:")
    c = price_list.mean()
    c
    st.write("Expected price: ", np.mean(price_list))
    st.write("Quantile (5%): ", np.percentile(price_list, 5))
    st.write("Quantile (95%): ", np.percentile(price_list, 95))

    plt.axvline(np.percentile(price_list,5), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(np.percentile(price_list,95), color='r', linestyle='dashed', linewidth=2)
    fig, ax = plt.subplots()
    ax.hist(price_list,bins=100)
    st.pyplot(fig)






monte_carlo(ticker)

def balance_sheet(ticker):
    from datetime import datetime
    import lxml
    from lxml import html
    import requests
    import numpy as np
    import pandas as pd

    symbol = ticker

    url = 'https://finance.yahoo.com/quote/' + symbol + '/balance-sheet?p=' + symbol

    # Set up the request headers that we're going to use, to simulate
    # a request by the Chrome browser. Simulating a request from a browser
    # is generally good practice when building a scraper
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'Connection': 'close',
        'DNT': '1',  # Do Not Track Request Header
        'Pragma': 'no-cache',
        'Referrer': 'https://google.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    }

    # Fetch the page that we're going to parse, using the request headers
    # defined above
    page = requests.get(url, headers=headers)

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

    parsed_rows = []

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

    df = pd.DataFrame(parsed_rows)
    st.dataframe(df,800,800)

def financials(ticker):
    from datetime import datetime
    import lxml
    from lxml import html
    import requests
    import numpy as np
    import pandas as pd

    symbol = ticker

    url = 'https://finance.yahoo.com/quote/' + symbol + '/financials?p=' + symbol

    # Set up the request headers that we're going to use, to simulate
    # a request by the Chrome browser. Simulating a request from a browser
    # is generally good practice when building a scraper
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'Connection': 'close',
        'DNT': '1',  # Do Not Track Request Header
        'Pragma': 'no-cache',
        'Referrer': 'https://google.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    }

    # Fetch the page that we're going to parse, using the request headers
    # defined above
    page = requests.get(url, headers=headers)

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

    parsed_rows = []

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

    df = pd.DataFrame(parsed_rows)
    st.dataframe(df,800,800)

def cash_flow(ticker):
    from datetime import datetime
    import lxml
    from lxml import html
    import requests
    import numpy as np
    import pandas as pd

    symbol = ticker

    url = 'https://finance.yahoo.com/quote/' + symbol + '/cash-flow?p=' + symbol

    # Set up the request headers that we're going to use, to simulate
    # a request by the Chrome browser. Simulating a request from a browser
    # is generally good practice when building a scraper
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'Connection': 'close',
        'DNT': '1',  # Do Not Track Request Header
        'Pragma': 'no-cache',
        'Referrer': 'https://google.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    }

    # Fetch the page that we're going to parse, using the request headers
    # defined above
    page = requests.get(url, headers=headers)

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

    parsed_rows = []

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

    df = pd.DataFrame(parsed_rows)
    st.dataframe(df,800,800)

def key_fianancials(ticker):
    import requests
    from bs4 import BeautifulSoup

    import requests
    from bs4 import BeautifulSoup

    url = "https://finance.yahoo.com/quote/"+ticker+ "/key-statistics?p="+ticker

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
    }

    soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")

    for t in soup.select("table"):
        for tr in t.select("tr:has(td)"):
            for sup in tr.select("sup"):
                sup.extract()
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if len(tds) == 2:
                st.write("{:<50} {}".format(*tds))
st.header("Balance Sheet")
balance_sheet(ticker)
st.header("Key Financials")
financials(ticker)
st.header("Cash Flow")
cash_flow(ticker)
st.header("Key Indicators")
key_fianancials(ticker)






fig = go.Figure(
    data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )]
)

st.subheader("Historical Prices")
st.write(df.head(15))

st.subheader("Data Statistics")
st.write(df.describe())

st.subheader("Historical Price Chart - Adjusted Close Price")
st.line_chart(df['Adj Close'])

st.subheader("Volume")
st.bar_chart(df['Volume'])

st.subheader("Candlestick Trend")
#d1 = candle_trend(df, patterns)
#st.dataframe(d1)

st.subheader("Candlestick Chart")
st.plotly_chart(fig)

def garch(stock_data):
    import math
    stock_data['Return'] = 100 * (stock_data['Close'].pct_change())
    stock_data.dropna(inplace=True)

    fig = plt.figure()
    fig.set_figwidth(12)
    plt.plot(stock_data['Return'], label='Daily Returns')
    plt.legend(loc='upper right')
    plt.title('Daily Returns Over Time')
    plt.show()
    st.pyplot(fig)
    daily_volatility = stock_data['Return'].std()
    st.write('Daily volatility: ', '{:.2f}%'.format(daily_volatility))

    monthly_volatility = math.sqrt(21) * daily_volatility
    st.write('Monthly volatility: ', '{:.2f}%'.format(monthly_volatility))

    annual_volatility = math.sqrt(252) * daily_volatility
    st.write('Annual volatility: ', '{:.2f}%'.format(annual_volatility))

    garch_model = arch_model(stock_data['Return'], p=1, q=1,
                             mean='constant', vol='GARCH', dist='normal')

    gm_result = garch_model.fit(disp='off')
    st.dataframe(gm_result.params)

    print('\n')

    gm_forecast = gm_result.forecast(horizon=5)
    st.dataframe(gm_forecast.variance[-1:])

    rolling_predictions = []
    test_size = 365

    for i in range(test_size):
        train = stock_data['Return'][:-(test_size - i)]
        model = arch_model(train, p=1, q=1)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

    rolling_predictions = pd.Series(rolling_predictions, index=stock_data['Return'].index[-365:])

    plt.figure(figsize=(10, 4))
    plt.plot(rolling_predictions)
    plt.title('Rolling Prediction')
    plt.show()
    st.line_chart(rolling_predictions)


garch(df)


df['EWMA12'] = df['Close'].ewm(span=12,adjust=False).mean()
df['12-month-SMA'] = df['Close'].rolling(window=12).mean()
st.line_chart(df[['Close','EWMA12','12-month-SMA']])
