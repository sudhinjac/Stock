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



     

dfticker = pd.read_csv("C:\\STOCK\\LATSTOCK.csv",index_col=False,encoding= 'unicode_escape')

dfticker['Security Name'] = dfticker['Security Name'].str.upper()
#MSFT_beta


st.title("Finding out ticker by company name search")
symb = st.text_input("Enter company to search")
symb = symb.upper()
symb
df2 = dfticker[dfticker['Security Name'].str.contains(symb)]
st.dataframe(df2,900,900)
#st.markdown("This application is a Share Price Analysis")
#st.sidebar.markdown("This application is a Share Price Anaalysis")

#st.dataframe(dfticker,900,900)

