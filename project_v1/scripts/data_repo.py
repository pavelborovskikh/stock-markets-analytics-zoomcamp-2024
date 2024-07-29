import numpy as np

import pandas as pd
import yfinance as yf

from tqdm import tqdm
from datetime import datetime, timedelta

import time
import os
import pandas_datareader as pdr

# US_STOCKS
# https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/
US_STOCKS = ['MSFT', 'AAPL', 'GOOG', 'NVDA', 'AMZN', 'META', 'BRK-B', 'LLY', 'TSLA', 'AVGO', 'JPM']
# EU_STOCKS
# https://companiesmarketcap.com/european-union/largest-companies-in-the-eu-by-market-cap/
EU_STOCKS = ['NVO', 'MC.PA', 'ASML', 'RMS.PA', 'OR.PA', 'ACN', 'TTE', 'PRX.AS', 'IDEXY', 'SU.PA', 'ETN']
# DE_STOCKS
# https://companiesmarketcap.com/germany/largest-companies-in-germany-by-market-cap/
DE_STOCKS = ["SAP", "SIE.DE", "DTE.DE", "ALV.DE", "P911.DE", "MBG.DE", "MRK.DE", "MUV2.DE", "SHL.DE", "BMW.DE", "VOW3.DE"]



class DataRepository:
  ticker_df: pd.DataFrame
  indexes_df: pd.DataFrame
  macro_df: pd.DataFrame

  min_date: str
  ALL_TICKERS: list[str] = US_STOCKS  + EU_STOCKS + DE_STOCKS

  def __init__(self):
    self.ticker_df = None
    self.indexes_df = None
    self.macro_df = None

  def _get_growth_df(self, df:pd.DataFrame, prefix:str)->pd.DataFrame:
    '''Help function to produce a df with growth columns'''
    for i in [1,3,7,30,90,365]:
      df['growth_'+prefix+'_'+str(i)+'d'] = df['Adj Close'] / df['Adj Close'].shift(i)
      GROWTH_KEYS = [k for k in df.keys() if k.startswith('growth')]
    return df[GROWTH_KEYS]
    
  def fetch(self, min_date = None):
    '''Fetch all data from APIs'''

    print('Fetching Tickers info from YFinance')
    self.fetch_tickers(min_date=min_date)
    print('Fetching Indexes info from YFinance')
    self.fetch_indexes(min_date=min_date)
    print('Fetching Macro info from FRED (Pandas_datareader)')
    self.fetch_macro(min_date=min_date)
  
  def fetch_tickers(self, min_date=None):
    '''Fetch Tickers data from the Yfinance API'''
    if min_date is None:
      min_date = "1970-01-01"
    else:
      min_date = pd.to_datetime(min_date)   

    print(f'Going download data for this tickers: {self.ALL_TICKERS[0:3]}')
    tq = tqdm(self.ALL_TICKERS)
    for i,ticker in enumerate(tq):
      tq.set_description(ticker)

      # Download stock prices from YFinance
      historyPrices = yf.download(tickers = ticker,
                          # period = "max",
                          start=min_date,
                          interval = "1d")

      # generate features for historical prices, and what we want to predict

      if ticker in US_STOCKS:
        historyPrices['ticker_type'] = 'US'
      elif ticker in EU_STOCKS:
        historyPrices['ticker_type'] = 'EU'
      elif ticker in DE_STOCKS:
        historyPrices['ticker_type'] = 'DE'
      else:
        historyPrices['ticker_type'] = 'ERROR'

      historyPrices['Ticker'] = ticker
      historyPrices['Year']= historyPrices.index.year
      historyPrices['Month'] = historyPrices.index.month
      historyPrices['Weekday'] = historyPrices.index.weekday
      historyPrices['Date'] = historyPrices.index.date

      # historical returns
      for i in [1,3,7,30,90,365]:
          historyPrices['growth_'+str(i)+'d'] = historyPrices['Adj Close'] / historyPrices['Adj Close'].shift(i)
      historyPrices['growth_future_3d'] = historyPrices['Adj Close'].shift(-3) / historyPrices['Adj Close']

      # Technical indicators
      # SimpleMovingAverage 10 days and 20 days
      historyPrices['SMA10']= historyPrices['Close'].rolling(10).mean()
      historyPrices['SMA20']= historyPrices['Close'].rolling(20).mean()
      historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
      historyPrices['high_minus_low_relative'] = (historyPrices.High - historyPrices.Low) / historyPrices['Adj Close']

      # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
      historyPrices['volatility'] =   historyPrices['Adj Close'].rolling(30).std() * np.sqrt(252)

      # what we want to predict
      historyPrices['is_positive_growth_3d_future'] = np.where(historyPrices['growth_future_3d'] > 1, 1, 0)

      # sleep 1 sec between downloads - not to overload the API server
      time.sleep(1)

      if self.ticker_df is None:
        self.ticker_df = historyPrices
      else:
        self.ticker_df = pd.concat([self.ticker_df, historyPrices], ignore_index=True)
      
  def fetch_indexes(self, min_date=None):
    '''Fetch Indexes data from the Yfinance API'''

    if min_date is None:
      min_date = "1970-01-01"
    else:
      min_date = pd.to_datetime(min_date)   
    
    # https://finance.yahoo.com/quote/%5EGDAXI/
    # DAX PERFORMANCE-INDEX
    dax_daily = yf.download(tickers = "^GDAXI",
                        start = min_date,    
                        # period = "max",
                        interval = "1d")
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)
    
    # https://finance.yahoo.com/quote/%5EGSPC/
    # SNP - SNP Real Time Price. Currency in USD
    snp500_daily = yf.download(tickers = "^GSPC",
                     start = min_date,          
                    #  period = "max",
                     interval = "1d")
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)
    
    # https://finance.yahoo.com/quote/%5EDJI?.tsrc=fin-srch
    # Dow Jones Industrial Average
    dji_daily = yf.download(tickers = "^DJI",
                     start = min_date,       
                    #  period = "max",
                     interval = "1d")
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)
    
    # GOLD
    # https://finance.yahoo.com/quote/GC%3DF
    gold = yf.download(tickers = "GC=F",
                     start = min_date,   
                    #  period = "max",
                     interval = "1d")
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)
    
    # WTI Crude Oil
    # https://uk.finance.yahoo.com/quote/CL=F/
    crude_oil = yf.download(tickers = "CL=F",
                     start = min_date,          
                    #  period = "max",
                     interval = "1d")
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)

    # Brent Oil
    # WEB: https://uk.finance.yahoo.com/quote/BZ=F/
    brent_oil = yf.download(tickers = "BZ=F",
                            start = min_date,
                            # period = "max",
                            interval = "1d")
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)


    # BTC_USD
    # WEB: https://finance.yahoo.com/quote/BTC-USD/
    btc_usd =  yf.download(tickers = "BTC-USD",
                           start = min_date,
                          #  period = "max",
                           interval = "1d")
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)

    # VIX - Volatility Index
    # https://finance.yahoo.com/quote/%5EVIX/
    vix = yf.download(tickers = "^VIX",
                        start = min_date,
                        # period = "max",
                        interval = "1d")
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)
    
    # Prepare to merge
    dax_daily_to_merge = self._get_growth_df(dax_daily, 'dax')
    snp500_daily_to_merge = self._get_growth_df(snp500_daily, 'snp500')
    dji_daily_to_merge = self._get_growth_df(dji_daily, 'dji')
    gold_to_merge = self._get_growth_df(gold, 'gold')
    crude_oil_to_merge = self._get_growth_df(crude_oil,'wti_oil')
    brent_oil_to_merge = self._get_growth_df(brent_oil,'brent_oil')
    btc_usd_to_merge = self._get_growth_df(btc_usd,'btc_usd')
    vix_to_merge = vix.rename(columns={'Adj Close': 'vix_adj_close'})[['vix_adj_close']]

    # Merging
    m2 = pd.merge(snp500_daily_to_merge,
                               dax_daily_to_merge,
                               left_index=True,
                               right_index=True,
                               how='left',
                               validate='one_to_one')
    
    m3 = pd.merge(m2,
                  dji_daily_to_merge,
                  left_index=True,
                  right_index=True,
                  how='left',
                  validate='one_to_one')
    
    m4 = pd.merge(m3,
                  gold_to_merge,
                  left_index=True,
                  right_index=True,
                  how='left',
                  validate='one_to_one')
    
    m5 = pd.merge(m4,
                  crude_oil_to_merge,
                  left_index=True,
                  right_index=True,
                  how='left',
                  validate='one_to_one')    

    m6 = pd.merge(m5,
                  brent_oil_to_merge,
                  left_index=True,
                  right_index=True,
                  how='left',
                  validate='one_to_one')  

    m7 = pd.merge(m6,
                  btc_usd_to_merge,
                  left_index=True,
                  right_index=True,
                  how='left',
                  validate='one_to_one')
    m8 = pd.merge(m7,
                  vix_to_merge,
                  left_index=True,
                  right_index=True,
                  how='left',
                  validate='one_to_one')

    self.indexes_df = m8

  def fetch_macro(self, min_date=None):
    '''Fetch Macro data from FRED (using Pandas datareader)'''

    if min_date is None:
      min_date = "1970-01-01"
    else:
      min_date = pd.to_datetime(min_date)

    # get the quarterly GDP data for Germany from Fred (https://fred.stlouisfed.org/series/NGDPRSAXDCDEQ)
    gdp = pdr.DataReader("NGDPRSAXDCDEQ", "fred", start=min_date)
    gdp['gdp_de_yoy'] = gdp.NGDPRSAXDCDEQ / gdp.NGDPRSAXDCDEQ.shift(4) - 1
    gdp['gdp_de_qoq'] = gdp.NGDPRSAXDCDEQ / gdp.NGDPRSAXDCDEQ.shift(1) - 1
    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(1)

    # CPI, monthly (https://fred.stlouisfed.org/series/DEUCPIALLMINMEI)
    cpi = pdr.DataReader("DEUCPIALLMINMEI", "fred", start=min_date)
    cpi['cpi_de_yoy'] = cpi.DEUCPIALLMINMEI / cpi.DEUCPIALLMINMEI.shift(12) - 1
    cpi['cpi_de_mom'] = cpi.DEUCPIALLMINMEI / cpi.DEUCPIALLMINMEI.shift(1) - 1
    time.sleep(1)

    # Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for Germany (https://fred.stlouisfed.org/series/IRLTLT01DEM156N)
    bond10 = pdr.DataReader("IRLTLT01DEM156N", "fred", start=min_date).rename(columns={'IRLTLT01DEM156N':'bond10'})
    time.sleep(1)

    gdp_to_merge = gdp[['gdp_de_yoy', 'gdp_de_qoq']]
    cpi_to_merge = cpi[['cpi_de_yoy', 'cpi_de_mom']]


    # Merging - start from monthly stats

    #bond10['Date'] = bond10.index
    #bond10['Month'] = bond10.Date.dt.to_period('M').dt.to_timestamp()
    #cpi_to_merge['Date'] = cpi_to_merge.index
    #cpi_to_merge['Month'] = cpi_to_merge.Date.dt.to_period('M').dt.to_timestamp()

    m2 = pd.merge(bond10,
                  cpi_to_merge,
                  left_index=True,
                  right_index=True,
                  how='left',
                  validate='one_to_one')
    
    m2['Date'] = m2.index

    # gdp_to_merge is Quarterly (but m2 index is monthly)
    m2['Quarter'] = m2.Date.dt.to_period('Q').dt.to_timestamp()

    m3 = pd.merge(m2,
                  gdp_to_merge,
                  left_on='Quarter',
                  right_index=True,
                  how='left',
                  validate='many_to_one')
    
    fields_to_fill = ['cpi_de_yoy',	'cpi_de_mom', "gdp_de_yoy", "gdp_de_qoq", "bond10"]
    # Fill missing values in selected fields with the last defined value
    for field in fields_to_fill:
      m3[field] = m3[field].ffill()

    self.macro_df = m3

  def persist(self, data_dir:str):
    '''Save dataframes to files in a local directory 'dir' '''
    os.makedirs(data_dir, exist_ok=True)

    file_name = 'tickers_df.parquet'
    if os.path.exists(file_name):
      os.remove(file_name)
    self.ticker_df.to_parquet(os.path.join(data_dir,file_name), compression='brotli')
  
    file_name = 'indexes_df.parquet'
    if os.path.exists(file_name):
      os.remove(file_name)
    self.indexes_df.to_parquet(os.path.join(data_dir,file_name), compression='brotli')
  
    file_name = 'macro_df.parquet'
    if os.path.exists(file_name):
      os.remove(file_name)
    self.macro_df.to_parquet(os.path.join(data_dir,file_name), compression='brotli')

  def load(self, data_dir:str):
    """Load files from the local directory"""
    self.ticker_df = pd.read_parquet(os.path.join(data_dir,'tickers_df.parquet'))
    self.macro_df = pd.read_parquet(os.path.join(data_dir,'macro_df.parquet'))
    self.indexes_df = pd.read_parquet(os.path.join(data_dir,'indexes_df.parquet'))