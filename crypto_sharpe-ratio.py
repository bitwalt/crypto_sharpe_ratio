import streamlit as st  
import requests

URL = "https://min-api.cryptocompare.com/data/top/coinsort?limit=100"

import time
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
#plt.style.use('ggplot')

stocks = ['BTC-USD', 'ETH-USD', 'LTC-USD','XRP-USD']


def get_coinmarketcap_data():
    #get daily prices from coinmarketcap of the top 100 coins by market cap
    coinmarketcap_url = "https://api.coinmarketcap.com/v1/ticker/?limit=100"
    r = requests.get(coinmarketcap_url)
    coinmarketcap_data = r.json()
    # parse coinmarket cap response
    coinmarketcap_df = pd.DataFrame(coinmarketcap_data)
    coinmarketcap_df = coinmarketcap_df.set_index(['rank'])
    coinmarketcap_df.index.names = ['rank']
    coinmarketcap_df = coinmarketcap_df.transpose()
    coinmarketcap_df = coinmarketcap_df.reset_index()
    coinmarketcap_df.columns = ['rank', 'symbol', 'name', 'price_usd', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d']
    return coinmarketcap_df

@st.cache
def get_yahoo_data(tickers, start_date, end_date):
    """Read in daily price(adjusted close) of asset from CSV files for a given set of dates."""
    # download daily price data for each of the stocks in the portfolio
    df = pdr.get_data_yahoo(stocks, start=start_date, end = end_date)['Adj Close']
    df.sort_index(inplace = True)
    return df

def calc_daily_returns(df):
    return (df.pct_change())

def calc_mean_daily_returns(daily_returns):
    return (daily_returns.mean())

def create_covariance_matrix(daily_returns):
    return daily_returns.cov()

def min_volatility(results_df):
    """locate portfolio with lowest volatility"""
    return results_df.iloc[results_df['stdev'].idxmin()]

@st.cache
def create_results_dataframe(tickers, number_portfolios, mean_daily_returns, cov_matrix):
    results_temp = np.zeros((4 + len(tickers) - 1, number_portfolios))

    for i in range(number_portfolios):
        # select random weights for portfolio holdings
        weights = np.array(np.random.random(4))
        # rebalance weights to sum to 1
        weights /= np.sum(weights)
        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        # 3 month T-Bill yield used as risk free rate of return
        risk_free_return = 0.0139
        # store results in results array
        results_temp[0, i] = portfolio_return
        results_temp[1, i] = portfolio_std_dev
        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results_temp[2, i] = (results_temp[0, i] - risk_free_return) / results_temp[1, i]
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_temp[j + 3, i] = weights[j]
    # convert results array to Pandas DataFrame
    results_df = pd.DataFrame(results_temp.T, columns=['ret', 'stdev', 'sharpe', tickers[0], 
                                                       tickers[1], tickers[2], tickers[3]])
    
    return results_df


def max_sharpe_ratio(results_df):
    """locate portfolio with highest Sharpe Ratio"""
    return results_df.iloc[results_df['sharpe'].idxmax()]

def plot_graph(results_df, max_sharpe_port, min_vol_port):
    ax = results_df.plot(kind= 'scatter', x = 'stdev', y='ret', s = 30, 
                         c=results_df.sharpe, cmap='RdYlBu',edgecolors='.1', figsize=(20,10))
    ax.grid(False, color='w', linestyle='-', linewidth=1)
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Returns')
    ax.tick_params(labelsize = 14)

    # # plot red star to highlight position of portfolio with highest Sharpe Ratio
    ax.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=1000)
    # # plot green star to highlight position of minimum variance portfolio
    ax.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=1000)
    ax.figure.savefig('sharpe.png')
    return ax.get_figure()

    
def main():
    st.title("Cryto Detector")
    
    # coinmarketcap_df = get_coinmarketcap_data()
    # st.write("Retrieving data from CoinMarketCap...")
    # st.write(coinmarketcap_df)
    crypto_prices = get_yahoo_data(stocks, '01/01/2019', '22/08/2021' )
    st.write("Crypto Prices")
    st.write(crypto_prices)
    crypto_daily_rets = calc_daily_returns(crypto_prices)
    st.write("Crypto Daily Returns")
    st.write(crypto_daily_rets)
    
    crypto_mean_daily_rets = calc_mean_daily_returns(crypto_daily_rets)
    crypto_cov_matrix = create_covariance_matrix(crypto_daily_rets)
    crypto_results = create_results_dataframe(stocks, 10000, crypto_mean_daily_rets, crypto_cov_matrix)
    st.write("Crypto Results")
    st.write(crypto_results)
    crypto_max_sharpe_portfolio = max_sharpe_ratio(crypto_results)
    st.write("Crypto Max Sharpe Ratio")
    st.write(crypto_max_sharpe_portfolio)

    crypto_min_vol_portfolio = min_volatility(crypto_results)
    st.write("Crypto Min Volatility")
    st.write(crypto_min_vol_portfolio)

    st.write("Crypto Plot")
    fig = plot_graph(crypto_results, crypto_max_sharpe_portfolio, crypto_min_vol_portfolio)
    st.pyplot(fig)
    
if __name__ == "__main__":
    main()
