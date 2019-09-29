# Zerodha Imports
from kiteconnect import KiteConnect

# standard python imports 
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import logging

# PostgresDB connection 
import psycopg2
from sqlalchemy import create_engine

# Relative imports
from  postgresql_conn import Database
logger = None
db_obj = Database(logger)

# Kite Connect keys
myApiKey="u3mlhhy976n8i7q5"
api_secret="qlwgh9fyrahjrl39qauxbnylrlzkx1gd"
publicToken="c31b29bd22fa6c181388a0c2a133c8c2"
request_token="a2d3emw6l2pfisgxmyi8j7vxvb6z2nty"

kite = KiteConnect(api_key=myApiKey)
data = kite.generate_session(request_token, api_secret=api_secret)
kite.set_access_token(data["access_token"])
kite.set_access_token(access_token)

def place_stock_orders(trading_symbol, qty):
    # Place an order
    try:
        order_id = kite.place_order(tradingsymbol="INFY",
                                    exchange=kite.EXCHANGE_NSE,
                                    transaction_type=kite.TRANSACTION_TYPE_BUY,
                                    quantity=qty,
                                    order_type=kite.ORDER_TYPE_MARKET,
                                    product=kite.PRODUCT_NRML
                                    )

        logging.info("Order placed. ID is: {}".format(order_id))
    except Exception as e:
        logging.info("Order placement failed: {}".format(e.message))


def retrieve_order_info():
    pass


def postgre_sql_read_df(query):
    """Creates postresql_conn object and closes connection immidiately after usage.
    This will help tackle connection pool exceed issue.
    
    Arguments:
        query {[String]} -- [SQL query to read data from postgresql and 
                            create a pandas dataframe]

    """
    conn_string_alchemy = "postgresql://whitedwarf:#finre123#@finre.cgk4260wbdoi.ap-south-1.rds.amazonaws.com/finre"
    engine = create_engine(conn_string_alchemy)
    df = pd.read_sql_query(query, con=engine)
    engine.dispose()
    return df


def get_index_nifty500_data(tuesday_date_string):
    global flag_buy
    # retrieve data for nifty500
    nifty500_jas_token = '10006'

    # Nifty500 data fetch
    query_nse_nifty500 = "SELECT * FROM nse_data WHERE jas_token={0} and timestamp<={1}".format(nifty500_jas_token,tuesday_date_string)
    df_nifty_500 = postgre_sql_read_df(query_nse_nifty500)
    df_nifty_500 = df_nifty_500.iloc[-200:,:]
    mv_100_close_nifty_500 = df_nifty_500['close'].mean()
    current_close_nifty_500 = df_nifty_500['close'].iloc[-1]

    if current_close_nifty_500 <= mv_100_close_nifty_500:
        flag_buy = False
    else:
        flag_buy = True

    print("Flag Buy..............."+str(flag_buy))
    return flag_buy

def get_user_momentum_portfolio_data():
    # User Momentum Portfolio Data
    query_user_momentum_portfolio = "SELECT * FROM user_momentum_portfolio ORDER BY timestamp DESC LIMIT 1 IN (SELECT DISTINCT user_id from user_momentum_portfolio)"
    df_user_momentum_portfolio = postgre_sql_read_df(query_user_momentum_portfolio)
    df_user_momentum_portfolio = df_user_momentum_portfolio['user_id'].drop_duplicates(keep='last')
    

def main_func():
    todays_date = datetime.now()
    tuesday_date_string = latest_tuesday_func(todays_date)
    # 1st check: If market is more than 200 DMA then buy 
    flag_buy = get_index_nifty500_data(tuesday_date_string)
