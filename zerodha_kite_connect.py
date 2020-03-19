# Zerodha Imports
from kiteconnect import KiteConnect

# standard python imports 
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import math
import logging

# PostgresDB connection 
import psycopg2
from sqlalchemy import create_engine

# Relative imports
from  postgresql_conn import Database

error_log_data_name = '../error_files_log/zerodha_data_{0}.log'.format(datetime.now().strftime("%Y_%m_%d"))

logging.basicConfig(filename = error_log_data_name, filemode = 'w')
conn_string = "host='localhost' dbname='jas' user='postgres' password='sau651994'"
db_obj = Database(logging, url=conn_string)
db_obj.connect()

# Kite Connect keys
myApiKey = "0vqaranvq9k56am0"
api_secret = "pehnl80uvmvovido6mjdhcxudxvte6ze"
access_token = "K1pEjHfy2bBCwHKY4GVFrcd3PAXPryrp"
# request_token = "ga94ubDv9OISYgYM37eWcSjm5XFSNi7U"

# publicToken = "c31b29bd22fa6c181388a0c2a133c8c2"

### ---- with apoorv key ---
# kite = KiteConnect(api_key=myApiKey)
# # access_token,public_token,request_token,api_key,api_secret = at.getAll()
# data = kite.generate_session(request_token, api_secret=api_secret)
# kite.set_access_token(data["access_token"])
# kite.set_access_token(access_token)

###----- temp approach -----
kite = KiteConnect(api_key=myApiKey)
print(kite.login_url())
# data = kite.generate_session(request_token, api_secret=api_secret)
# print("Access Token ------------ ",data["access_token"])
kite.set_access_token(access_token)
print("Connection successfull ......")

def postgre_sql_read_df(query):
    """Creates postresql_conn object and closes connection immidiately after usage.
    This will help tackle connection pool exceed issue.
    
    Arguments:
        query {[String]} -- [SQL query to read data from postgresql and 
                            create a pandas dataframe]

    """
    # conn_string_alchemy = "postgresql://whitedwarf:#finre123#@finre.cgk4260wbdoi.ap-south-1.rds.amazonaws.com/finre"
    conn_string_pd = 'postgresql://postgres:sau651994@localhost:5432/jas'

    engine = create_engine(conn_string_pd)
    df = pd.read_sql_query(query, con=engine)
    engine.dispose()
    return df

def historical_date_to_date(star_date, end_date, jas_mapping_df=None):
    """Historical and daily zerodha Data Fetch Function.
    For Historical Data Fetch provide start_date_string and end_date_string
    not more than 250 days.
    For daily data Fetch pass todays start date and end date.
    It updates data for Nifty 500 Stocks.
    Nifty 500 Stocks list is organised according to FrTokens.
        
    Arguments:
        star_date {[String]} -- [Start date string eg. '2019-09-28']
        end_date {[String]} --  [End date string eg. '2019-09-30']
        jas_mapping_df {[DataFrame]} -- [Pandas DataFrame with mapping data]

    """
    for k,val in jas_mapping_df.iterrows():
        #print k,val
        #nse
        # count=0
        # int(jas_token)==1014 or 
        jas_token = str(val['jas_token'])
        if(int(jas_token) == 4 ):
            if(math.isnan(val['nse_instrument_token']) != True):
                time.sleep(0.5)
                print(val['nse_instrument_token'],star_date, end_date )
                response = kite.historical_data(int(val['nse_instrument_token']),
                                                star_date, 
                                                end_date, 
                                                interval='day',
                                                continuous=False)
                # print(response)

                for i in response:
                    try:

                        #print(i['date'],val['fr'],unicode(str(i['close'])))                            
                        _date = i['date'].strftime("%Y-%m-%d")	
                        openVal = i['open']
                        highVal = i['high']
                        lowVal = i['low']
                        closeVal = i['close']
                        volume = i['volume']
                        tradingSymbol = str(val['nse_symbol'])
                        nameMarket = str(val['name'])
                        dateNew = str(i['date'].strftime("%d-%b-%y"))
                        
                        #timestamp = datetime.now()

                        timeObj = datetime.strptime(i['date'].strftime("%Y-%m-%d"), "%Y-%m-%d")
                        timestampObj = time.mktime(timeObj.timetuple())
                        timestampNew = datetime.utcfromtimestamp(timestampObj)

                        json_obj_nse={}
                        json_obj_nse['nse_id']= str(jas_token)+"_"+_date
                        json_obj_nse['jas_token'] = int(jas_token)
                        json_obj_nse['name_market']= nameMarket
                        json_obj_nse['timestamp_string']= _date
                        json_obj_nse['trading_symbol'] = tradingSymbol
                        json_obj_nse['date_new']= dateNew
                        json_obj_nse['timestamp_date'] = timestampNew
                        json_obj_nse['open']= openVal
                        json_obj_nse['high']= highVal
                        json_obj_nse['low']= lowVal
                        json_obj_nse['close']= closeVal
                        json_obj_nse['volume']= volume
                        # print(json_obj_nse)

                        # Inserting data into database
                        columns = json_obj_nse.keys()
                        values = [json_obj_nse[column] for column in columns]
                        insert_statement = 'insert into nse_data (%s) values %s'
                        db_obj.insertQueryDict(insert_statement, columns, values)
                        print('running done for {0} -----   nse ----- {1}'.format(jas_token, _date)) 

                    except Exception as e:
                            print("Error in NSE Update  data for FR Token: {0}".format(jas_token))
                            print(e)
                            logging.error("historical_date_to_date function {0}".format(e))
                            print("---------------------------------------------------------------")
                            
                        #fsrefnse.document(u')        


# historical_date_to_date('2008-01-01','2018-07-05')
def main_date_loop(df_mapping):  
    val = datetime.now() - timedelta(days=365*5-1) ## Historical date update
    # val = datetime.now()                           ## Daily date update
    val_1 = datetime.now()

    val = val.strftime('%Y-%m-%d')
    val_1 = val_1.strftime('%Y-%m-%d')
    
    print(val,val_1)        
    historical_date_to_date(val, val_1, df_mapping)
    
# Mapping DF instance
query_mapping = "SELECT * FROM mapping_data"
# query_mapping = "SELECT * FROM mapping_data WHERE fr_token=17"
df_mapping = postgre_sql_read_df(query_mapping)
print(df_mapping)
# Read Historical data --- [Comment the below line when daily running data]
main_date_loop(df_mapping)

# Daily Data Fetch
start_date = datetime.now().strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
historical_date_to_date(start_date, end_date, df_mapping)


# Close database connection1
db_obj.closeConn()
