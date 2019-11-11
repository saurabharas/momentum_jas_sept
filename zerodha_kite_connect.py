# Zerodha Imports
from kiteconnect import KiteConnect

# standard python imports 
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

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
# access_token,public_token,request_token,api_key,api_secret = at.getAll()
data = kite.generate_session(request_token, api_secret=api_secret)
kite.set_access_token(data["access_token"])
kite.set_access_token(access_token)


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
        jas_token = str(val['FR Token'])
        if(int(jas_token)>=0):            
            if int(jas_token)==1020:
                try:
                    time.sleep(0.5)
                    response = kite.historical_data(int(val['instrument_token']),
                                                    star_date, 
                                                    end_date, 
                                                    interval='day',
                                                    continuous=False)

                    for i in response:
                        #print(i['date'],val['fr'],unicode(str(i['close'])))                            
                        _date = i['date'].strftime("%Y-%m-%d")	
                        openVal = i['open']
                        highVal = i['high']
                        lowVal = i['low']
                        closeVal = i['close']
                        volume = i['volume']
                        tradingSymbol = str(val['tradingsymbol'])
                        nameMarket = str(val['name'])
                        dateNew = str(i['date'].strftime("%d-%b-%y"))
                        
                        #timestamp = datetime.now()

                        timeObj = datetime.strptime(i['date'].strftime("%Y-%m-%d"), "%Y-%m-%d")
                        timestampObj = time.mktime(timeObj.timetuple())
                        timestampNew = datetime.utcfromtimestamp(timestampObj)

                        json_nse_dict={}
                        json_nse_dict['id']= str(jas_token)+"_"+_date
                        json_nse_dict['close']= closeVal
                        json_nse_dict['jas_token']= int(jas_token)
                        json_nse_dict['high']= highVal
                        json_nse_dict['low']= lowVal
                        json_nse_dict['nameMarket']= nameMarket
                        json_nse_dict['open']= openVal
                        json_nse_dict['timestamp']= _date
                        json_nse_dict['tradingSymbol']= tradingSymbol
                        json_nse_dict['volume']= volume
                        json_nse_dict['dateNew']= dateNew 

                        # Inserting data into database
                        columns = json_nse_dict.keys()
                        values = [json_nse_dict[column] for column in columns]
                        insert_statement = 'insert into nse_data (%s) values %s'
                        db_obj.insertQueryDict(insert_statement, columns, values)
                        print('running done for '+jas_token+' -----   '+' nse '+str(star_date)) 

                except Exception as e:
                        print("Couldn't get NSE Mongo Update  data for FR Token: {0}".format(jas_token))
                        print(e)            
                        #fsrefnse.document(u')        


# historical_date_to_date('2008-01-01','2018-07-05')
def historical_date_loop(df_mapping):
    val = '2008-01-01'
    val_1 = datetime.now() - timedelta(days=365*10)
    val_1 = val_1.strftime('%Y-%m-%d')

    print(val,val_1)        
    historical_date_to_date(val, val_1, df_mapping)
    
    val = datetime.now() - timedelta(days=365*10-1)
    val_1 = datetime.now() - timedelta(days=365*5)
    
    val = val.strftime('%Y-%m-%d')
    val_1 = val_1.strftime('%Y-%m-%d')
    print(val,val_1)        
  
    historical_date_to_date(val, val_1, df_mapping)

    val = datetime.now() - timedelta(days=365*5-1)
    val_1 = datetime.now() 
    val = val.strftime('%Y-%m-%d')
    val_1 = val_1.strftime('%Y-%m-%d')
    print(val,val_1)        
    historical_date_to_date(val, val_1, df_mapping)
    
# Mapping DF instance
query_mapping = "SELECT * FROM mapping_data"
df_mapping = postgre_sql_read_df(query_mapping)

# Read Historical data --- [Comment the below line when daily running data]
historical_date_loop(df_mapping)

# Daily Data Fetch
start_date = datetime.now().strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
historical_date_to_date(start_date, end_date, df_mapping)


# Close database connection
db_obj.closeConn()
