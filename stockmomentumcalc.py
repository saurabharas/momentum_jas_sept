# Standard Imports
import pandas as pd
import sys
import numpy as np
import logging
from datetime import datetime, timedelta

# from testMailPython import mailDesc

import math
import scipy.stats
from collections import OrderedDict
from get_latest_tuesday import latest_tuesday_func

mainDownloadDir = 'excelfilesfolder'
downloadDir = 'excelStockMomentum'

# PostgresDB connection 
import psycopg2
from sqlalchemy import create_engine

# Relative imports
from  postgresql_conn import Database
import get_latest_tuesday

##### Database connection Strings
conn_string = "host='localhost' dbname='jas' user='postgres' password='sau651994'"
conn_string_pd = 'postgresql://postgres:sau651994@localhost:5432/jas'

logging.basicConfig(filename="../error_files_log/stockmomentumcalc_{0}.txt".format(datetime.now().strftime("%Y_%m_%d")), filemode = 'w')
db_obj = Database(logging, url = conn_string)

##Ahlawat Funct####
def PostgreSQLConnect(query):
    db_postgresql = psycopg2.connect(conn_string_pd)
    cur = db_postgresql.cursor()
    data = []
    cur.execute(query)

    _d = cur.fetchall()
    data.extend(_d)

    cur.close()
    db_postgresql.close()

    return data

##Ahlawat Funct###
def PostgreSQLDML(query):
    db_postgresql = psycopg2.connect(conn_string_pd)
    cur = db_postgresql.cursor()
    cur.execute(query)
    db_postgresql.commit()
    db_postgresql.close()


##Nifty500
def postgre_sql_read_df(query):
    """Creates postresql_conn object and closes connection immidiately after usage.
    This will help tackle connection pool exceed issue.
    
    Arguments:
        query {[String]} -- [SQL query to read data from postgresql and 
                            create a pandas dataframe]

    """
    # conn_string_alchemy = "postgresql://whitedwarf:#finre123#@finre.cgk4260wbdoi.ap-south-1.rds.amazonaws.com/finre"
    engine = create_engine(conn_string_pd)
    df = pd.read_sql_query(query, con=engine)
    engine.dispose()
    return df

### Calculation Functions
def atr_loop(df):
    atr_count = 0
    df['atr'] = np.nan
    df['index_val'] = atr_count
    for index, val in df.iterrows():
        # pass
        if(atr_count == 19):
            # df.loc[35, 'atr']  = df['atr'].iloc[0:20].mean()
            df['atr'].iloc[19] = df['tr'].iloc[0:20].mean()
        if(atr_count >= 20):
            previous_atr_val = df['atr'].iloc[atr_count-1]
            current_tr_val = df['tr'].iloc[atr_count]
            df['atr'].iloc[atr_count] = (
                (previous_atr_val*19)+current_tr_val)/20
            # print(atr_count,previous_atr_val,current_tr_val)
        df['index_val'].iloc[atr_count] = atr_count
        atr_count = atr_count+1
    return df


def trueRangeCalc(df):
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['l_pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    return df


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return slope, r_value**2


def getXAndYforSlopeCalc(df):
    x_vals = list()
    y_vals = list()
    x_count = 0
    for index, val in df.iterrows():
        # print(index,val)
        x_count = x_count+1
        x_vals.append(x_count)
        y_vals.append(val['log_close'])

    return x_vals, y_vals


def log_val(df):
    df['log_close'] = np.log(df['close'])

    return df


def pct_change(df):
    df['pct_change'] = df['close'].pct_change(1)
    # print(df)
    return df


def sliceDf(df, row_count):
    #sliceDf latest 105 Days
    df = df.iloc[-row_count:, :]
    # df = df.iloc[:,:]
    # print(df)
    return df

def main_stock_momemtum_calc_jas():
    pd.options.mode.chained_assignment = None  # default='warn'
    todaysDate = datetime.now()
    # todaysDateString = todaysDate.strftime('%Y-%m-%d')
    latest_date = latest_tuesday_func(todaysDate)
    list_jobj_momentum_500 = list()
    # worksheet = excelSheetPart('stockMomentum_nifty500')
    # worksheet = excelSheetPart('actual_return_ratio')
    query_mapping = "Select jas_token, nse_symbol, name, mcap_crore from mapping_data WHERE nifty_500='Yes' ORDER BY jas_token"
    tokens_df = postgre_sql_read_df(query_mapping)

    for key, value in tokens_df.iterrows():
        try:
            if(value['jas_token'] >= 0):
                jas_token = value['jas_token']
                trading_symbol = value['nse_symbol']
                name_market = value['name']
                query_nse_data = "Select * from nse_data WHERE jas_token={0} ORDER BY timestamp_date".format(jas_token)
                df = postgre_sql_read_df(query_nse_data)
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric)

                # df = sliceDf(df,105)
                df = sliceDf(df, 100)

                # print("************* Moving Average ***************")
                # print(df)
                mv_avg = df['close'].iloc[:].mean() # Moving Avg 
                df = sliceDf(df, 91)                

                # print("************* pct Chnage ***************")
                # print(df)
                df = pct_change(df)
                df = sliceDf(df, 90)

                # print("************* LOG VAL ***************")
                # print(df)
                df = log_val(df)
                # print(df.to_string())

                x_vals, y_vals = getXAndYforSlopeCalc(df)
                slope_val, r_sqr = rsquared(x_vals, y_vals)
                df = trueRangeCalc(df)
                df = atr_loop(df)

                exp_slope = slope_val
                exp_slope_val = math.exp(exp_slope)
                ann_slope = (exp_slope_val**250)-1
                adj_slope = r_sqr*ann_slope

                latest_nse_id = df['nse_id'].iloc[-1]
                latest_open = df['open'].iloc[-1]
                latest_high = df['high'].iloc[-1]
                latest_low = df['low'].iloc[-1]
                latest_close = df['close'].iloc[-1]
                latest_volume = df['volume'].iloc[-1]

                latest_timestamp = df.index[-1]
                max_pct_change = df['pct_change'].max()
                min_pct_change = df['pct_change'].min()

                # print(latest_close,mv_avg)
                bol_mv = 'False'
                bol_pct_change = 'False'

                latest_atr = df['atr'].iloc[-1]
                inv_amount = 10000
                # print(df)
                atr_final = (inv_amount*0.001)/latest_atr

                # print(latest_close)
                # print(mv_avg)

                if(latest_close > mv_avg):
                    bol_mv = 'True'

                if(max_pct_change > 0.15 or min_pct_change < -0.15):
                    bol_pct_change = 'True'

                # df.to_excel('/home/saurabh/excelfilesfolder/excelStockMomentum/3mindiaDF.xlsx')

                # print(df)

                jobj_stock_momentum = OrderedDict()

                jobj_stock_momentum['row_id'] = latest_nse_id
                jobj_stock_momentum['jas_token'] = jas_token
                jobj_stock_momentum['trading_symbol'] = trading_symbol
                jobj_stock_momentum['name_market'] = name_market
                jobj_stock_momentum['timestamp_string'] = latest_date
                jobj_stock_momentum['open'] = latest_open
                jobj_stock_momentum['high'] = latest_high
                jobj_stock_momentum['low'] = latest_low
                jobj_stock_momentum['close'] = latest_close
                jobj_stock_momentum['volume'] = latest_volume

                jobj_stock_momentum['exponential_slope'] = exp_slope
                jobj_stock_momentum['annualized_slope'] = ann_slope
                jobj_stock_momentum['r2'] = r_sqr
                jobj_stock_momentum['adjusted_slope'] = adj_slope
                jobj_stock_momentum['moving_avg'] = bol_mv
                jobj_stock_momentum['max_daily_move'] = bol_pct_change
                jobj_stock_momentum['stock_to_be_invested'] = atr_final
                jobj_stock_momentum['atr_latest'] = latest_atr
                list_jobj_momentum_500.append(jobj_stock_momentum)

                # print(list_jobj_momentum_500)
                # Inserting data into database
                columns = jobj_stock_momentum.keys()
                values = [jobj_stock_momentum[column] for column in columns]
                insert_statement = 'insert into stock_momentum_jas (%s) values %s'
                db_obj.insertQueryDict(insert_statement, columns, values)
                print('running done for {0} -----  nse ------ {1} '.format(value['jas_token'], jobj_stock_momentum['timestamp_string'])) 

        except Exception as e:
            print("********************* Exception for "+str(jas_token)+"   "+str(trading_symbol)+" *****************************")
            print(e)
            logging.error("main_stock_momemtum_calc_jas ------ {0}".format(e))


    try:
        df_jobj_momentum = pd.DataFrame(list_jobj_momentum_500)
        df_jobj_momentum = df_jobj_momentum.sort_values(by='adjusted_slope',ascending=False)
        workBookName = 'stock_momentum_nifty500_'+str(latest_date)+'.xlsx'
        df_jobj_momentum.to_excel('../excel_data_output/{0}'.format(workBookName), sheet_name='nifty500')

        list_mom_500 = df_jobj_momentum.to_dict('records')
        print(df_jobj_momentum)

    except Exception as e:
        print("********************* Exception Excel Data Update *****************************")
        print(e)
        logging.error("Excel Update ------ {0}".format(e))


# def mainStockMomemtumAllFr():
#     fr, frToNse, frToBse, bseToFr, indicesNseInd = ubuntuPickle()  # ---Change required
#     # nifty500_items=nifty500()

#     # df_iter = pd.read_excel('/home/saurabh/excelfilesfolder/importantExcelDataRead/Stock Momentum required format.xlsx')
#     # df_iter = df_iter.iloc[:,0:3]
#     # # print(df_iter)
#     pd.options.mode.chained_assignment = None  # default='warn'
#     todaysDate = datetime.now()
#     todaysDateString = todaysDate.strftime('%Y-%m-%d')
#     worksheet = excelSheetPart('stockMomentum_All_Fr')

#     query = "Select jas_token, trading_symbol, name, mcap from stocks_token ORDER BY jas_token"
#     tokens = PostgreSQLConnect(query)
#     tokensdf = pd.DataFrame(tokens, columns = ['jas_token','trading_symbol','name','mcap'])
#     list_jobj_momentum_all = list()

#     todaysDate = datetime.now()
#     # todaysDateString = todaysDate.strftime('%Y-%m-%d')
#     latest_date = latest_tuesday_func(todaysDate)
#     print(tokensdf)
#     for key, value in tokensdf.iterrows():
#         try:
#             # print(value)
#             if(value['jas_token'] >= 0):
#                 fr = value['jas_token']
#                 trading_symbol = value['trading_symbol']
#                 name_market = value['name']
#                 df = getPandasDf(getDataFromDynamo(
#                     fr, dynamo_Hist10Y,latest_date))  # Daily Data

#                 df = df.apply(pd.to_numeric)
#                 # df = sliceDf(df,105)
#                 df = sliceDf(df, 100)
#                 # print("************* Moving Average ***************")
#                 # print(df)
#                 mv_avg = df['close'].iloc[:].mean()
#                 df = sliceDf(df, 91)
#                 # print("************* pct Chnage ***************")
#                 # print(df)
#                 df = pct_change(df)
#                 df = sliceDf(df, 90)
#                 # print("************* LOG VAL ***************")
#                 # print(df)
#                 # print("***************************************************")
#                 df = log_val(df)
#                 # print(df.to_string())
#                 print(df)
#                 x_vals, y_vals = getXAndYforSlopeCalc(df)
#                 # print(x_vals, y_vals)
#                 slope_val, r_sqr = rsquared(x_vals, y_vals)
#                 df = trueRangeCalc(df)
#                 df = atr_loop(df)

#                 exp_slope = slope_val
#                 exp_slope_val = math.exp(exp_slope)
#                 ann_slope = (exp_slope_val**250)-1
#                 adj_slope = r_sqr*ann_slope

#                 latest_close = df['close'].iloc[-1]

#                 latest_timestamp = df.index[-1]
#                 max_pct_change = df['pct_change'].max()
#                 min_pct_change = df['pct_change'].min()

#                 # print(latest_close,mv_avg)
#                 bol_mv = 'False'
#                 bol_pct_change = 'False'

#                 latest_atr = df['atr'].iloc[-1]
#                 inv_amount = 10000
#                 # print(df)
#                 atr_final = (inv_amount*0.001)/latest_atr

#                 # print(latest_close)
#                 # print(mv_avg)

#                 if(latest_close > mv_avg):
#                     bol_mv = 'True'

#                 if(max_pct_change > 0.15 or min_pct_change < -0.15):
#                     bol_pct_change = 'True'
#                 # print(df)
#                 # df.to_excel('/home/saurabh/excelfilesfolder/excelStockMomentum/3mindiaDF.xlsx')

#                 jobj_stock_momentum = OrderedDict()
#                 jobj_stock_momentum['_id'] = fr
#                 jobj_stock_momentum['jas_token'] = fr
#                 jobj_stock_momentum['tradingSymbol'] = trading_symbol
#                 jobj_stock_momentum['nameMarket'] = name_market
#                 jobj_stock_momentum['timestamp'] = latest_date
#                 jobj_stock_momentum['close'] = latest_close

#                 jobj_stock_momentum['exponential_slope'] = exp_slope
#                 jobj_stock_momentum['annualized_slope'] = ann_slope
#                 jobj_stock_momentum['r2'] = r_sqr
#                 jobj_stock_momentum['adjusted_slope'] = adj_slope
#                 jobj_stock_momentum['moving_avg'] = bol_mv
#                 jobj_stock_momentum['max_daily_move'] = bol_pct_change
#                 jobj_stock_momentum['stock_to_be_invested'] = atr_final
#                 jobj_stock_momentum['atr_latest'] = latest_atr

#                 jobj_stock_momentum = float_to_decimal_convert(jobj_stock_momentum)
#                 list_jobj_momentum_all.append(jobj_stock_momentum)

#                 # write_to_dynamodb(dynamo_stock_momentum_raw_all,jobj_stock_momentum,str(todaysDateString))
#                 print("Running for ....."+str(fr))


#                 # print("atr: "+str(atr_final))

#                 # print(jobj_stock_momentum)
#                 # print("slope_val,r_sqr: "+str(slope_val)+"------"+str(r_sqr))

#         except Exception as e:
#             # print('not cont..')
#             # print("********************* Exception for "+str(fr)+"   "+str(trading_symbol)+" *****************************")
#             print(e)

#     df_jobj_momentum = pd.DataFrame(list_jobj_momentum_all)
#     df_jobj_momentum = df_jobj_momentum.sort_values(by='adjusted_slope',ascending=False)
#     list_mom_all = df_jobj_momentum.to_dict('records')
#     rank=1
#     for jobj_stock_momentum in list_mom_all:
#         jobj_stock_momentum['rank_momentum'] = rank
#         write_to_dynamodb(dynamo_stock_momentum_raw_all,jobj_stock_momentum,str(jobj_stock_momentum['jas_token']))
#         rank=rank+1
#         print("Done for ....."+str(jobj_stock_momentum['jas_token']))

#     workBookName = 'stock_momentum_nifty500_'+str(latest_date)+'.xlsx'

#     writeToWorkbook('/home/saurabh/'+mainDownloadDir +
#                     '/'+downloadDir+'/'+workBookName)
#     print("******************* All Write Done ***************************************")


# def mainStockMomemtumCalcFrNifty100():
#     fr_items, frToNse, frToBse, bseToFr, nseIndFr = ubuntuPickle()
#     # nifty100_items=nifty100()
#     pd.options.mode.chained_assignment = None  # default='warn'
#     df_iter = pd.read_excel(
#         '/home/saurabh/excelfilesfolder/importantExcelDataRead/Stock Momentum required format.xlsx')
#     df_iter = df_iter.iloc[:, 0:3]
#     # print(df_iter)
#     todaysDate = datetime.now()
#     # todaysDateString = todaysDate.strftime('%Y-%m-%d')
#     latest_date = latest_tuesday_func(todaysDate)
#     list_jobj_momentum_100 = list()
#     worksheet = excelSheetPart('stockMomentum_nifty100')
#     # worksheet = excelSheetPart('actual_return_ratio')
#     query = "Select jas_token, trading_symbol, name, mcap from stocks_token WHERE nifty100=TRUE ORDER BY jas_token"
#     tokens = PostgreSQLConnect(query)
#     tokensdf = pd.DataFrame(tokens, columns = ['jas_token','trading_symbol','name','mcap'])

#     for key, value in tokensdf.iterrows():
#         # print(key,value)
#         #3272,1491
#         # print(value['fr'])
#         try:
#             if(value['jas_token'] >= 0):
#                 fr = value['jas_token']
#                 trading_symbol = value['trading_symbol']
#                 name_market = value['name']
#                 df = getPandasDf(getDataFromDynamo(
#                     fr, dynamo_Hist10Y,latest_date))  # Daily Data
#                 df = df.apply(pd.to_numeric)
#                 # print(df)
#                 # print("*************************************")
#                 # df = sliceDf(df,105)
#                 df = sliceDf(df, 100)
#                 # print("************* Moving Average ***************")
#                 # print(df)
#                 mv_avg = df['close'].iloc[:].mean()
#                 df = sliceDf(df, 91)
#                 # print("************* pct Chnage ***************")
#                 # print(df)
#                 df = pct_change(df)
#                 df = sliceDf(df, 90)
#                 # print("************* LOG VAL ***************")
#                 # print(df)
#                 df = log_val(df)
#                 # print(df.to_string())

#                 x_vals, y_vals = getXAndYforSlopeCalc(df)
#                 slope_val, r_sqr = rsquared(x_vals, y_vals)
#                 df = trueRangeCalc(df)
#                 df = atr_loop(df)

#                 exp_slope = slope_val
#                 exp_slope_val = math.exp(exp_slope)
#                 ann_slope = (exp_slope_val**250)-1
#                 adj_slope = r_sqr*ann_slope

#                 latest_close = df['close'].iloc[-1]

#                 latest_timestamp = df.index[-1]
#                 max_pct_change = df['pct_change'].max()
#                 min_pct_change = df['pct_change'].min()

#                 # print(latest_close,mv_avg)
#                 bol_mv = 'False'
#                 bol_pct_change = 'False'

#                 latest_atr = df['atr'].iloc[-1]
#                 inv_amount = 10000
#                 # print(df)
#                 atr_final = (inv_amount*0.001)/latest_atr

#                 # print(latest_close)
#                 # print(mv_avg)

#                 if(latest_close > mv_avg):
#                     bol_mv = 'True'

#                 if(max_pct_change > 0.15 or min_pct_change < -0.15):
#                     bol_pct_change = 'True'
#                 # print(df)
#                 # df.to_excel('/home/saurabh/excelfilesfolder/excelStockMomentum/3mindiaDF.xlsx')

#                 jobj_stock_momentum = OrderedDict()

#                 jobj_stock_momentum['jas_token'] = fr
#                 jobj_stock_momentum['tradingSymbol'] = trading_symbol
#                 jobj_stock_momentum['nameMarket'] = name_market
#                 jobj_stock_momentum['timestamp'] = latest_date
#                 jobj_stock_momentum['close'] = latest_close

#                 jobj_stock_momentum['exponential_slope'] = exp_slope
#                 jobj_stock_momentum['annualized_slope'] = ann_slope
#                 jobj_stock_momentum['r2'] = r_sqr
#                 jobj_stock_momentum['adjusted_slope'] = adj_slope
#                 jobj_stock_momentum['moving_avg'] = bol_mv
#                 jobj_stock_momentum['max_daily_move'] = bol_pct_change
#                 jobj_stock_momentum['stock_to_be_invested'] = atr_final
#                 jobj_stock_momentum['atr_latest'] = latest_atr
#                 # print(jobj_stock_momentum)
#                 # print("***********************************")
#                 jobj_stock_momentum = float_to_decimal_convert(jobj_stock_momentum)
#                 # excelSheetWriteData(jobj_stock_momentum, worksheet)
#                 # writeToMongo(mongo_db_stockmomentum_nifty100,
#                 #              jobj_stock_momentum, str(todaysDateString))
#                 # print(jobj_stock_momentum)
#                 list_jobj_momentum_100.append(jobj_stock_momentum)
#                 # write_to_dynamodb(dynamo_stock_momentum_raw_nifty100,jobj_stock_momentum,str(latest_date))
#                 print("Running for ....."+str(fr))
#                 # print("atr: "+str(atr_final))

#                 # print(jobj_stock_momentum)
#                 # print("slope_val,r_sqr: "+str(slope_val)+"------"+str(r_sqr))

#         except Exception as e:
#             # print('not cont..')
#             # print("********************* Exception for "+str(fr)+"   "+str(trading_symbol)+" *****************************")
#             print(e)
#     df_jobj_momentum = pd.DataFrame(list_jobj_momentum_100)
#     df_jobj_momentum = df_jobj_momentum.sort_values(by='adjusted_slope',ascending=False)
#     workBookName = 'stock_momentum_nifty100_'+str(latest_date)+'.xlsx'
#     df_jobj_momentum.to_excel('/home/saurabh/'+mainDownloadDir +
#                         '/'+downloadDir+'/'+workBookName,sheet_name='nifty100')

#     list_mom_100 = df_jobj_momentum.to_dict('records')

#     # rank=1
#     # for jobj_stock_momentum in list_mom_100:
#     #     jobj_stock_momentum['rank_momentum'] = rank
#     #     write_to_dynamodb(dynamo_stock_momentum_raw_nifty100,jobj_stock_momentum,str(jobj_stock_momentum['jas_token']))
#     #     rank=rank+1
#     #     print("Done for ....."+str(jobj_stock_momentum['jas_token']))

#     # workBookName = 'stock_momentum_nifty100_'+str(latest_date)+'.xlsx'
#     # writeToWorkbook('/home/saurabh/'+mainDownloadDir +
#     #                     '/'+downloadDir+'/'+workBookName)
#     print("******************* All Write Done ***************************************")


# def mainStockMomemtumCalcFrNiftyMid150():
#     fr_items, frToNse, frToBse, bseToFr, nseIndFr = ubuntuPickle()
#     # niftymidcap150_items=niftymidcap150()
#     pd.options.mode.chained_assignment = None  # default='warn'
#     df_iter = pd.read_excel(
#         '/home/saurabh/excelfilesfolder/importantExcelDataRead/Stock Momentum required format.xlsx')
#     df_iter = df_iter.iloc[:, 0:3]
#     # print(df_iter)
#     todaysDate = datetime.now()
#     # todaysDateString = todaysDate.strftime('%Y-%m-%d')
#     latest_date = latest_tuesday_func(todaysDate)
#     list_jobj_momentum_midcap150 = list()
#     worksheet = excelSheetPart('stockMomentum_niftymidcap150')
#     # worksheet = excelSheetPart('actual_return_ratio')
#     query = "Select jas_token, trading_symbol, name, mcap from stocks_token WHERE niftymidcap150=TRUE ORDER BY jas_token"
#     tokens = PostgreSQLConnect(query)
#     tokensdf = pd.DataFrame(tokens, columns = ['jas_token','trading_symbol','name','mcap'])

#     for key, value in tokensdf.iterrows():
#         # print(key,value)
#         #3272,1491
#         # print(value['fr'])
#         try:
#             if(value['jas_token'] >= 0):
#                 fr = value['jas_token']
#                 trading_symbol = value['trading_symbol']
#                 name_market = value['name']
#                 df = getPandasDf(getDataFromDynamo(
#                     fr, dynamo_Hist10Y,latest_date))  # Daily Data
#                 df = df.apply(pd.to_numeric)
#                 # print(df)
#                 # print("*************************************")
#                 # df = sliceDf(df,105)
#                 df = sliceDf(df, 100)
#                 # print("************* Moving Average ***************")
#                 # print(df)
#                 mv_avg = df['close'].iloc[:].mean()
#                 df = sliceDf(df, 91)
#                 # print("************* pct Chnage ***************")
#                 # print(df)
#                 df = pct_change(df)
#                 df = sliceDf(df, 90)
#                 # print("************* LOG VAL ***************")
#                 # print(df)
#                 df = log_val(df)
#                 # print(df.to_string())

#                 x_vals, y_vals = getXAndYforSlopeCalc(df)
#                 slope_val, r_sqr = rsquared(x_vals, y_vals)
#                 df = trueRangeCalc(df)
#                 df = atr_loop(df)

#                 exp_slope = slope_val
#                 exp_slope_val = math.exp(exp_slope)
#                 ann_slope = (exp_slope_val**250)-1
#                 adj_slope = r_sqr*ann_slope

#                 latest_close = df['close'].iloc[-1]

#                 latest_timestamp = df.index[-1]
#                 max_pct_change = df['pct_change'].max()
#                 min_pct_change = df['pct_change'].min()

#                 # print(latest_close,mv_avg)
#                 bol_mv = 'False'
#                 bol_pct_change = 'False'

#                 latest_atr = df['atr'].iloc[-1]
#                 inv_amount = 10000
#                 # print(df)
#                 atr_final = (inv_amount*0.001)/latest_atr

#                 # print(latest_close)
#                 # print(mv_avg)

#                 if(latest_close > mv_avg):
#                     bol_mv = 'True'

#                 if(max_pct_change > 0.15 or min_pct_change < -0.15):
#                     bol_pct_change = 'True'
#                 # print(df)
#                 # df.to_excel('/home/saurabh/excelfilesfolder/excelStockMomentum/3mindiaDF.xlsx')

#                 jobj_stock_momentum = OrderedDict()

#                 jobj_stock_momentum['jas_token'] = fr
#                 jobj_stock_momentum['tradingSymbol'] = trading_symbol
#                 jobj_stock_momentum['nameMarket'] = name_market
#                 jobj_stock_momentum['timestamp'] = latest_date
#                 jobj_stock_momentum['close'] = latest_close

#                 jobj_stock_momentum['exponential_slope'] = exp_slope
#                 jobj_stock_momentum['annualized_slope'] = ann_slope
#                 jobj_stock_momentum['r2'] = r_sqr
#                 jobj_stock_momentum['adjusted_slope'] = adj_slope
#                 jobj_stock_momentum['moving_avg'] = bol_mv
#                 jobj_stock_momentum['max_daily_move'] = bol_pct_change
#                 jobj_stock_momentum['stock_to_be_invested'] = atr_final
#                 jobj_stock_momentum['atr_latest'] = latest_atr
#                 # print(jobj_stock_momentum)
#                 # print("***********************************")
#                 jobj_stock_momentum = float_to_decimal_convert(jobj_stock_momentum)
#                 # excelSheetWriteData(jobj_stock_momentum, worksheet)
#                 # writeToMongo(mongo_db_stockmomentum_niftymidcap150,
#                 #              jobj_stock_momentum, str(todaysDateString))
#                 # print(jobj_stock_momentum)
#                 list_jobj_momentum_midcap150.append(jobj_stock_momentum)
#                 # write_to_dynamodb(dynamo_stock_momentum_raw_niftymidcap150,jobj_stock_momentum,str(latest_date))
#                 print("Running for ....."+str(fr))
#                 # print("atr: "+str(atr_final))

#                 # print(jobj_stock_momentum)
#                 # print("slope_val,r_sqr: "+str(slope_val)+"------"+str(r_sqr))

#         except Exception as e:
#             # print('not cont..')
#             # print("********************* Exception for "+str(fr)+"   "+str(trading_symbol)+" *****************************")
#             print(e)
#     df_jobj_momentum = pd.DataFrame(list_jobj_momentum_midcap150)
#     df_jobj_momentum = df_jobj_momentum.sort_values(by='adjusted_slope',ascending=False)
#     workBookName = 'stock_momentum_nifty150_'+str(latest_date)+'.xlsx'
#     df_jobj_momentum.to_excel('/home/saurabh/'+mainDownloadDir +
#                         '/'+downloadDir+'/'+workBookName,sheet_name='nifty150')

#     list_mom_midcap150 = df_jobj_momentum.to_dict('records')

#     # rank=1
#     # for jobj_stock_momentum in list_mom_midcap150:
#     #     jobj_stock_momentum['rank_momentum'] = rank
#     #     write_to_dynamodb(dynamo_stock_momentum_raw_niftymidcap150,jobj_stock_momentum,str(jobj_stock_momentum['jas_token']))
#     #     rank=rank+1
#     #     print("Done for ....."+str(jobj_stock_momentum['jas_token']))

#     # workBookName = 'stock_momentum_niftymidcap150_'+str(latest_date)+'.xlsx'
#     # writeToWorkbook('/home/saurabh/'+mainDownloadDir +
#     #                     '/'+downloadDir+'/'+workBookName)
#     print("******************* All Write Done ***************************************")



# def mainStockMomemtumCalcFrNiftySmall250():
#     fr_items, frToNse, frToBse, bseToFr, nseIndFr = ubuntuPickle()
#     pd.options.mode.chained_assignment = None  # default='warn'
#     df_iter = pd.read_excel(
#         '/home/saurabh/excelfilesfolder/importantExcelDataRead/Stock Momentum required format.xlsx')
#     df_iter = df_iter.iloc[:, 0:3]
#     # print(df_iter)
#     todaysDate = datetime.now()
#     # todaysDateString = todaysDate.strftime('%Y-%m-%d')
#     latest_date = latest_tuesday_func(todaysDate)
#     list_jobj_momentum_smallcap250 = list()
#     worksheet = excelSheetPart('stockMomentum_niftysmallcap250')
#     # worksheet = excelSheetPart('actual_return_ratio')
#     query = "Select jas_token, trading_symbol, name, mcap from stocks_token WHERE niftysmallcap250=TRUE ORDER BY jas_token"
#     tokens = PostgreSQLConnect(query)
#     tokensdf = pd.DataFrame(tokens, columns = ['jas_token','trading_symbol','name','mcap'])

#     for key, value in tokensdf.iterrows():
#         # print(key,value)
#         #3272,1491
#         # print(value['fr'])
#         try:
#             if(value['jas_token'] >= 0):
#                 fr = value['jas_token']
#                 trading_symbol = value['trading_symbol']
#                 name_market = value['name']
#                 df = getPandasDf(getDataFromDynamo(
#                     fr, dynamo_Hist10Y,latest_date))  # Daily Data
#                 df = df.apply(pd.to_numeric)
#                 # print(df)
#                 # print("*************************************")
#                 # df = sliceDf(df,105)
#                 df = sliceDf(df, 100)
#                 # print("************* Moving Average ***************")
#                 # print(df)
#                 mv_avg = df['close'].iloc[:].mean()
#                 df = sliceDf(df, 91)
#                 # print("************* pct Chnage ***************")
#                 # print(df)
#                 df = pct_change(df)
#                 df = sliceDf(df, 90)
#                 # print("************* LOG VAL ***************")
#                 # print(df)
#                 df = log_val(df)
#                 # print(df.to_string())

#                 x_vals, y_vals = getXAndYforSlopeCalc(df)
#                 slope_val, r_sqr = rsquared(x_vals, y_vals)
#                 df = trueRangeCalc(df)
#                 df = atr_loop(df)

#                 exp_slope = slope_val
#                 exp_slope_val = math.exp(exp_slope)
#                 ann_slope = (exp_slope_val**250)-1
#                 adj_slope = r_sqr*ann_slope

#                 latest_close = df['close'].iloc[-1]

#                 latest_timestamp = df.index[-1]
#                 max_pct_change = df['pct_change'].max()
#                 min_pct_change = df['pct_change'].min()

#                 # print(latest_close,mv_avg)
#                 bol_mv = 'False'
#                 bol_pct_change = 'False'

#                 latest_atr = df['atr'].iloc[-1]
#                 inv_amount = 10000
#                 # print(df)
#                 atr_final = (inv_amount*0.001)/latest_atr

#                 # print(latest_close)
#                 # print(mv_avg)

#                 if(latest_close > mv_avg):
#                     bol_mv = 'True'

#                 if(max_pct_change > 0.15 or min_pct_change < -0.15):
#                     bol_pct_change = 'True'
#                 # print(df)
#                 # df.to_excel('/home/saurabh/excelfilesfolder/excelStockMomentum/3mindiaDF.xlsx')

#                 jobj_stock_momentum = OrderedDict()

#                 jobj_stock_momentum['jas_token'] = fr
#                 jobj_stock_momentum['tradingSymbol'] = trading_symbol
#                 jobj_stock_momentum['nameMarket'] = name_market
#                 jobj_stock_momentum['timestamp'] = latest_date
#                 jobj_stock_momentum['close'] = latest_close

#                 jobj_stock_momentum['exponential_slope'] = exp_slope
#                 jobj_stock_momentum['annualized_slope'] = ann_slope
#                 jobj_stock_momentum['r2'] = r_sqr
#                 jobj_stock_momentum['adjusted_slope'] = adj_slope
#                 jobj_stock_momentum['moving_avg'] = bol_mv
#                 jobj_stock_momentum['max_daily_move'] = bol_pct_change
#                 jobj_stock_momentum['stock_to_be_invested'] = atr_final
#                 jobj_stock_momentum['atr_latest'] = latest_atr
#                 # print(jobj_stock_momentum)
#                 # print("***********************************")
#                 jobj_stock_momentum = float_to_decimal_convert(jobj_stock_momentum)
#                 # excelSheetWriteData(jobj_stock_momentum, worksheet)
#                 # writeToMongo(mongo_db_stockmomentum_niftysmallcap250,
#                 #              jobj_stock_momentum, str(todaysDateString))
#                 # print(jobj_stock_momentum)
#                 list_jobj_momentum_smallcap250.append(jobj_stock_momentum)
#                 # write_to_dynamodb(dynamo_stock_momentum_raw_niftysmallcap250,jobj_stock_momentum,str(latest_date))
#                 print("Running for ....."+str(fr))
#                 # print("atr: "+str(atr_final))

#                 # print(jobj_stock_momentum)
#                 # print("slope_val,r_sqr: "+str(slope_val)+"------"+str(r_sqr))

#         except Exception as e:
#             # print('not cont..')
#             # print("********************* Exception for "+str(fr)+"   "+str(trading_symbol)+" *****************************")
#             print(e)
#     df_jobj_momentum = pd.DataFrame(list_jobj_momentum_smallcap250)
#     df_jobj_momentum = df_jobj_momentum.sort_values(by='adjusted_slope',ascending=False)
#     workBookName = 'stock_momentum_nifty250_'+str(latest_date)+'.xlsx'
#     df_jobj_momentum.to_excel('/home/saurabh/'+mainDownloadDir +
#                         '/'+downloadDir+'/'+workBookName,sheet_name='nifty250')

#     list_mom_smallcap250 = df_jobj_momentum.to_dict('records')

#     # rank=1
#     # for jobj_stock_momentum in list_mom_smallcap250:
#     #     jobj_stock_momentum['rank_momentum'] = rank
#     #     write_to_dynamodb(dynamo_stock_momentum_raw_niftysmallcap250,jobj_stock_momentum,str(jobj_stock_momentum['jas_token']))
#     #     rank=rank+1
#     #     print("Done for ....."+str(jobj_stock_momentum['jas_token']))

#     # workBookName = 'stock_momentum_niftysmallcap250_'+str(latest_date)+'.xlsx'
#     # writeToWorkbook('/home/saurabh/'+mainDownloadDir +
#     #                     '/'+downloadDir+'/'+workBookName)
#     print("******************* All Write Done ***************************************")


main_stock_momemtum_calc_jas()
# mainStockMomemtumCalcFrNifty100()
# mainStockMomemtumCalcFrNiftyMid150()
# mainStockMomemtumCalcFrNiftySmall250()
# mainStockMomemtumAllFr()
