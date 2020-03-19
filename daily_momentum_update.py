'''
    This file will daily update all the calculations of user like:
    inv_amount,inv_value,total_return,total_realized_return,
    total_unrealized_return,todays_return
'''
import pandas as pd
import sys
import numpy as np
# sys.path.insert(4,'C:\\MyDesktop\\pyFirefox') #---Change required
sys.path.insert(4, '/home/saurabh/pymainsau/pyfirefox')  # ---Change required
sys.path.insert(4, '/home/saurabh/pymainsau/pymail')
sys.path.insert(4, '/home/saurabh/pymainsau/pyutilities')

from datetime import datetime, timedelta

from testMailPython import mailDesc,mailDescHtml
from mappingActivity import windowsPickle, ubuntuPickle, nifty500
# from mongoActivity import mongo_db_return, writeToMongo, bulkWriteMongo
# from all_pandas_func import bse_stocks_pandas
from firefoxActivity import ubuntuFirefox, windowsFirefox
from renameFiles import file_rename
# from dataframe_fr_mapper import mapper_dataframe_ubuntu

import pymongo
from excelGenericModule import excelSheetPart, excelSheetWriteData, writeToWorkbook, excelSheetWriteData_new
import math
# from mongoActivity import mongo_db_return, writeToMongo, bulkWriteMongo, getFromMongo, closeMongoConnection,updateMongoEntireDoc
from collections import OrderedDict

# mongo_db_Hist10Y = mongo_db_return('Hist10Year')
# mongo_db_stock_momentum_user = mongo_db_return('stock_momentum_user')


mainDownloadDir = 'excelfilesfolder'
downloadDir = 'excelStockMomentum'
from datetime import datetime
from dateutil.relativedelta import relativedelta, TU
from get_latest_tuesday import latest_tuesday_func
'''
Algorithm Rebalancing and Reposistioning:
1.Save user portfolio in mongo every week (col:date,_id:uuid_date)
2.Get the last amount of the portfolio
3.Pass that amount and get the latest portfolio for the user
4.Perform Rebalancing and Repositioning on that.
'''
pd.options.mode.chained_assignment = None  # default='warn'
from sqlalchemy import create_engine
import psycopg2
CONNSTRING = "host='finre.cgk4260wbdoi.ap-south-1.rds.amazonaws.com' dbname='finre' user='whitedwarf' password='#finre123#'"
conn = psycopg2.connect(CONNSTRING)
cur = conn.cursor()
engine = create_engine('postgresql://whitedwarf:#finre123#@finre.cgk4260wbdoi.ap-south-1.rds.amazonaws.com:5432/finre')
from email.mime.text import MIMEText

'''
    Dynamo Addition
'''
from dynamoActivity import dynamo_db_return,write_to_dynamodb,bulk_write_dynamo,update_dynamo_single,update_dynamo_entire_doc,get_item,query_item,float_to_decimal_convert,df_decimal_to_float,decimlal_to_float
from boto3.dynamodb.conditions import Key, Attr ##dynamo
# dynamo_momentum_user_main = dynamo_db_return('stock_momentum_user_main') ##dynamo
dynamo_momentum_user_main = dynamo_db_return('stock_momentum_user') ##dynamo

dynamo_Hist10Year = dynamo_db_return('Hist10Year') ##dynamo

import psycopg2
from sqlalchemy import create_engine

conn_string = "host='finre.cgk4260wbdoi.ap-south-1.rds.amazonaws.com'dbname='finre' user='whitedwarf' password='#finre123#'"
conn_string_alchemy = "postgresql://whitedwarf:#finre123#@finre.cgk4260wbdoi.ap-south-1.rds.amazonaws.com/finre" 
engine = create_engine(conn_string_alchemy,pool_pre_ping=True)

amficode = 119164
# mongo_db_Hist10Y = mongo_db_return('Hist10Year')
# mongo_db_stock_momentum_user = mongo_db_return('stock_momentum_user')
list_res_obj = list()

def list_of_dict_to_df(doc_val):
    # print(doc_val)
    uuid = doc_val['uuid']
    # name = doc_val['name']
    arr_portfolio = doc_val['arr_momentum_data']
    arr_sold_current = doc_val['current_sold']
    arr_bought_current = doc_val['current_bought']
    arr_sold_total = doc_val['total_sold']
    arr_bought_total = doc_val['total_bought']

    df_portfolio = pd.DataFrame(arr_portfolio)
    df_portfolio['uuid'] = uuid
    print(df_portfolio)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    df_buy = pd.DataFrame(arr_bought_current)
    df_buy['uuid'] = uuid
    print(df_buy)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    df_sell = pd.DataFrame(arr_sold_current)
    df_sell['uuid'] = uuid
    print(df_sell)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    df_sell_total = pd.DataFrame(arr_sold_total)
    df_sell_total = calc_avg_sell_price(df_sell_total)
    df_sell_total['uuid'] = uuid


    df_bought_total = pd.DataFrame(arr_bought_total)
    df_bought_total = calc_avg_buy_price(df_bought_total)
    df_bought_total['uuid'] = uuid
    print(df_bought_total)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    '''
        ADD avg_buy_price to df_sell_total and df_portfolio
    '''
    if(not df_bought_total.empty):
        df_bought_total_temp = df_bought_total.drop_duplicates(subset=['fr_token'])
        if(not df_sell_total.empty):
            df_sell_total = pd.merge(df_sell_total,df_bought_total_temp[['fr_token','avg_buy_price']],on=['fr_token'],how='left',suffixes=('_x',''))
            if('avg_sell_price_y' in df_sell_total.columns):
                df_sell_total['avg_sell_price'] = df_sell_total['avg_sell_price_y']

        if(not df_portfolio.empty):
            print(df_portfolio)
            print("*********************************")
            df_portfolio = pd.merge(df_portfolio,df_bought_total_temp[['fr_token','avg_buy_price']],on=['fr_token'],how='left')

    print(df_sell_total)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


    print(df_portfolio)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


    df_portfolio = df_decimal_to_float(df_portfolio)
    df_buy = df_decimal_to_float(df_buy)
    df_sell = df_decimal_to_float(df_sell)
    df_bought_total = df_decimal_to_float(df_bought_total)
    df_sell_total = df_decimal_to_float(df_sell_total)
    return df_portfolio,df_buy,df_sell,df_bought_total,df_sell_total

def get_data_latest(dynamodb_db_Hist10Y,fr_token):
    res = dynamodb_db_Hist10Y.query(
        ScanIndexForward = False,
        Limit=3,
        KeyConditionExpression= Key('fr_token').eq(int(fr_token))
    )
    latest_record = res['Items']

    return latest_record

def get_inv_amount(df_portfolio):
    inv_value = 0
    if(not df_portfolio.empty):
        df_portfolio['current_inv_val'] = df_portfolio['current_close']*df_portfolio['stocks_to_be_bought']
        inv_value = df_portfolio['current_inv_val'].sum()
    else:
        inv_value = 0
    return inv_value

def get_unrealized_return(df_portfolio):
    unrealized_return = 0
    try:
        df_portfolio['unrealized_return'] = (df_portfolio['current_close']*df_portfolio['stocks_to_be_bought'])-(df_portfolio['avg_buy_price']*df_portfolio['stocks_to_be_bought'])
        unrealized_return = df_portfolio['unrealized_return'].sum()
    except Exception as e:
        print(e)

    return unrealized_return

def get_realized_return(df_sell_total):
    realized_return = 0
    try:
        df_sell_total['realized_return'] = (df_sell_total['avg_sell_price']*df_sell_total['stocks_to_be_sold']) - (df_sell_total['avg_buy_price']*df_sell_total['stocks_to_be_sold'])
        realized_return = df_sell_total['realized_return'].sum()
    except Exception as e:
        print(e)

    return realized_return

def get_todays_return(df_portfolio):
    todays_return = 0
    try:
        df_portfolio['todays_return'] = (df_portfolio['current_close']*df_portfolio['stocks_to_be_bought'])-(df_portfolio['yest_close']*df_portfolio['stocks_to_be_bought'])
        # df_portfolio['yest_return'] = (df_portfolio['yest_close']*df_portfolio['stocks_to_be_bought'])-(df_portfolio['day_before_yest_close']*df_portfolio['stocks_to_be_bought'])

        todays_return = df_portfolio['todays_return'].sum()
        df_portfolio['yest_val'] = (df_portfolio['yest_close']*df_portfolio['stocks_to_be_bought'])
        yest_val = df_portfolio['yest_val'].sum()
        # yest_return = df_portfolio['yest_return'].sum()
        todays_return_per = (todays_return/yest_val)*100

    except Exception as e:
        print(e)
    return todays_return,todays_return_per

'''
    input: total_bought
'''
def calc_avg_buy_price(df_total_bought):
    if(not df_total_bought.empty):
        df_total_bought['total_price_stock'] = df_total_bought['stocks_to_be_bought_new']*df_total_bought['close']
        temp_avg_df = df_total_bought.groupby(['fr_token'], as_index=False,sort=False)['stocks_to_be_bought_new','total_price_stock'].sum()
        temp_avg_df['avg_buy_price'] = temp_avg_df['total_price_stock']/temp_avg_df['stocks_to_be_bought_new']
        df_total_bought = pd.merge(df_total_bought,temp_avg_df[['fr_token','avg_buy_price']],on=['fr_token'],how='left')
    else:
        df_total_bought = df_total_bought

    return df_total_bought


'''
    input: total_sold
'''
def calc_avg_sell_price(df_total_sold):
    if(not df_total_sold.empty):
        df_total_sold['total_price_stock'] = df_total_sold['stocks_to_be_sold']*df_total_sold['close']
        temp_avg_df = df_total_sold.groupby(['fr_token'], as_index=False,sort=False)['stocks_to_be_sold','total_price_stock'].sum()
        temp_avg_df['avg_sell_price'] = temp_avg_df['total_price_stock']/temp_avg_df['stocks_to_be_sold']
        df_total_sold = pd.merge(df_total_sold,temp_avg_df[['fr_token','avg_sell_price']],on=['fr_token'],how='left')
    else:
        df_total_sold = df_total_sold

    return df_total_sold



def momentum_calculations(df_portfolio,inv_amount,df_sell_total):
    sum_close_qty = 0
    for index,row in df_portfolio.iterrows():
        fr_token = row['fr_token']
        latest_record = get_data_latest(dynamo_Hist10Year,fr_token)
        current_close = float(latest_record[0]['close'])
        yest_close = float(latest_record[1]['close'])
        day_before_yest_close = float(latest_record[2]['close'])

        # print(current_close,yest_close)
        df_portfolio.loc[df_portfolio['fr_token'] == fr_token,'current_close'] = current_close
        df_portfolio.loc[df_portfolio['fr_token'] == fr_token,'yest_close'] = yest_close
        df_portfolio.loc[df_portfolio['fr_token'] == fr_token,'day_before_yest_close'] = day_before_yest_close

        # print(df_portfolio)
        # print("*****************")
    inv_value = get_inv_amount(df_portfolio)
    # print(inv_value,inv_amount)
    # inv_value = 100050 ###change here
    return_amount = inv_value - inv_amount
    return_per = (return_amount/inv_amount)*100

    if(not df_portfolio.empty):
        unrealized_return = get_unrealized_return(df_portfolio)
        todays_return,todays_return_per = get_todays_return(df_portfolio)
    else:
        unrealized_return = 0
        todays_return = 0
        todays_return_per = 0

    if(not df_sell_total.empty):
        realized_return = get_realized_return(df_sell_total)
    else:
        realized_return = 0

    return inv_value,return_amount,return_per,unrealized_return,realized_return,todays_return,todays_return_per


def get_current_nav(amficode):
    query = "SELECT * FROM histnav WHERE amficode=%d ORDER BY hdate DESC LIMIT 10"%(int(amficode))
    df = pd.read_sql_query(query,con=engine)
    # print(df)
    nav = df.iloc[0]['nav']
    previous_nav = df.iloc[1]['nav']
    mf_todays_return = ((nav-previous_nav)/previous_nav)*100
    mf_name = df.iloc[0]['fund_name']
    return nav,mf_name,mf_todays_return

'''
    Here we are calculating return from index when the user invested in our portfolio
    Algo:
    1. Get investment date from totalbought
    2. Get the return value from the index on the amount invested by user
    3. Get the inv_amount,inv_val and total_return
'''
def index_return_calc(df_total_bought,df_index_nifty_500,inv_amount):
    df_total_bought = df_total_bought.sort_values(by='timestamp',ascending=True)
    inv_date = df_total_bought.loc[0,'timestamp']
    # print(df_index_nifty_500)
    row_nifty_500 = df_index_nifty_500.loc[df_index_nifty_500['timestamp'] == inv_date ]
    inv_date_close = row_nifty_500.iloc[0]['close'] ## Close on the date user invested the amount
    current_close = df_index_nifty_500.iloc[-1]['close'] ##current close nifty500
    yest_close = df_index_nifty_500.iloc[-2]['close'] ##current close nifty500
    total_stocks_bought = float(inv_amount)/float(inv_date_close) ##total index qty bought in the amount
    inv_val_index = total_stocks_bought*float(current_close)

    # total_return_index = inv_val_index - inv_amount
    # total_return_per_index = ((inv_val_index-inv_amount)/inv_amount)*100
    total_return_index = current_close - inv_date_close
    total_return_per_index = ((inv_val_index-inv_amount)/inv_amount)*100

    today_return_index = current_close - yest_close
    today_return_per_index = ((current_close - yest_close)/yest_close)*100


    # print(inv_date,inv_date_close,current_close,inv_amount,inv_val_index,total_return_index,total_return_per_index)
    return inv_date_close,current_close,total_return_index,total_return_per_index,today_return_index,today_return_per_index


def check_nifty500_below_100dma(tuesday_date_string,df_total_bought,inv_amount):
    tuesday_date_string = latest_tuesday_func(datetime.now())

    dynamo_db_stockmomentum_nifty500 = dynamo_db_return(
    'stock_momentum_raw_nifty500')
    dynamo_db_index_nifty500 = dynamo_db_return('Hist10YearIndices')


    nifty500_fr_token = '10006'
    res = dynamo_db_index_nifty500.query(
        ScanIndexForward = True,
        KeyConditionExpression= Key('fr_token').eq(int(nifty500_fr_token)),
        FilterExpression= Attr('timestamp').lte(tuesday_date_string)
    )
    arr_momentum_raw = res['Items']
    df_index_nifty_500 = pd.DataFrame(arr_momentum_raw)
    # df_index_nifty_500 = pd.to_numeric(arr_momentum_raw,errors='ignore')
    # df_index_nifty_500 = pd.to_numeric(df_index_nifty_500, errors='coerce')
    inv_amount_index,inv_val_index,total_return_index,total_return_per_index,today_return_index,today_return_per_index = index_return_calc(df_total_bought,df_index_nifty_500,inv_amount) ##index return calulation function call


    df_last_100 = df_index_nifty_500.iloc[-200:,:]
    # print(df_last_100)
    mv_100_close = df_last_100['close'].mean()
    current_close = df_last_100['close'].iloc[-1]
    # print(mv_100_close,current_close)
    # print("*********************")
    flag_buy=False
    if(current_close<=mv_100_close):
        flag_buy=False
    else:
        flag_buy=True

    # flag_buy=False
    print(current_close,mv_100_close)
    print("Flag Buy..............."+str(flag_buy))
    return flag_buy,inv_amount_index,inv_val_index,total_return_index,total_return_per_index,today_return_index,today_return_per_index

'''
Algo:
1. Get the cash amount
2. If(market<100dma):
    - Invest in Mf:
        - Get nav for Mf
        - Get previous total units,Get previous inv_val
        - current_units = current_cash/nav
        - total_units = total_units+current_units
        - inv_amount = inv_amount+cash
        - inv_val = total_units*current_nav
        - return = inv_val - inv_amount
        - return_per = (inv_val - inv_amount)/inv_amount
13. else If(marke>100dma):
    - Keep cash as cash:
'''
def mutual_funds_calc(mf_current_cash_eq,mf_total_units,mf_inv_amount):
    mf_current_nav,mf_name,mf_todays_return = get_current_nav(amficode)
    # print("Current Nav: "+str(mf_current_nav))
    mf_current_units = mf_current_cash_eq/mf_current_nav
    # print("Current units: "+str(mf_current_cash_eq))
    # mf_inv_amount = 138803.74+4830.3
    # mf_total_units = 69.777
    # mf_current_units = 2.38
    mf_total_units = mf_total_units + mf_current_units
    # print("mf_total_units : "+str(mf_total_units))
    print(mf_inv_amount,mf_current_cash_eq)
    mf_inv_amount = mf_inv_amount+mf_current_cash_eq
    mf_inv_val = mf_total_units*mf_current_nav
    # print(mf_inv_val)
    mf_return_per = ((mf_inv_val-mf_inv_amount)/mf_inv_amount)*100
    mf_return_val = mf_inv_val - mf_inv_amount
    mf_unrealized_return = mf_inv_val - mf_inv_amount

    return mf_inv_amount,mf_total_units,mf_inv_val,mf_return_val,mf_return_per,mf_unrealized_return,mf_current_nav,mf_name,mf_todays_return


def main_calc(dynamo_momentum_user_main):
    flag_tue=0

    latest_tuesday_date = latest_tuesday_func(datetime.now())
    print(latest_tuesday_date)

    res = dynamo_momentum_user_main.query(
        ScanIndexForward = True,
        KeyConditionExpression= Key('timestamp').eq(latest_tuesday_date)
    )
    res = res['Items']
    print(latest_tuesday_date,(datetime.now()))
    if(len(res)==0 and latest_tuesday_date==(datetime.now() )): ###change remove timedelta
        flag_tue=1
        latest_tuesday_date = latest_tuesday_func(datetime.now()-timedelta(1)) ##change timedelta=1
        print(latest_tuesday_date)

        res = dynamo_momentum_user_main.query(
            ScanIndexForward = True,
            KeyConditionExpression= Key('timestamp').eq(latest_tuesday_date)
        )
        res = res['Items']

    # print(res)
    for doc_val in res:
        #'hY5fh1xr9uS3cXyEggdNCRSteEY2'
        if(doc_val['uuid'] != 'a'):
            res_previous=list()
            if(flag_tue==1):
                previous_res_days_num = 8 ##change value to 8
            else:
                previous_res_days_num = 7


            while(len(res_previous)==0):
                last_tuesday_date = latest_tuesday_func(datetime.now()-timedelta(previous_res_days_num))
                res_previous = dynamo_momentum_user_main.query(
                    ScanIndexForward = True,
                    KeyConditionExpression= Key('timestamp').eq(last_tuesday_date) & Key('uuid').eq(doc_val['uuid'])
                )
                res_previous = res_previous['Items']
                print(res_previous,last_tuesday_date)
                previous_res_days_num = previous_res_days_num+7
                if(len(res_previous)>0):
                    break

            # doc_val = decimlal_to_float(doc_val)
            user_obj = doc_val
            cash_eq = doc_val['cash_eq']
            inv_amount = doc_val['inv_amount']
            if(cash_eq is None):
                cash_eq=0
            print(cash_eq,inv_amount)
            cash_eq = float(cash_eq)
            inv_amount = float(inv_amount)
            mf_total_units = 0
            mf_inv_amount = 0
            mf_inv_val = 0
            mf_unrealized_return = 0
            mf_realized_return = 0

            if('mf_inv_amount' in doc_val):
                mf_inv_amount = float(doc_val['mf_inv_amount'])
                mf_total_units = float(doc_val['mf_total_units'])
                mf_inv_val = float(doc_val['mf_inv_val'])
                mf_realized_return = float(doc_val['mf_realized_return'])
                mf_return_val = float(doc_val['mf_return_val'])
                mf_unrealized_return = float(doc_val['mf_unrealized_return'])
            else:
                if(len(res_previous)>0):
                    if('mf_inv_amount' in res_previous[0]):
                        mf_inv_amount = float(res_previous[0]['mf_inv_amount'])
                    if('mf_total_units' in res_previous[0]):
                        mf_total_units = float(res_previous[0]['mf_total_units'])
                    if('mf_inv_val' in res_previous[0]):
                        mf_inv_val = float(res_previous[0]['mf_inv_val'])
                    if('mf_realized_return' in res_previous[0]):
                        mf_realized_return = float(res_previous[0]['mf_realized_return'])
                    if('mf_return_val' in res_previous[0]):
                        mf_return_val = float(res_previous[0]['mf_return_val'])
                    if('mf_unrealized_return' in res_previous[0]):
                        mf_unrealized_return = float(res_previous[0]['mf_unrealized_return'])

            '''
                converts list of dicts like arr_momentum_data,current_bought,current_sold,total_bought,total_sold
            '''
            df_portfolio,df_buy,df_sell,df_bought_total,df_sell_total = list_of_dict_to_df(doc_val)

            flag_nifty500,inv_amount_index,inv_val_index,total_return_index,total_return_per_index,today_return_index,today_return_per_index = check_nifty500_below_100dma(latest_tuesday_date,df_bought_total,inv_amount)

            print(datetime.now().strftime('%Y-%m-%d'),latest_tuesday_date)
            if(flag_nifty500==True):
                mf_current_nav,mf_name,mf_todays_return = get_current_nav(amficode)
                if(flag_tue==True):
                    cash_eq = cash_eq + mf_inv_val
                    mf_realized_return = mf_realized_return + mf_inv_val - mf_inv_amount
                    mf_return_val = mf_realized_return
                    mf_inv_val = 0
                    mf_inv_amount = 0
                    mf_unrealized_return = 0
                    mf_total_units = 0
                else:
                    cash_eq = cash_eq
                    mf_realized_return = mf_realized_return
                    mf_return_val = mf_return_val
                    mf_inv_val = 0
                    mf_inv_amount = 0
                    mf_unrealized_return = 0
                    mf_total_units = 0


                    # mf_realized_return = mf_inv_amount
            elif(flag_nifty500==False):
                # mf_inv_amount = mf_inv_amount+mf_inv_amount_previous+ cash_eq
                # print(mf_inv_amount)
                # mf_inv_amount = 138803.74
                # mf_total_units = 69.777
                mf_inv_amount,mf_total_units,mf_inv_val,mf_return_val,mf_return_per,mf_unrealized_return,mf_current_nav,mf_name,mf_todays_return = mutual_funds_calc(cash_eq,mf_total_units,mf_inv_amount)
                cash_eq = 0
                # mf_realized_return = 0

            inv_value,return_amount,return_per,unrealized_return,realized_return,todays_return,todays_return_per = momentum_calculations(df_portfolio,inv_amount,df_sell_total)
            # print(mf_inv_val)
            ### mf_fund calculation values

            user_obj['cash_eq'] = cash_eq

            user_obj['mf_name'] = mf_name
            user_obj['mf_inv_amount'] = mf_inv_amount
            user_obj['mf_total_units'] = mf_total_units
            user_obj['mf_inv_val'] = mf_inv_val
            user_obj['mf_return_val'] = mf_return_val
            user_obj['mf_current_nav'] = mf_current_nav
            if(mf_inv_amount != 0):
                user_obj['mf_return_per'] = ((mf_inv_val-mf_inv_amount)/mf_inv_amount)*100
            else:
                user_obj['mf_return_per'] = 0

            user_obj['mf_unrealized_return'] = mf_unrealized_return
            user_obj['mf_realized_return'] = mf_realized_return
            if(mf_inv_amount!=0):
                user_obj['mf_avg_nav'] = mf_inv_amount/mf_total_units
            else:
                user_obj['mf_avg_nav'] = 0



            user_obj['mf_todays_return'] = mf_todays_return

            # user_obj['mf_return_per'] = mf_return_per

            ### stocks calculation values
            # user_obj['stocks_return_val'] = return_amount
            # user_obj['stocks_realized_return'] = realized_return
            # user_obj['stocks_unrealized_return'] = unrealized_return

            ## Portfolio Calculation Values
            user_obj['inv_amount'] = inv_amount
            user_obj['inv_val'] = inv_value+mf_inv_val+cash_eq
            user_obj['total_return_val'] = user_obj['inv_val'] - user_obj['inv_amount']
            user_obj['total_return_per'] = (user_obj['total_return_val']/inv_amount)*100
            # user_obj['total_realized_return'] = realized_return+mf_realized_return
            user_obj['total_unrealized_return'] = unrealized_return+mf_unrealized_return
            user_obj['total_unrealized_return_per'] = (user_obj['total_unrealized_return']/user_obj['inv_amount'])*100
            user_obj['total_realized_return'] = user_obj['total_return_val'] - user_obj['total_unrealized_return']
            user_obj['total_realized_return_per'] = (user_obj['total_realized_return']/user_obj['inv_amount'])*100
            user_obj['todays_return'] = todays_return
            user_obj['todays_return_per'] = todays_return_per

            ##inv_val_index,total_return_index,total_return_per_index
            user_obj['inv_amount_index'] = inv_amount_index
            user_obj['inv_val_index'] = inv_val_index
            user_obj['total_return_index'] = total_return_index
            user_obj['total_return_per_index'] = total_return_per_index
            user_obj['total_unrealized_return_index'] = total_return_index
            user_obj['total_unrealized_return_index_per'] = total_return_per_index
            user_obj['total_realized_return_index'] = np.nan
            user_obj['total_realized_return_index_per'] = np.nan
            user_obj['today_return_index'] = today_return_index
            user_obj['today_return_index_per'] = today_return_per_index


            # mf_inv_amount,mf_total_units,mf_inv_val,mf_return_val,mf_return_per
            print("cash Eq: "+str(cash_eq))
            print("mf_inv_amount: "+str(mf_inv_amount))
            print("mf_total_units: "+str(mf_total_units))
            print("mf_inv_val: "+str(mf_inv_val))
            print("mf_return_val: "+str(mf_return_val))
            print("mf_unrealized_return: "+str(mf_unrealized_return))
            print("mf_realized_return: "+str(mf_realized_return))

            print("Inv Amount: "+str(user_obj['inv_amount']))
            print("Inv Value: "+str(user_obj['inv_val']))
            print("Total Return: "+str(user_obj['total_return_val']))
            print("return_per: "+str(user_obj['total_return_per']))
            print("unrealized_return "+str(user_obj['total_unrealized_return']))
            print("unrealized_return_per "+str(user_obj['total_unrealized_return_per']))
            print("realized_return "+str(user_obj['total_realized_return']))
            print("realized_return_per "+str(user_obj['total_realized_return_per']))
            print("todays_return "+str(user_obj['todays_return']))
            print("todays_return_per "+str(user_obj['todays_return_per']))

            ##inv_val_index,total_return_index,total_return_per_index
            print("inv_amount_index "+str(user_obj['inv_amount_index']))
            print("inv_val_index "+str(user_obj['inv_val_index']))
            print("total_return_index "+str(user_obj['total_return_index']))
            print("total_return_per_index "+str(user_obj['total_return_per_index']))
            print("total_unrealized_return_index "+str(user_obj['total_unrealized_return_index']))
            print("total_unrealized_return_index_per "+str(user_obj['total_unrealized_return_index_per']))
            print("total_realized_return_index "+str(user_obj['total_realized_return_index']))
            print("total_realized_return_index_per "+str(user_obj['total_realized_return_index_per']))
            print("today_return_index "+str(user_obj['today_return_index']))
            print("today_return_index_per "+str(user_obj['today_return_index_per']))


            user_obj = float_to_decimal_convert(user_obj)
            list_res_obj.append(user_obj)
            # write_to_dynamodb(dynamo_momentum_user_main,user_obj,user_obj['timestamp'])

    for obj_dynamo in list_res_obj:
        write_to_dynamodb(dynamo_momentum_user_main,obj_dynamo,obj_dynamo['timestamp'])

latest_tuesday_date = latest_tuesday_func(datetime.now())
main_calc(dynamo_momentum_user_main)
# mutual_funds_calc(100000,latest_tuesday_date)
