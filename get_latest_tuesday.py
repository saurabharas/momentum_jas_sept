import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta, TU
from datetime import date
from datetime import timedelta

'''
    given a date gives u the latest tuesday for that date
'''
def latest_tuesday_func(latest_date):
    today = latest_date
    offset = (today.weekday() - 1) % 7
    last_tuesday = today - timedelta(days=offset)
    last_tuesday_string = datetime.strftime(last_tuesday,'%Y-%m-%d')
    # print(last_tuesday_string)
    return last_tuesday_string

def next_tuesday_func(latest_date):
    if(latest_date.weekday()!=1):
        today = latest_date
        offset = (today.weekday() - 1 ) % 7
        last_tuesday = today - timedelta(days=offset)
        next_tuesday = last_tuesday + timedelta(days=7)
        next_tuesday_string  = datetime.strftime(next_tuesday,'%Y-%m-%d')
        # print(next_tuesday_string)
    else:
        next_tuesday_string = latest_tuesday_func(latest_date)
    return next_tuesday_string
