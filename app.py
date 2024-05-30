# -*- coding: utf-8 -*-
"""
Dashboard for GAW QC tool, based on Dash

Historical and near-real time CAMS data are read from a sqlite database
Data to analyze can be uploaded by the user in the form of a xls or csv file

Requires folder 'assets' containing logos and css file

@author: bryu
"""

from dash import Dash, dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from pandas.core.tools.datetimes import guess_datetime_format
from datetime import date, datetime, timedelta
import calendar
import statsmodels.api as sm
from pyod.models.lof import LOF
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
import sqlite3
import base64
from io import StringIO, BytesIO
import glob
import warnings
warnings.filterwarnings('ignore')



# Path to database
inpath = 'test.db'

# Path to log file
logpath = 'log.csv'

# Minimum number of hourly values required to calculate monthly means
n_min = 300

# Sub-LOF parameters
window_size = 5
n_neighbors = 100

# SARIMAX parameter (confidence level probability for the shaded area in the plot)
p_conf = 0.01

# CAMS-based anomaly score parameter (minimum flaggable window size in hours)
window_size_cams = 50

# Threshold parameters (thr0 + incr * strictness level)
thr0_lof = 0.994
incr_lof = 0.002
thr0_cams = 0.97
incr_cams = 0.01

# Define LOF instance
subLOF = LOF(n_neighbors=n_neighbors, metric='euclidean')

# Define regression model
ml_model = ExtraTreesRegressor(criterion='squared_error', n_estimators=200, 
                               max_depth=15, max_features=None, 
                               min_samples_leaf=20, random_state=42, n_jobs=-1)



# Define functions

def read_data(db_file, gaw_id, v, h):
    """ read data for the target station from the database
    :param db_file: Path of database file
    :param gaw_id: GAW id of the target station
    :param v: Variable (one of ch4, co, co2, o3)
    :param h: Height from the ground
    :return: Data frame of hourly or monthly data, and a string giving the time resolution
    """
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute('SELECT id FROM series WHERE ' + \
                        'gaw_id=? AND variable=? AND height=?', 
                    (gaw_id, v, h))
        series_id = cur.fetchone()[0]
        cur.execute('SELECT time,value,n_meas FROM gaw_hourly WHERE ' + \
                        'series_id=?', (series_id,))
        data_gaw = cur.fetchall()        
        if len(data_gaw) > 0:
            names_gaw = [x[0] for x in cur.description]
            if v != 'co2':
                cur.execute('SELECT * FROM cams_hourly WHERE ' + \
                                'series_id=?', (series_id,))
                data_cams = cur.fetchall()
                names_cams = [x[0] for x in cur.description]
            res = 'hourly'
        else:
            cur.execute('SELECT time,value,n_meas FROM gaw_monthly WHERE ' + \
                            'series_id=?', (series_id,))
            data_gaw = cur.fetchall()
            names_gaw = [x[0] for x in cur.description]
            if v != 'co2':
                cur.execute('SELECT * FROM cams_monthly WHERE ' + \
                                'series_id=?', (series_id,))
                data_cams = cur.fetchall()
                names_cams = [x[0] for x in cur.description]
            res = 'monthly'
            
    out_gaw = pd.DataFrame(data_gaw, columns=names_gaw)
    out_gaw.rename(columns={'value':v}, inplace=True)
    
    if v == 'co2':
        out = out_gaw
    elif len(data_cams) == 0:
        raise Exception('CAMS data not found for this station')
    else:
        out_cams = pd.DataFrame(data_cams, columns=names_cams)
        out_cams['value'].interpolate(inplace=True) # increase time resolution through linear interpolation
        out_cams.rename(columns={'value':v+'_cams', 'value_tc':'tc_'+v}, inplace=True)
        out_cams.drop(columns=['id','series_id'], inplace=True)
        out = pd.merge(out_gaw, out_cams, how='outer', on='time')
        
    out['time'] = pd.to_datetime(out['time'], format='ISO8601')
    out['hour'] = out['time'].dt.hour
    out['doy'] = out['time'].dt.dayofyear
    out['trend'] = np.arange(out.shape[0])
    out.set_index('time', inplace=True)
    
    return out, res


def filter_data(df, res, thr_h, thr_m, w):
    """ assign NaN to hourly/monthly means based on an insufficient number of measurements
    :param df: Data frame produced by read_data
    :param res: temporal resolution ('hourly' or 'monthly')
    :param thr_h: Minimum fraction of measurements required in an hour with respect of a moving maximum 
    :param thr_m: Minimum number of days required in a month
    :param w: Window size for moving maximum in hourly data
    :return: Data frame with NaNs in place of data based on insufficient measurements (column n_meas is dropped)
    """
    par = df.columns[0]
    if res == 'hourly':
        running_max = df['n_meas'].rolling(w, min_periods=1, center=True).max()
        df.loc[df['n_meas']<thr_h*running_max, par] = np.nan
    else:
        df.loc[(df['n_meas']>0) & (df['n_meas']<thr_m), par] = np.nan
        
    return df.drop(columns=['n_meas'])


def read_meta(db_file):
    """ return the metadata of the stations that have data
    :param db_file: Path of database file
    :return: Data frame of metadata
    """
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM stations WHERE ' + \
                        'gaw_id IN (' + \
                            'SELECT gaw_id FROM series)')
        meta = cur.fetchall()
        names = [x[0] for x in cur.description]
        
    return pd.DataFrame(meta, columns=names)


def read_series(db_file, gaw_id):
    """ return the available variables and heights for a given station
    :param db_file: Path of database file
    :param gaw_id: GAW id of the target station
    :return: Data frame with columns 'variable' and 'height'
    """
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute('SELECT variable,height FROM series WHERE ' + \
                        'gaw_id=?', (gaw_id,))
        series = cur.fetchall()
        names = [x[0] for x in cur.description]
        
    return pd.DataFrame(series, columns=names)


def read_start_end(db_file, gaw_id, v, h):
    """ return the first and last date of a series
    :param db_file: Path of database file
    :param gaw_id: GAW id of the target station
    :param v: Variable (one of ch4, co, co2, o3)
    :param h: Height from the ground
    :return: Tuple of two dates
    """
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute('SELECT id FROM series WHERE ' + \
                        'gaw_id=? AND variable=? AND height=?', 
                    (gaw_id, v, h))
        series_id = cur.fetchone()[0]
        cur.execute('SELECT MIN(time),MAX(time) FROM gaw_hourly WHERE ' + \
                        'series_id=?', (series_id,))
        dates = cur.fetchall()[0]
        
        if dates[0] is None:
            cur.execute('SELECT MIN(time),MAX(time) FROM gaw_monthly WHERE ' + \
                            'series_id=?', (series_id,))
            dates = cur.fetchall()[0]     
            last_day = calendar.monthrange(int(dates[1][:4]),int(dates[1][5:7]))[1]
            return (dates[0]+'-01', dates[1]+'-'+str(last_day))
        
        else:   
            return (dates[0][:10], dates[1][:10])
        
        
def parse_data(content, filename, par, tz):
    """ parse the data uploaded by the user
    :param content: content of uploaded file (output of upload-data button)
    :param filename: name of uploaded file (output of upload-data button)
    :param par: variable code (output of param-dropdown)
    :param tz: time zone (output of timezone-dropdown)
    :return: Data frame of parsed data with time as index
    """
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if (filename[-3:] == 'csv') | (filename[-3:] == 'txt'):
            df_up = pd.read_csv(StringIO(decoded.decode('utf-8')), sep=None, 
                                header=None, skiprows=1, usecols=[0,1], quoting=3,
                                names=['Time',par], engine='python')
        else:
            df_up = pd.read_excel(BytesIO(decoded), header=None, skiprows=1,
                                  usecols=[0,1], names=['Time',par])            
    except:
        print('Could not recognize file format')
        return []
    df_up.replace('"', '', regex=True, inplace=True) # get rid of quotes
    
    # Deal with time format
    fmt = guess_datetime_format(df_up['Time'].iloc[0])
    try:
        df_up['Time'] = df_up['Time'].apply(datetime.strptime, args=(fmt,))
    except:
        try:
            if fmt[1] == 'm': # change mdy format to dmy
                fmt = fmt.replace('d', 'm')
                fmt = list(fmt)
                fmt[1] = 'd'
                fmt = ''.join(fmt)
            df_up['Time'] = df_up['Time'].apply(datetime.strptime, args=(fmt,))
        except:
            print('Could not recognize time format')
            return []
    df_up['Time'] = df_up['Time'].dt.round('H') # round to nearest hour
    if tz != 'UTC':
        df_up['Time'] = to_utc(df_up['Time'], tz) # convert time to UTC
    df_up.sort_values(['Time'], inplace=True) # sort chronologically
    df_up.set_index('Time', inplace=True)
    
    # Deal with decimal separator and missing values
    try:
        df_up[par] = pd.to_numeric(df_up[par].replace(',', '.', regex=True))
    except:
        print('Could not convert data column to numeric')
        return []
    df_up.loc[df_up[par]<0, par] = np.nan # assign NaN to all negative values
    
    return df_up
        
        
def to_utc(series, tz, reverse=False):
    """ convert local time to UTC and vice versa
    :param series: Series of times to be converted
    :param tz: Local time zone as 'UTC+xx:xx' or 'UTC-xx:xx'
    :param reverse: Set to True to covert from UTC to local
    :return: Series of converted times
    """
    k = -1 if reverse else 1
    if tz[3] == '-':    
        series = series + k*timedelta(hours=int(tz[4:6]), minutes=int(tz[7:9]))
    elif tz[3] == '+':
        series = series - k*timedelta(hours=int(tz[4:6]), minutes=int(tz[7:9]))
    return series


def write_log(log_file, row):
    """ write log about user activity
    :param db_file: Path of log file
    :param row: Row to add to log (list)
    :return: None
    """
    log = pd.DataFrame(dict(time=row[0], gaw_id=row[1], variable=row[2], height=row[3],
                            start=row[4], end=row[5], upload=row[6]),
                       index=[0])
    h = True if len(glob.glob(log_file)) == 0 else False      
    log.to_csv(log_file, header=h, index=False, mode='a', date_format='%Y-%m-%d')


def aggregate_scores(scores, w_size): 
    """ assign score to each point as the average of the scores of the windows it is in
    :param scores: Series produced by run_sublof
    :param w_size: Window size (integer)
    :return: Series of aggregated scores
    """
    scores = add_missing(pd.DataFrame(scores))
    added_times = pd.date_range(scores.index[-1]+timedelta(hours=1), 
                                scores.index[-1]+timedelta(hours=w_size-1), freq='H')
    scores = pd.concat([scores, pd.Series(np.nan,index=added_times)])
    
    return scores.rolling(w_size, min_periods=1).mean()


def run_sublof(series, model, win): 
    """ run Sub-LOF algorithm
    :param series: Data series to analyze (must be complete - no times missing)
    :param model: Instance of the LOF model to use
    :param win: Window size (integer)
    :return: Series of anomaly scores
    """
    input_data = pd.DataFrame(sliding_window_view(series.values, window_shape=win))
    is_nan = input_data.isna().any(axis=1)
    input_data = input_data[~is_nan]
    times = series[:-win+1].index[~is_nan]
    model.fit(input_data)
    scores = pd.Series(model.decision_scores_, index=times)
    if is_nan.iloc[0]: # restore initial missing value
        scores = pd.concat([pd.Series(np.nan,index=series.index[:1]), scores])
    if is_nan.iloc[-1]: # restore final missing value
        scores = pd.concat([scores, pd.Series(np.nan,index=series.index[-win:-win+1])])
    scores = aggregate_scores(scores, win)
    
    return scores


def add_missing(timeseries):
    """ fill missing times with NaN for hourly data
    :param timeseries: Data frame with hourly resolution (time as index)
    :return: Filled in data frame
    """
    idx = pd.date_range(timeseries.index[0], timeseries.index[-1], freq='H')
    idx = pd.Series(idx)
    if timeseries.shape[0] < idx.size:
        empty_df = pd.DataFrame(np.empty([idx.size,timeseries.shape[1]]))
        empty_df.columns = timeseries.columns
        empty_df.index = idx
        empty_df.iloc[:,:] = np.nan
        empty_df = empty_df[~empty_df.index.isin(timeseries.index)]
        out = pd.concat([timeseries, empty_df]).sort_index()
    else:
        out = timeseries
        
    return out


def monthly_means(timeseries, n_min):
    """ calculate monthly means from hourly data (require at least n_min values per month)
    :param timeseries: Data frame with one variable and time as index
    :param n_min: Integer
    :return: Data frame of monthly means
    """
    out = timeseries.groupby(pd.Grouper(freq='1M',label='left')).mean().round(2)
    n = timeseries.groupby(pd.Grouper(freq='1M',label='left')).count()
    out[n<n_min] = np.nan
    out.index = out.index + timedelta(days=1) # first day of the month as index
    
    return out


def forecast_sm(model, n, pp, idx): 
    """ make prediction using a SARIMA model
    :param model: Instance of an already fitted SARIMA model
    :param n: Number of time steps to predict
    :param pp: Proability level to use for the confidence interval
    :param idx: Time index of the full time series considered (including part to be predicted)
    :return: Prediction and confidence interval series
    """
    future_fcst = model.get_forecast(n)
    conf = future_fcst.conf_int(alpha=pp)
    conf.index = idx[-n:]
    fore = future_fcst.predicted_mean
    fore.index = idx[-n:]
    
    return fore, conf


def debias(df_train, cams, par):
    """ debias monthly CAMS data using linear regression (one model for each calendar month)
    :param df_train: Data frame of training data (containing both measurements and CAMS data, with time as index)
    :param cams: Data frame of CAMS data to debias (time as index)
    :param par: Variable to debias (defines the column names)
    :return: Data frame of debiased CAMS data
    """
    LR = LinearRegression()
    df_diff = df_train[par] - df_train[par+'_cams']
    df_diff.dropna(inplace=True)
    months = np.unique(cams.index.month)
    for m in months: # fit a LR model for each month
        df_m = df_diff[df_diff.index.month==m]
        X = df_m.index.year.to_numpy().reshape(-1, 1)
        LR.fit(X, df_m)
        X = cams[cams.index.month==m].index.year.to_numpy().reshape(-1, 1)
        cams[cams.index.month==m] = cams[cams.index.month==m] + LR.predict(X)
        
    return cams


def downscaling(df_train, df_val, par, w, model):
    """ dowscaling algorithm for CAMS forecasts; the anomaly score is calculated as the moving median of the prediction error
    :param df_train: Data frame of training data (containing both measurements and CAMS data, with time as index)
    :param df_val: Same as df_train but for the target period
    :param par: Variable to debias (defines the column names)
    :param w: Size of the moving window used to calculate the anomaly score (integer)
    :param model: Instance of a sklearn regression model
    :return: Series of downscaled data for the target period (hourly and monthly), series of the anomaly score
    """
    df_train_cams = df_train.dropna()
    df_val_cams = df_val.drop(par,axis=1).dropna()
    model.fit(df_train_cams.drop(par,axis=1).to_numpy(), df_train_cams[par].to_numpy())
    y_pred = model.predict(df_val_cams.to_numpy())
    y_pred = pd.Series(y_pred, index=df_val_cams.index).reindex(index=df_val.index)
    
    # Monthly data (for SARIMA plot)  
    y_pred_mon = y_pred.groupby(pd.Grouper(freq='1M',label='left')).mean()
    y_pred_mon.index = y_pred_mon.index + timedelta(days=1)

    # Anomaly score
    x_to_predict = pd.concat([df_train_cams.drop(par,axis=1), df_val_cams]).sort_index()
    y_pred_all = model.predict(x_to_predict.to_numpy())
    y_pred_all = pd.Series(y_pred_all, index=x_to_predict.index)
    y_to_compare = pd.concat([df_train_cams[par], 
                              df_val.loc[df_val.index.isin(df_val_cams.index), par]]).sort_index()
    errors = y_pred_all - y_to_compare
    all_times = pd.concat([df_train, df_val]).sort_index().index
    errors = errors.reindex(index=all_times[~all_times.duplicated()])
    errors_train = errors[errors.index.isin(df_train_cams.index)]
    diff_series = errors - np.median(errors_train)
    anom_score = diff_series.rolling(w, min_periods=int(w/2)).median()
    
    return y_pred, y_pred_mon, anom_score


def empty_plot(msg): 
    """ return an empty plot containing a text message
    :param msg: Message to print in the plot area
    :return: Figure object
    """
    fig = go.Figure()
    fig.update_layout(
        annotations = [{'text':msg,
                        'xref':'paper',
                        'yref':'paper',
                        'showarrow':False,
                        'font':{'size':28}}])
    fig.update_xaxes(showticklabels=False, zeroline=False, showgrid=False, 
                     linecolor='black', mirror=True)
    fig.update_yaxes(showticklabels=False, zeroline=False, showgrid=False, 
                     linecolor='black', mirror=True)
    
    return fig


def empty_subplot(fig, msg, row, col):
    """ return an empty subplot containing a text message
    :param fig: Plotly figure object
    :param msg: Message to print in the plot area
    :param row: Row number
    :param col: Column number
    :return: Plotly figure object
    """
    fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3], mode='markers+text', text=['',msg,''], 
                             textfont_size=28, marker_opacity=0, showlegend=False), 
                  row=row, col=col)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=col)
    
    return fig


def add_logo(fig, x, y, sx, sy):
    """ Add Empa logo to a figure
    :param fig: Plotly figure object
    :param x: Horizonal position as fraction of x axis
    :param y: Vertical position as fraction of y axis
    :param sx: Horizonal size as fraction of x axis
    :param sy: Vertical size as fraction of y axis
    :return: Plotly figure object
    """
    fig.add_layout_image(
        dict(source='assets/Logo_Empa.png',
             xref='paper', yref='paper', x=x, y=y, sizex=sx, sizey=sy,
             xanchor='left', yanchor='bottom', layer='above', opacity=0.3))
    
    return fig



### DASH APP ###

app = Dash(external_stylesheets=[dbc.themes.MATERIA], 
           title='GAW-qc', update_title='Loading...')

metadata = read_meta(inpath)

app.layout = html.Div([
    
    # Intro
    html.Div([
        html.Div([
            html.Div([
                html.Img(src='assets/Logo_Empa.png'),
                html.Img(src='assets/wmo-gaw.png',
                         style={'padding-left':'125px', 'width':'400px'})                  
                ], style={'display':'flex', 'align-items':'center', 'justify-content':'center',
                          'margin-bottom':'25px'}),
            html.Div([
                html.H1('GAW-qc (beta)'),
                html.P(html.B('An interactive visualization and quality control tool for atmospheric composition measurements of CH4, CO, CO2, and O3'),
                       style={'font-size':'20px', 'margin-bottom':'25px'}),
                html.P('On this page you can visualize the measurements of methane, carbon dioxide, carbon monoxide, ' +
                       'and ozone that are available in the database of the Global Atmosphere Watch (GAW) World Data Centres, ' +
                       'and compare them with model data by the Copernicus Atmosphere Monitoring Service (CAMS; from 2020 onward).'),
                html.P('You can also upload recent measurements from a GAW station to check their quality. ' +
                       'Any anomaly in the measurements will be highlighted by a data-driven algorithm based on historical and CAMS data.'),
                html.P([
                    'For more information see the ',
                    html.A('wiki', href='https://github.com/ybrugnara/gaw-qc/wiki', target='_blank'),
                    '.'
                    ])
                ], style={'font-size':'16px', 'margin-top':'50px'})
            ], style={'text-align':'center'}),
        html.Div([
            html.Div([
                dbc.Button('Data credits', id='open-credits', n_clicks=0, color='info', size='sm'),
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle('Data credits')),
                    dbc.ModalBody([
                        html.P([
                            html.B('Measurements of CH4, CO, and CO2 '),
                            'are provided by the ',
                            html.B('World Data Centre for Greenhouse Gases (WDCGG) '),
                            'at JMA. Data users must follow the ',
                            html.A('GAW data policy', 
                                   href='https://gaw.kishou.go.jp/policy/gaw', target='_blank'),
                            '.'
                            ]),
                        html.P([
                            html.B('Measurements of O3 '),
                            'are provided by the ',
                            html.B('World Data Centre for Reactive Gases (WDCRG) '),
                            'at NILU. The data is licensed under a ',
                            html.A('Creative Commons Attribution 4.0 International License', 
                                   href='http://creativecommons.org/licenses/by/4.0/', target='_blank'),
                            '.'
                            ]),
                        html.P([
                            'Some of the plots are generated using ',
                            html.A('Copernicus Atmosphere Monitoring Service',
                                   href='https://atmosphere.copernicus.eu/', target='_blank'),
                            ' information and/or contain modified Copernicus Atmosphere Monitoring Service information [2024].'
                            ])
                        ])
                    ], id='modal-credits', size='lg', is_open=False),
                ]),
            html.Div([
                dbc.Button('Acknowledgments', id='open-acknowl', n_clicks=0, color='info', size='sm'),
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle('Acknowledgments')),
                    dbc.ModalBody([
                        html.P([
                            'This app was developed by the ',
                            html.A('Quality Assurance / Scientific Activity Centre (QA/SAC) Switzerland',
                                   href='https://www.empa.ch/web/s503/qa-sac-switzerland',
                                   target='_blank'),
                            ' at Empa, which is supported by the Federal Office of Meteorology MeteoSwiss',
                            ' in the framework of the Global Atmosphere Watch Programme of the World Meteorological Organization.'
                            ]),
                        html.P([
                            'Use of the output is free within the limits of the underlying data policies; ',
                            'in any case, appropriate credit must be given to QA/SAC Switzerland.'
                            ]),
                        html.P([
                            'For additional information please contact ',
                            html.A('martin.steinbacher@empa.ch', href='mailto:martin.steinbacher@empa.ch')
                            ]),
                        ]),
                    ], id='modal-acknowl', size='lg', is_open=False),                 
                ], style={'padding-left':'25px'}),
            ], style={'display':'flex', 'margin-top':'25px', 
                      'align-items':'center', 'justify-content':'center'}),
        ], style={'width':'800px', 'margin-left':'auto', 'margin-right':'auto'}),
                      
    # Input form
    html.Div([
        html.Div([
            html.Label(html.B('1. Select the station and gas'))
            ], style={'text-align':'center', 'font-size':'20px'}),
        html.Div([
            html.Div([
                dcc.Dropdown(metadata[['name','gaw_id']].set_index('gaw_id') \
                                 .sort_values(by='name').to_dict()['name'], 
                             id='station-dropdown', placeholder='Select station',
                             style={'font-size':'16px'})
                ], style={'width':'200px'}),
            html.Div([
                dcc.Dropdown(id='param-dropdown', placeholder='Select parameter',
                             style={'font-size':'16px'})
                ], style={'width':'200px', 'padding-left':'25px'}),
            html.Div([
                dcc.Dropdown(id='height-dropdown', placeholder='Select height (m)',
                             style={'font-size':'16px'})
                ], style={'width':'200px', 'padding-left':'25px'}),
            ], style={'display':'flex', 'margin-top':'25px', 
                      'align-items':'center', 'justify-content':'center'}),
        html.Div([
            html.P(html.B('2. Select a period to analyze or upload your own data (hourly means)'))
            ], style={'font-size':'20px', 'margin-top':'50px', 'text-align':'center'}),            
        html.Div([
            dcc.Loading(
                id='loading-dates',
                type='circle',
                fullscreen=False,
                children=dcc.DatePickerRange(id='date-range', updatemode='bothdates')
                )
            ], style={'display':'flex', 'margin-top':'25px', 'align-items':'center', 
                      'justify-content':'center'}),
        html.Div([
            html.Label('(a maximum period length of one year can be selected)')
            ], style={'text-align':'center', 'margin-top':'10px', 'font-size':'12px'}),
        html.Div([
            html.Label('or', style={'font-size':'18px'})
            ], style={'display':'flex', 'margin-top':'25px', 'align-items':'center', 
                      'justify-content':'center'}),
        html.Div([
            dcc.Dropdown(['UTC',
                          'UTC-11:00','UTC-10:00','UTC-09:00','UTC-08:00','UTC-07:00',
                          'UTC-06:00','UTC-05:00','UTC-04:00','UTC-03:00','UTC-02:00',
                          'UTC-01:00','UTC+01:00','UTC+02:00','UTC+03:00',
                          'UTC+04:00','UTC+05:00','UTC+06:00','UTC+07:00','UTC+08:00',
                          'UTC+09:00','UTC+10:00','UTC+11:00','UTC+12:00'],
                         id='timezone-dropdown', placeholder='Choose a time zone',
                         style={'font-size':'16px', 'width':'250px', 'heigth':'50px'}),
            html.Label('and', style={'font-size':'18px', 'padding-left':'25px'}),
            dcc.Upload(children=dbc.Button('Upload File (.csv/.xls)*',
                                           style={'font-size':'18px', 'width':'250px', 
                                                   'height':'50px', 'margin-left':'25px'}), 
                       id='upload-data')
        ], style={'display':'flex', 'margin-top':'25px', 'align-items':'center', 
                  'justify-content':'center'}),
        html.Div([
            html.Label('*The file must have two columns: time and value (hourly mean). It cannot exceed one year of data. An example is given'),
            html.A('here', target='_blank',
                   href='https://raw.githubusercontent.com/ybrugnara/gaw-qc/main/examples/Jungfraujoch_CO2_20240101-20240331.csv',
                   style={'padding-left':'3px'})
            ], style={'text-align':'center', 'margin-top':'10px', 'font-size':'12px'})
    ], className='menu'),
                  
    # Data loading with message
    html.Div([
        html.Label([], id='label-wait'),
        # Data storage
        dcc.Loading(
            id='loading-data',
            type='circle',
            fullscreen=False, 
            children=dcc.Store(id='input-data', data=[], storage_type='memory')
            )
        ], style={'text-align':'center', 'margin-top':'50px'}),

    # Div containing all the plots
    html.Div([
        
        # Lines/points switch
        html.Div([
            html.Label('Lines', 
                       style={'font-size':'14px', 'font-weight':'bold',
                              'margin-top':'3px'}),
            daq.BooleanSwitch(id='points-switch', on=False, color='#e7e7e7',
                              style={'padding-left':'5px'}),
            html.Label('Points', 
                       style={'font-size':'14px', 'font-weight':'bold',
                              'padding-left':'5px', 'margin-top':'3px'}),
            ], style={'display':'none'}, id='div-switch'),
                      
        # Hourly plot
        html.Div([
            html.Div([
                dcc.Loading(
                    dcc.Graph(id='graph-hourly'),
                    id='loading-1',
                    type='circle'
                    )
            ]),
            html.Div([
                html.Div([
                    html.Label('Toggle CAMS', 
                               style={'font-size':'14px', 'padding-right':'20px'}),
                    daq.BooleanSwitch(id='cams-switch-1', on=True)
                    ], style={'display':'flex'}),
                html.Div([
                    html.Label('Strictness', style={'font-size':'14px'}),
                    daq.Slider(
                        min=1,
                        max=3,
                        step=1,
                        value=2,
                        marks={1: {'label':'More flags', 'style':{'font-size':'14px'}},
                               3: {'label':'Fewer flags', 'style':{'font-size':'14px'}}},
                        id='threshold-slider',
                        updatemode='mouseup',
                        handleLabel={'showCurrentValue':True, 'label':'level'},
                        size=250
                        )
                    ], style={'padding-left':'190px'}),
                html.Div([
                    dbc.Button('Export to CSV', id='btn_csv', size='sm'),
                    dcc.Download(id='export-csv'),
                    ], style={'padding-left':'260px'}),
                html.Div([
                    html.Label('Bin size', style={'font-size':'14px'}),
                    daq.Slider(
                        min=0.1,
                        max=10.0,
                        step=0.1,
                        value=1.0,
                        marks={0.1: {'label':'0.1', 'style':{'font-size':'14px'}},
                               10: {'label':'10', 'style':{'font-size':'14px'}}},
                        id='bin-slider',
                        size=250
                        ),
                    ], style={'padding-left':'160px'})        
                ], style={'display':'none'},
                   id='div-hourly-settings'),
             html.Div([
                 dbc.Button('Help', id='info-button-1', color='info', n_clicks=0, size='sm'),
                 dbc.Collapse(
                     dbc.Card(
                         dbc.CardBody([
                             html.P([
                                 html.B('Measurements'),
                                 html.Br(),
                                 'Hourly means of the measurements made at the station for the selected / uploaded period.'
                                 ]),
                             html.P([
                                 html.B('CAMS'), ' (not available for CO2)',
                                 html.Br(),
                                 'Data from the ', 
                                 html.A('Copernicus Atmosphere Monitoring Service global atmospheric composition forecasts',
                                        href='https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts?tab=overview',
                                        target='_blank'),
                                 ', a model simulation that assimilates satellite measurements.',
                                 ' This product can have a large bias, especially for methane (CH4), and is given just for reference.',
                                 ' The original 3-hourly resolution was increased to hourly through linear interpolation.',
                                 ' Note that it is not available for carbon dioxide (CO2) or before 2020.'
                                 ]),
                             html.P([
                                 html.B('CAMS + ML'), ' (not available for CO2)',
                                 html.Br(),
                                 'A statistically improved version of CAMS in which the data are debiased through machine learning (ML).',
                                 ' Periods with significant deviations from the measurements are highlighted by a ',
                                 html.B('yellow shading'),
                                 ' and indicate a potential bias in the measurements.',
                                 ' However, the ML model also has limitations and should not be trusted blindly.',
                                 ]),
                             html.P([
                                 html.B('Yellow flags'),
                                 html.Br(),
                                 'Data that exhibit an anomalous behaviour with respect to historical data.',
                                 ' The flags are based on the anomaly score of the Local Outlier Factor (LOF) method applied to sequences of ' + str(window_size) + ' measurements.'
                                 ]),
                             html.P([
                                 html.B('Red flags'),
                                 html.Br(),
                                 'Data that exhibit a very anomalous behaviour with respect to historical data (see yellow flags),',
                                 ' or yellow flags that occur in a period where measurements are significantly different from CAMS + ML',
                                 ' (i.e., double yellow flags).'
                                 ]),
                             html.P([
                                 html.B('Toggle CAMS'),
                                 html.Br(),
                                 'Remove CAMS and CAMS + ML time series from the plot and from the export file.',
                                 ' To just hide any time series you can click on the respective entry in the legend (they will still be exported).'
                                 ]),
                             html.P([
                                 html.B('Strictness'),
                                 html.Br(),
                                 'The strictness slider allows you to adjust the significance levels required to produce yellow and red flags.',
                                 ' A higher strictness level implies higher probability thresholds and fewer flags.',
                                 ' This is useful if you think that the algorithm is flagging too many measurements or vice versa.'
                                 ]),
                             html.P([
                                 html.B('Pie chart'),
                                 html.Br(),
                                 'The chart on the top-right corner shows the fraction of normal (green), anomalous (yellow), and very anomalous (red) data.',
                                 ' Both yellow flags and yellow shadings are considered anomalous data.',
                                 ]),        
                             html.P([
                                 html.B('Histogram'),
                                 html.Br(),
                                 'The chart on the bottom-right corner shows the distribution of the measurements, CAMS, and CAMS + ML,',
                                 ' using the same color code of the main chart.',
                                 ' It is possible to adjust the size of the bins through a slider.'
                                 ]),
                             html.P([
                                 html.B('Export to CSV'),
                                 html.Br(),
                                 'Download the time series shown in the main chart (flags included) as a Comma Separated Value (CSV) file.'
                                 ]),                         
                             ]),
                         style={'width':'1000px'}),
                     id='collapse-1', is_open=False)
                 ], style={'display':'none'},
                    id='div-hourly-info'),
             ], className='main', style={'display':'none'}, id='div-hourly'),
    
        # Monthly plot
        html.Div([
            html.Div([
                html.Div([
                    dcc.Loading(
                        dcc.Graph(id='graph-monthly'),
                        id='loading-2',
                        type='circle'
                        )
                    ], style={'padding-left':'100px'}),
                html.Div([
                    html.Label('Type of trend', 
                               style={'font-size':'14px', 
                                      'text-decoration':'underline'}),
                    dcc.RadioItems(
                        options=[
                            {'label': 'No trend', 'value': 'n'},
                            {'label': 'Linear', 'value': 'c'},
                            {'label': 'Quadratic', 'value': 't'},
                        ],
                        value='c',
                        inline=False,
                        id='trend-radio',
                        labelStyle={'font-size':'14px'}
                        ),
                    html.Label('Number of years to use', 
                               style={'font-size':'14px', 'text-decoration':'underline',
                                      'margin-top':'50px'}),
                    dcc.Slider(
                        min=4,
                        max=10,
                        step=1,
                        value=7,
                        marks={4: {'label': '4', 'style':{'font-size':'16px'}},
                               5: {'label': '5', 'style':{'font-size':'16px'}},
                               6: {'label': '6', 'style':{'font-size':'16px'}},
                               7: {'label': '7', 'style':{'font-size':'16px'}},
                               8: {'label': '8', 'style':{'font-size':'16px'}},
                               9: {'label': '9', 'style':{'font-size':'16px'}},
                               10: {'label': '10', 'style':{'font-size':'16px'}}},
                        id='length-slider',
                        updatemode='mouseup',
                        vertical=True,
                        verticalHeight=250
                        ), 
                    ], style={'padding-top':'100px'})
                ], style={'display':'flex', 'margin-left':'auto', 'margin-right':'auto'}),
            html.Div([
                html.Div([
                    html.Label('Toggle CAMS', 
                               style={'font-size':'14px', 'padding-right':'20px'}),
                    daq.BooleanSwitch(id='cams-switch-2', on=True)
                    ], style={'display':'flex', 'padding-left':'175px'}),
                html.Div([
                    dbc.Button('Export to CSV', id='btn_csv_monthly', size='sm'),
                    dcc.Download(id='export-csv-monthly'),
                    ], style={'padding-left':'775px'}),
                ], style={'display':'flex'}),
            html.Div([
                dbc.Button('Help', id='info-button-2', color='info', n_clicks=0, size='sm'),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody([
                            html.P([
                                html.B('Measurements'),
                                html.Br(),
                                'Monthly means of the measurements made at the station in recent years (the number of years is controlled by the slider next to the chart).',
                                ' Months with less than ' + str(n_min) + ' valid hourly values are not shown.'
                                ]),
                            html.P([
                                html.B('CAMS'), ' (not available for CO2)',
                                html.Br(),
                                'Data from the ', 
                                html.A('Copernicus Atmosphere Monitoring Service global atmospheric composition forecasts',
                                       href='https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts?tab=overview',
                                       target='_blank'),
                                ', a model simulation that assimilates satellite measurements.',
                                ' This product can have a large bias, especially for methane (CH4), and is given just for reference.',
                                ' Note that it is not available for carbon dioxide (CO2) or before 2020.'
                                ]),
                            html.P([
                                html.B('CAMS + ML'), ' (not available for CO2)',
                                html.Br(),
                                'A statistically improved version of CAMS in which the data are debiased through machine learning (ML).',
                                ' A large difference with the measurements might indicate a systematic bias.',
                                ' However, the ML model also has limitations and should not be trusted blindly.',
                                ]),
                            html.P([
                                html.B('SARIMA'),
                                html.Br(),
                                'Predicted values for the selected / uploaded period using a Seasonal Auto-Regressive Integrated Moving Average model based on the historical data,',
                                ' with ' + str(int(100*(1-p_conf))) + '% confidence range. SARIMA is a purely statistical alternative to CAMS + ML.'
                                ' Only the data shown in the plot are considered in order to fit the model, meaning that the prediction can change by modifying the number of years. Moreover, the',
                                html.B(' type of trend'),
                                ' assumed by the model can be selected by the user (top-right corner).'
                                ]),
                            html.P([
                                html.B('Flags'),
                                html.Br(),
                                'Measurements that are outside the confidence range of the SARIMA prediction are marked with a red circle.'
                                ]),
                            html.P([
                                html.B('Toggle CAMS'),
                                html.Br(),
                                'Remove CAMS and CAMS + ML time series from the plot and from the export file.',
                                ' To just hide any time series you can click on the respective entry in the legend (they will still be exported).'
                                ]),
                            html.P([
                                html.B('Export to CSV'),
                                html.Br(),
                                'Download the data from the selected / uploaded period (flags included) as a Comma Separated Value (CSV) file.'
                                ]),                         
                            ]),
                        style={'width':'1000px'}),
                    id='collapse-2', is_open=False)
                ], style={'display':'flex', 'padding-left':'25px'})
            ], className='main', style={'display':'none'}, id='div-monthly'),
        
        # Other plots   
        html.Div([
            html.Div([
                dcc.Loading(
                    dcc.Graph(id='graph-cycles'),
                    id='loading-3',
                    type='circle'
                    )
                ]),
            html.Div([
                html.Div([
                    html.Label('Number of years to use', style={'font-size':'14px'}),
                    dcc.Slider(
                        min=1,
                        max=7,
                        step=1,
                        value=4,
                        marks={1: {'label':'1', 'style':{'font-size':'16px'}},
                               2: {'label':'2', 'style':{'font-size':'16px'}},
                               3: {'label':'3', 'style':{'font-size':'16px'}},
                               4: {'label':'4', 'style':{'font-size':'16px'}},
                               5: {'label':'5', 'style':{'font-size':'16px'}},
                               6: {'label':'6', 'style':{'font-size':'16px'}},
                               7: {'label':'7', 'style':{'font-size':'16px'}}},
                        id='n-years',
                        updatemode='mouseup'
                        )
                    ], style={'padding-left':'100px', 'width':'250px'}),
                html.Div([
                    dbc.Button('Export to CSV', id='btn_csv_dc', size='sm'),
                    dcc.Download(id='export-csv-dc'),
                    ], style={'padding-left':'90px'}),
                html.Div([
                    dbc.Button('Export to CSV', id='btn_csv_sc', size='sm'),
                    dcc.Download(id='export-csv-sc'),
                    ], style={'padding-left':'355px'}),
                html.Div([
                    dbc.Button('Export to CSV', id='btn_csv_vc', size='sm'),
                    dcc.Download(id='export-csv-vc'),
                    ], style={'padding-left':'355px'})
                ], style={'display':'flex'}),
            html.Div([
                dbc.Button('Help', id='info-button-3', color='info', n_clicks=0, size='sm'),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody([
                            html.P([
                                html.B('Diurnal cycle'),
                                html.Br(),
                                'The first panel compares the mean diurnal cycle in the selected / uploaded period with other years and with the multi-year average.',
                                ' If there is a trend, the curves will be shifted accordingly. However, the shape of the curves should be fairly similar.',
                                ' If not, it might be worth investigating the causes of the differences.',
                                html.Br(),
                                'By hovering over a certain value the number of underlying hourly values is also displayed.'
                                ]),
                            html.P([
                                html.B('Seasonal cycle'),
                                html.Br(),
                                'The second panel compares the monthly mean values in the selected / uploaded period with other years and with the multi-year average.',
                                ' If there is a trend, the curves will be shifted accordingly. However, the shape of the curves should be fairly similar.',
                                ' If not, it might be worth investigating the causes of the differences.',
                                html.Br(),
                                'Months with less than ' + str(n_min) + ' valid hourly values are not shown.'
                                ]),
                            html.P([
                                html.B('Variability cycle'),
                                html.Br(),
                                'The third panel compares the monthly variability in the selected / uploaded period with other years and with the multi-year average.',
                                ' The variability is defined as the standard deviation of the hourly means. All curves should be fairly similar.',
                                ' If not, it might be worth investigating the causes of the differences.',
                                html.Br(),
                                'Months with less than ' + str(n_min) + ' valid hourly values are not shown.'
                                ]),
                            html.P([
                                html.B('Export to CSV'),
                                html.Br(),
                                'Download the data shown in a specific panel as a Comma Separated Value (CSV) file.'
                                ]),                         
                            ]),
                        style={'width':'1000px'}),
                    id='collapse-3', is_open=False)
                ], style={'display':'flex', 'padding-left':'25px'})
            ], className='main', style={'display':'none'}, id='div-cycles')
        
        ], style={'margin-top':'50px', 'margin-bottom':'25px'}),
    
    html.Div([
        html.A('Report bug', href='mailto:yuri.brugnara@empa.ch')
        ], style={'text-align':'center', 'font-size':'12px', 'margin-bottom':'25px'})

    ], style={'width':'1600px', 'margin-left':'auto', 'margin-right':'auto'})


    
# Callbacks

@app.callback(Output('modal-credits', 'is_open'),
              [Input('open-credits', 'n_clicks')],
              [State('modal-credits', 'is_open')])
def toggle_credits(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(Output('modal-acknowl', 'is_open'),
              [Input('open-acknowl', 'n_clicks')],
              [State('modal-acknowl', 'is_open')])
def toggle_acknowledgments(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(Output('param-dropdown', 'options', allow_duplicate=True),
          Output('param-dropdown', 'value', allow_duplicate=True),
          Input('station-dropdown', 'value'),
          prevent_initial_call=True)
def update_variables(stat):
    if stat is None:
        raise PreventUpdate
        
    meta = read_series(inpath, stat)
    pars = map(str.upper, np.sort(np.unique(meta['variable'])))
    
    return list(pars), None


@callback(Output('height-dropdown', 'options', allow_duplicate=True),
          Output('height-dropdown', 'value', allow_duplicate=True),
          Input('station-dropdown', 'value'),
          Input('param-dropdown', 'value'),
          Input('height-dropdown', 'value'),
          prevent_initial_call=True)
def update_heights(stat, par, hei):
    if (stat is None) | (par is None):
        raise PreventUpdate
        
    meta = read_series(inpath, stat)
    heights = np.sort(np.unique(meta['height'][meta['variable']==par.lower()]))
    if hei in heights:
        val = hei
    elif len(heights) == 1:
        val = heights.item()
    else:
        val = None
    
    return list(heights), val


@callback(Output('date-range', 'min_date_allowed'),
          Output('date-range', 'max_date_allowed'),
          Output('date-range', 'initial_visible_month'),
          Output('loading-dates', 'display', allow_duplicate=True),
          Input('station-dropdown', 'value'),
          Input('param-dropdown', 'value'),
          Input('height-dropdown', 'value'),
          prevent_initial_call=True)
def update_dates(cod, par, hei):
    if (cod is None) | (par is None) | (hei is None):
        raise PreventUpdate
        
    dates = read_start_end(inpath, cod, par.lower(), hei)
    last_year = int(dates[1][:4])
    
    return datetime.strptime(dates[0], '%Y-%m-%d'), \
        datetime.strptime(dates[1], '%Y-%m-%d'), date(last_year, 1, 1), 'auto'
        
        
@callback(Output('timezone-dropdown', 'value', allow_duplicate=True),
          Input('timezone-dropdown', 'value'),
          Input('upload-data', 'contents'),
          prevent_initial_call=True)
def update_tz(tz, content):
    if (tz != None) | (content is None):
        raise PreventUpdate
        
    return 'UTC'


@callback(Output('label-wait', 'children'),
          Output('label-wait', 'style'),
          Output('loading-dates', 'display', allow_duplicate=True),
          Input('station-dropdown', 'value'),
          Input('param-dropdown', 'value'),
          Input('height-dropdown', 'value'),
          Input('date-range', 'end_date'),
          Input('timezone-dropdown', 'value'),
          Input('upload-data', 'contents'),
          prevent_initial_call=True)
def wait_message (cod, par, hei, date1, tz, content):
    if (cod is None) | (par is None) | (hei is None) | \
        ((date1 is None) & ((tz is None) | (content is None))):
        return '', {'display':'none'}, 'auto'
    else:
        return 'Please wait - Dashboard is loading...', \
            {'font-size':'18px', 'font-weight': 'bold', 'margin-bottom':'25px'}, \
            'hide'


@callback(Output('input-data', 'data'),
          Output('date-range', 'start_date', allow_duplicate=True),
          Output('date-range', 'end_date', allow_duplicate=True),
          Output('upload-data', 'contents', allow_duplicate=True),
          Input('station-dropdown', 'value'),
          Input('param-dropdown', 'value'),
          Input('height-dropdown', 'value'),
          Input('date-range', 'start_date'),
          Input('date-range', 'end_date'),
          Input('timezone-dropdown', 'value'),
          Input('upload-data', 'contents'),
          State('upload-data', 'filename'),
          prevent_initial_call=True)
def update_data(cod, par, hei, date0, date1, tz, content, filename):
    if (cod is None) | (par is None) | (hei is None) | \
        ((date1 is None) & ((tz is None) | (content is None))):
        raise PreventUpdate
        
    par = par.lower()
    
    if content is None:
        time0 = datetime.strptime(date0, '%Y-%m-%d')
        time1 = datetime.strptime(date1, '%Y-%m-%d')
        is_new = False
    else:
        # Parse uploaded data
        df_up = parse_data(content, filename, par, tz)
        time0 = df_up.index[0]
        time1 = df_up.index[-1]
        is_new = True
        
    # Limit to one year of data (retain most recent year)
    leap_years = np.arange(1904, 2100, 4)
    if ((time1.year in leap_years) & (time1.month > 2)) | \
        ((time1.year-1 in leap_years) & (time1.month <= 2)) | \
        ((time1.year in leap_years) & (time1.month == 2) & (time1.day==29)):
        n_max = 366
    else:
        n_max = 365
    if (time1-time0).days > n_max-1:
        time0 = time1 - timedelta(days=n_max) + timedelta(hours=1)
        if content != None:
            df_up = df_up[df_up.index >= time0]
    
    # Read data and remove values based on insufficient measurements
    df, res = read_data(inpath, cod, par, hei)
    df = filter_data(df, res, 0.25, 15, 5)
    
    # Calculate monthly means of uploaded data
    if (res == 'monthly') & (content != None):
        df_up = df_up.groupby(pd.Grouper(freq='1M',label='left')).mean()
        df_up.index = df_up.index + timedelta(days=1)
            
    # Merge uploaded data with historical data
    if content != None:
        if par == 'co2':
            df = pd.concat([df, df_up])
            df = df.groupby(df.index).last() # drop duplicated times (keep the last instance)
        else:
            df.loc[df.index.isin(df_up.index), par] = df_up.loc[df_up.index.isin(df.index), par]
        df = df[df.index <= time1]
    
    # Calculate monthly means of merged data (require at least n_min measurements per month)
    if res == 'hourly':
        df_mon = monthly_means(df[[par]], n_min)
        if par != 'co2':
            df_mon[par+'_cams'] = monthly_means(df[[par+'_cams']], n_min)
    else:
        df_mon = df[[par]].copy()
        if par != 'co2':
            df_mon[par+'_cams'] = df[par+'_cams']
        
    # Define training and 'validation' sets
    df_train = df.copy()
    df_train.loc[(df.index >= time0) & (df.index < time1+timedelta(days=1)), par] = np.nan
    df_val = df[(df.index >= time0) & (df.index < time1+timedelta(days=1))]
    if df_val[par].notna().sum() == 0:
        return None, None, None, None

    # Downscale/debias CAMS
    empty_df = df_val.drop(index=df_val.index)
    if par == 'co2':        
        y_pred, y_pred_mon, anom_score = [empty_df, empty_df, empty_df]
    elif res == 'monthly':
        y_pred, anom_score = [empty_df, empty_df]
        y_pred_mon = debias(df_train, df_val[par+'_cams'], par).round(2)
    else:
        y_pred, y_pred_mon, anom_score = downscaling(df_train, df_val, par, window_size_cams, ml_model)
        y_pred_mon = y_pred_mon[y_pred_mon.index.isin(df_mon.index)]
        if y_pred_mon.shape[0] > 1:
            if y_pred.index[0] > y_pred_mon.index[1]-timedelta(days=1): 
                y_pred_mon = y_pred_mon.iloc[1:] # exclude first month if it has less than one day of data
        y_pred = y_pred.round(2)
        y_pred_mon = y_pred_mon.round(2)
            
    # Apply Sub-LOF on training and validation periods
    if res == 'hourly':
        score_train = run_sublof(df_train[par], subLOF, window_size)
        score_val = run_sublof(df_val[par], subLOF, window_size)
    else:
        score_train = score_val = pd.Series([])

    # Write log
    write_log(logpath, 
              [datetime.isoformat(datetime.now(), sep=' ', timespec='seconds'),
               cod, par, hei, time0, time1, int(is_new)])
    
    out = [par, cod, res, is_new,
           df.to_json(date_format='iso', orient='columns'),
           df_mon.to_json(date_format='iso', orient='columns'),
           df_train.to_json(date_format='iso', orient='columns'),
           df_val.to_json(date_format='iso', orient='columns'),
           y_pred.to_json(date_format='iso', orient='index'),
           y_pred_mon.to_json(date_format='iso', orient='index'),
           anom_score.to_json(date_format='iso', orient='index'),
           score_train.to_json(date_format='iso', orient='index'),
           score_val.to_json(date_format='iso', orient='index')]   
    
    return out, None, None, None



@callback(Output('export-csv', 'data'),
          Input('btn_csv', 'n_clicks'),
          prevent_initial_call=True)
def export_csv(n_clicks):
    if outfile == None:
        raise PreventUpdate
        
    return dcc.send_data_frame(df_exp.to_csv, outfile, index=False)


@callback(Output('export-csv-monthly', 'data'),
          Input('btn_csv_monthly', 'n_clicks'),
          prevent_initial_call=True)
def export_csv_monthly(n_clicks):
    if outfile_monthly == None:
        raise PreventUpdate
        
    return dcc.send_data_frame(df_exp_mon.to_csv, outfile_monthly, index=False)


@callback(Output('export-csv-dc', 'data'),
          Input('btn_csv_dc', 'n_clicks'),
          prevent_initial_call=True)
def export_csv_dc(n_clicks):
    if outfile_dc == None:
        raise PreventUpdate
        
    return dcc.send_data_frame(df_exp_dc.to_csv, outfile_dc, index=False)


@callback(Output('export-csv-sc', 'data'),
          Input('btn_csv_sc', 'n_clicks'),
          prevent_initial_call=True)
def export_csv_sc(n_clicks):
    if outfile_sc == None:
        raise PreventUpdate
        
    return dcc.send_data_frame(df_exp_sc.to_csv, outfile_sc, index=False)


@callback(Output('export-csv-vc', 'data'),
          Input('btn_csv_vc', 'n_clicks'),
          prevent_initial_call=True)
def export_csv_vc(n_clicks):
    if outfile_vc == None:
        raise PreventUpdate
        
    return dcc.send_data_frame(df_exp_vc.to_csv, outfile_vc, index=False)



@callback(Output('graph-hourly', 'figure'),
          Output('div-switch', 'style'),
          Output('div-hourly', 'style'),
          Output('div-hourly-settings', 'style'),
          Output('div-hourly-info', 'style'),
          Input('points-switch', 'on'),
          Input('cams-switch-1', 'on'),
          Input('threshold-slider', 'value'),
          Input('bin-slider', 'value'),
          Input('input-data', 'data'))
def update_figure_1(points_on, cams_on, selected_q, bin_size, input_data):
    if input_data == []:
        raise PreventUpdate
        
    global outfile
    global df_exp
        
    if input_data is None:
        outfile = None
        return empty_plot('No data available in the selected period - Try choosing a longer period'), \
            {'display':'none'}, {'display':'block'}, {'display':'none'}, {'display':'none'}
        
    param, cod, res, is_new, df, df_mon, df_train, df_val, y_pred, \
        y_pred_mon, anom_score, score_train, score_val = input_data
        
    if res == 'monthly':
        return empty_plot('No hourly data available'), \
            {'display':'flex'}, {'display':'block'}, {'display':'none'}, {'display':'none'}
        
    df_train = pd.read_json(df_train, orient='columns')
    df_val = pd.read_json(df_val, orient='columns')
    y_pred = pd.read_json(y_pred, orient='index', typ='series')
    anom_score = pd.read_json(anom_score, orient='index', typ='series')
    score_train = pd.read_json(score_train, orient='index')
    score_val = pd.read_json(score_val, orient='index')
    
    # Define filename for export file
    outfile = cod + '_' + param + '_' + \
        str(df_val.index[0])[:10].replace('-','') + '-' + \
        str(df_val.index[-1])[:10].replace('-','') + '_' + \
        str(selected_q) + '_hourly.csv'
    
    # Flag data after Sub-LOF
    threshold = thr0_lof+incr_lof*selected_q
    thr_yellow = np.quantile(score_train.dropna(), threshold)
    thr_red = thr_yellow * 2
    flags_yellow = df_val[(score_val[0]>thr_yellow) & (score_val[0]<=thr_red)]
    n_flags_yellow = flags_yellow.shape[0]
    flags_red = df_val[score_val[0]>thr_red]
    n_obs = df_val.shape[0]
    
    # Flag extreme outliers that exceed historical records by half the historical range (red flags only)
    half_range = (df_train[param].max() - df_train[param].min()) / 2
    thr_max = df_train[param].max() + half_range
    thr_min = df_train[param].min() - half_range
    flags_red = pd.concat([flags_red, df_val[(df_val[param]>thr_max)|(df_val[param]<thr_min)]])
    flags_red = flags_red.groupby(flags_red.index).first() # drop duplicates
    n_flags_red = flags_red.shape[0]
    
    # Create data frame for pie chart
    df_pie = pd.DataFrame({'color':['green','yellow','red'],
                           'name':['normal','anomalous','very anomalous'],
                           'value':[n_obs-n_flags_red-n_flags_yellow, 
                                    n_flags_yellow, n_flags_red]})
    
    # Create data frame for export
    df_exp = pd.DataFrame({'Time':df_val.index, 
                           'Value':df_val[param],
                           'CAMS':np.nan,
                           'CAMS + ML':np.nan,
                           'Flag1':0, 
                           'Flag2':np.nan,
                           'Strictness':selected_q})
    df_exp.loc[df_exp['Time'].isin(flags_yellow.index), 'Flag1'] = 1
    df_exp.loc[df_exp['Time'].isin(flags_red.index), 'Flag1'] = 2
    if cams_on & (param != 'co2'):
        df_exp['CAMS'] = df_val[param+'_cams']
        df_exp['CAMS + ML'] = y_pred
    else:
        df_exp.drop(columns=['CAMS','CAMS + ML','Flag2'], 
                    inplace=True)
    
    # Flag data after CAMS (yellow only; two yellow flags [LOF+CAMS] trigger a red flag)
    if cams_on & (param != 'co2'):
        df_exp['Flag2'] = 0
        threshold_cum = thr0_cams + incr_cams*selected_q
        anom_score_train = anom_score[anom_score.index.isin(df_train.dropna().index)]
        thr_cum_yellow = np.quantile(anom_score_train.dropna(), 
                                     [1-threshold_cum, threshold_cum]) * 2
        anom_score_val = anom_score[anom_score.index.isin(df_val.index)]
        flags_cum_yellow = (anom_score_val < thr_cum_yellow[0]) | \
            (anom_score_val > thr_cum_yellow[1])
        flags_cum_yellow = pd.Series(df_val.index[flags_cum_yellow])
        for fl in flags_cum_yellow: # flags must be extended to the entire window used by cumulative_score
            time0 = max(fl-timedelta(hours=window_size_cams-1), df_val.index[0])
            additional_times = pd.Series(pd.date_range(time0, fl, freq='H'))
            flags_cum_yellow = pd.concat([flags_cum_yellow, additional_times], ignore_index=True)
        flags_cum_yellow.drop_duplicates(inplace=True)
        flags_cum_yellow.sort_values(inplace=True)
        flags_cum_red = flags_cum_yellow[flags_cum_yellow.isin(flags_yellow.index)]
        df_pie.loc[df_pie['color']=='red', 'value'] += len(flags_cum_red)
        df_pie.loc[df_pie['color']=='yellow', 'value'] += (len(flags_cum_yellow) - len(flags_cum_red))
        df_pie.loc[df_pie['color']=='green', 'value'] -= len(flags_cum_yellow)
        df_exp.loc[df_exp['Time'].isin(flags_cum_yellow), 'Flag2'] = 1
        i_yellow = (df_exp['Flag1'] == 1) & (df_exp['Flag2'] != 1)
        i_red = (df_exp['Flag1'] + df_exp['Flag2']) > 1
    else:
        i_yellow = df_exp['Flag1'] == 1
        i_red = df_exp['Flag1'] == 2
    
    # Define plot panels
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'rowspan': 2}, {'type': 'domain'}], [None, {}]],
                        column_widths=[0.75, 0.25], row_heights=[0.5, 0.5],
                        horizontal_spacing=0.075, vertical_spacing=0.075,
                        subplot_titles=(' ','',''))
    
    # Plot measurements
    if points_on:
        fig.add_trace(go.Scatter(x=df_val.index, y=df_val[param], mode='markers', marker_color='black',
                                 marker_size=3, hoverinfo='skip', name='Measurements'),
                      row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df_val.index, y=df_val[param], mode='lines', line_color='black',
                                 line_width=2, hoverinfo='skip', name='Measurements'),
                      row=1, col=1)

    # Plot CAMS
    if cams_on & (param != 'co2'):
        if points_on:
            fig.add_trace(go.Scatter(x=df_val.index, y=df_val[param+'_cams'], mode='markers',
                                     hoverinfo='skip', marker_color='gray', marker_size=2,
                                     name='CAMS'),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='markers', 
                                     hoverinfo='skip', marker_color='orange', marker_size=2.5,
                                     name='CAMS + ML'),
                          row=1, col=1)            
        else:
            fig.add_trace(go.Scatter(x=df_val.index, y=df_val[param+'_cams'], mode='lines', 
                                     hoverinfo='skip', line_color='gray', line_width=0.75,
                                     name='CAMS'),
                          row=1, col=1) 
            fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines', 
                                     hoverinfo='skip', line_color='orange', line_width=1.5,
                                     name='CAMS + ML'),
                          row=1, col=1)           
            
        # Plot shaded areas for significant differences
        if len(flags_cum_yellow) > 0:
            flags_cum = (flags_cum_yellow - df_val.index[0]).astype('int64')/3.6e12 # convert to number of hours from time 0
            flags_cum = flags_cum.reset_index(drop=True)
            flags_cum_diff = np.diff(flags_cum, prepend=-window_size_cams)
            i_blocks = np.array(np.where(flags_cum_diff >= window_size_cams), ndmin=1)
            if i_blocks.size == 1:
                flags_cum = np.array(flags_cum[i_blocks.item()], ndmin=1)
            else:
                i_blocks = i_blocks.squeeze()
                flags_cum = flags_cum[i_blocks]
            last_pos = 0
            for i, fl in enumerate(flags_cum):
                t_flag = df_val.index[0] + timedelta(hours=fl)
                if i == i_blocks.size-1:
                    n = np.sum(flags_cum_diff[last_pos+1:])
                else:
                    n = np.sum(flags_cum_diff[i_blocks[i]+1:i_blocks[i+1]])
                    last_pos = i_blocks[i+1]
                time1 = t_flag + timedelta(hours=n)
                fig.add_vrect(x0=t_flag, x1=time1, fillcolor='gold', opacity=0.3, line_width=0,
                              row=1, col=1)
        
    # Plot flags
    fig.add_trace(go.Scatter(x=df_exp.loc[i_yellow,'Time'], y=df_exp.loc[i_yellow,'Value'], 
                             mode='markers', marker_symbol='circle-open', marker_line_width=3,
                             marker_size=8, marker_color='yellow', name='Yellow flags (LOF)'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_exp.loc[i_red,'Time'], y=df_exp.loc[i_red,'Value'], 
                             mode='markers', marker_symbol='circle-open', marker_line_width=3,
                             marker_size=8, marker_color='red', name='Red flags (LOF)'),
                  row=1, col=1)
    
    # Plot pie chart
    fig.add_trace(go.Pie(labels=df_pie['name'], values=df_pie['value'], 
                         marker={'colors':df_pie['color']}, showlegend=False),
                  row=1, col=2)
    
    # Plot histogram
    fig.add_trace(go.Histogram(x=df_val[param], histnorm='probability', 
                               name='Measurements', marker_color='black',
                               xbins=dict(size=bin_size), showlegend=False),
                  row=2, col=2)
    if cams_on & (param != 'co2'):
        fig.add_trace(go.Histogram(x=df_val[param+'_cams'], histnorm='probability', 
                                   name='CAMS', marker_color='gray',
                                   xbins=dict(size=bin_size), showlegend=False),
                      row=2, col=2)        
        fig.add_trace(go.Histogram(x=y_pred, histnorm='probability', 
                                   name='CAMS + ML', marker_color='orange',
                                   xbins=dict(size=bin_size), showlegend=False),
                      row=2, col=2)
    
    # Add logo Empa
    fig = add_logo(fig, 0.0, 0.0, 0.075, 0.2)
    
    # Formatting (borders, labels, figure size, etc.)
    meta = metadata[metadata['gaw_id']==cod]
    fig.update_xaxes(title_text='Time UTC', showline=True, linewidth=1, 
                     linecolor='black', mirror=True,
                     row=1, col=1)
    fig.update_xaxes(title_text=param.upper() + ' mole fraction ' + 
                         ('(ppm)' if param=='co2' else '(ppb)'),
                     showline=True, linewidth=1, linecolor='black', mirror=True,
                     showgrid=True, row=2, col=2)
    fig.update_xaxes(tickfont_size=14)
    fig.update_yaxes(title_text=param.upper() + ' mole fraction ' + 
                         ('(ppm)' if param=='co2' else '(ppb)'), 
                    showline=True, linewidth=1, linecolor='black', mirror=True, 
                    row=1, col=1)
    fig.update_yaxes(title_text='Density', showline=True, linewidth=1,
                     linecolor='black', mirror=True, row=2, col=2)
    fig.update_yaxes(tickfont_size=14)
    fig.update_traces(name='', row=1, col=2)
    fig.update_traces(opacity=0.8, row=2, col=2)
    fig.update_layout(title={'text':meta['name'].item() + ' (' + cod + '): hourly means', 
                             'xanchor':'center', 'x':0.5},
                      legend={'orientation':'h', 'xanchor':'left', 'x':0.0,
                              'yanchor':'bottom', 'y':1.01, 
                              'itemsizing': 'constant'},
                      margin={'t':75}, barmode='overlay', width=1600, height=600)
    return fig, {'display':'flex'}, {'display':'block'}, \
        {'display':'flex', 'align-items':'center', 'justify-content':'center'}, \
        {'display':'flex', 'padding-left':'25px'}


@app.callback(Output('collapse-1', 'is_open'),
              [Input('info-button-1', 'n_clicks')],
              [State('collapse-1', 'is_open')])
def toggle_collapse_1(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(Output('graph-monthly', 'figure'),
          Output('div-monthly', 'style'),
          Input('points-switch', 'on'),
          Input('cams-switch-2', 'on'),
          Input('trend-radio', 'value'),
          Input('trend-radio', 'options'),
          Input('length-slider', 'value'),
          Input('input-data', 'data'))
def update_figure_2(points_on, cams_on, selected_trend, label_trend, max_length, input_data):
    if input_data == []:
        raise PreventUpdate
        
    global outfile_monthly
    global df_exp_mon
        
    if input_data is None:
        outfile_monthly = None
        return empty_plot(''), {'display':'none'}

    param, cod, res, is_new, df, df_mon, df_train, df_val, y_pred, \
        y_pred_mon, anom_score, score_train, score_val = input_data
        
    df_mon = pd.read_json(df_mon, orient='columns')
    df_val = pd.read_json(df_val, orient='columns')
    y_pred_mon = pd.read_json(y_pred_mon, orient='index', typ='series')
    label_trend = pd.DataFrame(label_trend)
    label_trend = label_trend['label'][label_trend['value']==selected_trend].item()
    
    # Define filename for export file
    outfile_monthly = cod + '_' + param + '_' + \
        str(df_val.index[0])[:10].replace('-','') + '-' + \
        str(df_val.index[-1])[:10].replace('-','') + '_' + \
        str(max_length) + '_' + label_trend.replace(' ','').lower() + '_monthly.csv'
    
    # Extract periods
    mtp = len(df_val.index.month.unique()) # number of months in the target period
    df_mon = df_mon[df_mon.index < df_val.index[-1]] # exclude data after target period
    length = np.min([df_mon.shape[0], max_length*12 + mtp])
    df_sar = df_mon[param].iloc[-length:]
    df_sar_to_predict = df_sar.iloc[-mtp:]
    
    # Fit SARIMAX
    sar = sm.tsa.statespace.SARIMAX(df_sar[:-mtp], 
                                    order=(1,0,0), 
                                    seasonal_order=(0,1,1,12), 
                                    trend=selected_trend,
                                    enforce_stationarity=False).fit(disp=False)
    # Make forecast
    fcst, confidence_int = forecast_sm(sar, mtp, p_conf, df_sar.index)
    fcst = fcst.round(2)
    confidence_int = confidence_int.round(2)
    
    # Flags
    flags = df_sar_to_predict[(df_sar_to_predict > confidence_int['upper '+param]) | 
                   (df_sar_to_predict < confidence_int['lower '+param])]
    
    # Create data frame for export
    n_meas = df_val[param].groupby(df_val.index.month).count()
    df_exp_mon = pd.DataFrame({'Time':df_sar_to_predict.index, 
                               'Value':df_sar_to_predict.values,
                               'N measurements':n_meas.values,
                               'SARIMA (best estimate)':fcst,
                               'SARIMA (lower limit)':confidence_int['lower '+param],
                               'SARIMA (upper limit)':confidence_int['upper '+param],
                               'CAMS':np.nan,
                               'CAMS + ML':np.nan,
                               'Flag':0, 
                               'Years used':int((length-mtp)/12),
                               'Trend':label_trend})
    df_exp_mon.loc[df_exp_mon['Time'].isin(flags.index), 'Flag'] = 1
    if cams_on & (param != 'co2'):
        df_exp_mon['CAMS'] = df_mon[param+'_cams'].iloc[-mtp:]
        df_exp_mon['CAMS + ML'] = y_pred_mon
    else:
        df_exp_mon.drop(columns=['CAMS','CAMS + ML'], inplace=True)
    
    # Plot
    fig = go.Figure(layout=go.Layout(width=1200, height=600))
    # SARIMA
    if (mtp > 1) & (not points_on): # plot a line + filled area
        fig.add_trace(go.Scatter(x=confidence_int.index, 
                                 y=confidence_int['lower '+param], 
                                 mode='lines', line=dict(color='orange',width=0.1), 
                                 showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=confidence_int.index, 
                                 y=confidence_int['upper '+param],
                                 mode='lines', line=dict(color='orange',width=0.1), 
                                 fill='tonextx', hoverinfo='skip',
                                 name=str(int(100*(1-p_conf)))+'% confidence range'))
        fig.add_trace(go.Scatter(x=fcst.index, y=fcst, 
                                 mode='lines', line=dict(color='orange',width=1.5), 
                                 name='SARIMA'))
    else: # plot a point + error bars
        confidence_int['upper '+param] -= fcst
        fig.add_trace(go.Scatter(x=fcst.index, y=fcst, mode='markers',
                                 marker_color='orange', marker_size=10,
                                 error_y=dict(type='data',visible=True,
                                              array=confidence_int['upper '+param]),
                                 name='SARIMA'))
    # CAMS    
    if cams_on & (param != 'co2'):
        fig.add_trace(go.Scatter(x=df_exp_mon.index, y=df_exp_mon['CAMS + ML'], mode='markers', 
                                 marker_color='royalblue', marker_size=11, marker_symbol='star', 
                                 name='CAMS + ML'))
        fig.add_trace(go.Scatter(x=df_exp_mon.index, y=df_exp_mon['CAMS'], mode='markers', 
                                 marker_color='white', marker_size=8, marker_symbol='diamond', 
                                 marker_line_color='gray', marker_line_width=2, name='CAMS'))        

    # Measurements
    if points_on:
        fig.add_trace(go.Scatter(x=df_sar.index, y=df_sar, 
                                 mode='markers', marker_color='black', marker_size=8, 
                                 name='Measurements'))
    else:
        fig.add_trace(go.Scatter(x=df_sar.index, y=df_sar, 
                                 mode='lines+markers', marker_color='black', marker_size=8, 
                                 line_color='black', name='Measurements'))
    # Flags
    if flags.shape[0] > 0:
        fig.add_trace(go.Scatter(x=flags.index, y=flags, mode='markers',
                                 marker_color='red', marker_size=20, marker_symbol='circle-open', 
                                 marker_line_width=3, name='Flag', showlegend=False))
        
    # Add logo Empa
    fig = add_logo(fig, 0.0, 0.0, 0.1, 0.2)
   
    # Formatting (borders, labels, etc.)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, tickfont_size=14)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, tickfont_size=14)
    meta = metadata[metadata['gaw_id']==cod] 
    fig.update_layout(title={'text':meta['name'].item() + ' (' + cod + '): monthly means', 
                             'xanchor':'center', 'x':0.5},
                      legend={'orientation':'h', 'xanchor':'left', 'x':0.0,
                              'yanchor':'bottom', 'y':1.01, 
                              'traceorder':'reversed'}, 
                      margin={'t':75}, xaxis_title='Time UTC',
                      yaxis_title=param.upper() + ' mole fraction ' + 
                                      ('(ppm)' if param=='co2' else '(ppb)'))
    
    return fig, {'display':'block'}


@app.callback(Output('collapse-2', 'is_open'),
              [Input('info-button-2', 'n_clicks')],
              [State('collapse-2', 'is_open')])
def toggle_collapse_2(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(Output('graph-cycles', 'figure'),
          Output('div-cycles', 'style'),
          Input('points-switch', 'on'),
          Input('n-years', 'value'),
          Input('input-data', 'data'))
def update_figure_3(points_on, n_years, input_data):
    if input_data == []:
        raise PreventUpdate
        
    global outfile_dc
    global df_exp_dc
    global outfile_sc
    global df_exp_sc
    global outfile_vc
    global df_exp_vc
        
    if input_data is None:
        return empty_plot(''), {'display':'none'}
    
    param, cod, res, is_new, df, df_mon, df_train, df_val, y_pred, \
        y_pred_mon, anom_score, score_train, score_val = input_data
        
    df = pd.read_json(df, orient='columns')
    df_mon = pd.read_json(df_mon, orient='columns')
    df_train = pd.read_json(df_train, orient='columns')
    df_val = pd.read_json(df_val, orient='columns')
    
    # Deal with periods spanning two calendar years by shifting the data before 01.01 by one year
    lastj = df_val.index.dayofyear[-1]
    if is_new & (df_val.index.dayofyear[0] > lastj):
        new_index = pd.Series(df.index)
        new_index[df.index.dayofyear>lastj] = \
            new_index[df.index.dayofyear>lastj] + timedelta(days=365)
        df.set_index(new_index, inplace=True)
        
    # Select data to be plotted
    lastyear = df_train[param].dropna().index.year[-1]
    firstyear = lastyear - n_years + 1
    firstmonth = df_val.index.month[0]
    df_train_sel_diurnal = df[(df.index.year>=firstyear) & 
                              (df.index.dayofyear.isin(df_val.index.dayofyear.unique()))]
    df_mon_new = df_mon[(df_mon.index>=df_val.index[0]) & (df_mon.index<df_val.index[-1])]
    df_mon_sel = df_mon[df_mon.index.year>=firstyear].copy()
    if is_new:
        df_mon_sel[df_mon_sel.index.isin(df_mon_new.index)] = np.nan
        df_train_sel = df_train.loc[df_train.index.year>=firstyear, param]
    else:
        df_train_sel = df.loc[df.index.year>=firstyear, param]
    
    # Calculate diurnal cycle and number of available days per hour
    if res != 'monthly':
        dc = df_train_sel_diurnal[[param]].groupby(df_train_sel_diurnal.index.hour).mean().round(2)
        dc_val = df_val[[param]].groupby(df_val.index.hour).mean().round(2)
        dc_n = df_val[[param]].groupby(df_val.index.hour).count()
    
    # Calculate mean seasonal cycle
    sc = df_mon_sel[[param]].groupby(df_mon_sel.index.month).mean().round(2)
    sc_new = df_mon_new[[param]].groupby(df_mon_new.index.month).mean().round(2)
    
    # Calculate variability for submitted data
    vc_new = df_val.groupby(df_val.index.month).std().round(2)
    n_new = df_val.groupby(df_val.index.month).count()
    vc_new[n_new<n_min] = np.nan
    
    # Define filenames for export
    outfile_dc = cod + '_' + param + '_' + \
        str(firstyear) + '-' + str(lastyear) + '_diurnal-cycle.csv'
    outfile_sc = cod + '_' + param + '_' + \
        str(firstyear) + '-' + str(lastyear) + '_seasonal-cycle.csv'
    outfile_vc = cod + '_' + param + '_' + \
        str(firstyear) + '-' + str(lastyear) + '_variability-cycle.csv'
    
    # Create data frames for export
    label = 'Your data' if is_new else 'Selected period'
    period_label = 'Mean  ' + str(firstyear) + '-' + str(lastyear)
    
    df_exp_sc = pd.DataFrame({'Month':np.arange(1,13), 
                             period_label:np.nan,
                             label:np.nan})
    df_exp_sc.set_index(df_exp_sc['Month'], inplace=True)
    df_exp_sc.loc[sc_new.index, label] = sc_new[param].values
    df_exp_sc.loc[sc.index, period_label] = sc[param]
    
    if res != 'monthly':
        
        df_exp_dc = pd.DataFrame({'Hour':dc.index, 
                                 period_label:dc[param],
                                 label:dc_val[param],
                                 'N days':dc_n[param]})
    
        df_exp_vc = pd.DataFrame({'Month':np.arange(1,13), 
                                 period_label:np.nan,
                                 label:np.nan})
        df_exp_vc.set_index(df_exp_vc['Month'], inplace=True)
        df_exp_vc.loc[vc_new.index, label] = vc_new[param].values
       
    if n_years == 1: # remove multi-year average if there is only one year
        df_exp_sc.drop(columns=[period_label], inplace=True)
        if res != 'monthly':
            df_exp_dc.drop(columns=[period_label], inplace=True)
            df_exp_vc.drop(columns=[period_label], inplace=True)
    
    # Define plot panels and line colors
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.075,
                        subplot_titles=('Diurnal cycle', 'Seasonal cycle', 'Variability cycle'))
    colors = ['#f0f921', '#fdb42f','#ed7953', '#cc4778', '#9c179e', '#5c01a6', '#0d0887'] # plasma palette
    years = np.arange(firstyear, firstyear+n_years)
       
    # Plot diurnal cycle
    if res != 'monthly':
        x_labels = pd.date_range('2020-01-01', periods=24, freq='h')
        for iy in years:
            if ((iy != lastyear) | (firstmonth == 1)) & (firstyear != lastyear):
                df_train_year = df_train_sel_diurnal[df_train_sel_diurnal.index.year==iy]
                dc_train_year = df_train_year[[param]].groupby(df_train_year.index.hour).mean().round(2)
                df_exp_dc[str(iy)] = dc_train_year[param]
                if points_on:
                    fig.add_trace(go.Scatter(x=x_labels, y=dc_train_year[param], mode='markers',
                                             marker_size=4, marker_color=colors[iy-firstyear-n_years], 
                                             customdata=df_train_year[[param]].groupby(df_train_year.index.hour).count(),
                                             hovertemplate='%{y} (N=%{customdata})',
                                             name=str(iy), showlegend=False),
                                  row=1, col=1)                   
                else:
                    fig.add_trace(go.Scatter(x=x_labels, y=dc_train_year[param], mode='lines',
                                             line_width=1, line_color=colors[iy-firstyear-n_years], 
                                             customdata=df_train_year[[param]].groupby(df_train_year.index.hour).count(),
                                             hovertemplate='%{y} (N=%{customdata})',
                                             name=str(iy), showlegend=False),
                                  row=1, col=1)
        if n_years > 1:
            if points_on:
                fig.add_trace(go.Scatter(x=x_labels, y=dc[param], mode='markers',
                                         marker_size=8, marker_color='royalblue',
                                         name=period_label, showlegend=False),
                              row=1, col=1)                
            else:
                fig.add_trace(go.Scatter(x=x_labels, y=dc[param], mode='lines',
                                         line_width=5, line_color='royalblue',
                                         name=period_label, showlegend=False),
                              row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=dc_val[param], mode='markers', 
                                 marker_color='black', marker_size=10, marker_symbol='circle',
                                 customdata=dc_n[param],
                                 hovertemplate='%{y} (N=%{customdata})',
                                 name=label, showlegend=False),
                      row=1, col=1)
    else:
        fig = empty_subplot(fig, 'No hourly data available', 1, 1)
    
    # Plot seasonal cycle
    for iy in years:
        df_mon_year = df_mon_sel[df_mon_sel.index.year==iy]
        df_exp_sc[str(iy)] = np.nan
        df_exp_sc.loc[df_mon_year.index.month, str(iy)] = df_mon_year[param].values
        if points_on:
            fig.add_trace(go.Scatter(x=df_mon_year.index.month, y=df_mon_year[param], mode='markers',
                                     marker_size=4, marker_color=colors[iy-firstyear-n_years], 
                                     name=str(iy)),
                          row=1, col=2)
        else:
            fig.add_trace(go.Scatter(x=df_mon_year.index.month, y=df_mon_year[param], mode='lines',
                                     line_width=1, line_color=colors[iy-firstyear-n_years], 
                                     name=str(iy)),
                          row=1, col=2)            
    if n_years > 1:
        if points_on:
            fig.add_trace(go.Scatter(x=sc.index, y=sc[param], mode='markers',
                                     marker_size=8, marker_color='royalblue',
                                     name=period_label),
                          row=1, col=2)       
        else:
            fig.add_trace(go.Scatter(x=sc.index, y=sc[param], mode='lines',
                                     line_width=5, line_color='royalblue',
                                     name=period_label),
                          row=1, col=2)
    fig.add_trace(go.Scatter(x=sc_new.index, y=sc_new[param], mode='markers', 
                             marker_color='black', marker_size=10, 
                             marker_symbol='circle', name=label),
                  row=1, col=2)
    
    # Plot seasonal cycle of variability
    if res != 'monthly':
        for iy in years:
            df_train_year = df_train_sel[df_train_sel.index.year==iy]
            vc_year = df_train_year.groupby(df_train_year.index.month).std().round(2)
            n_year = df_train_year.groupby(df_train_year.index.month).count()
            vc_year[n_year<n_min] = np.nan
            df_exp_vc[str(iy)] = np.nan
            df_exp_vc.loc[vc_year.index, str(iy)] = vc_year.values
            if points_on:
                fig.add_trace(go.Scatter(x=vc_year.index, y=vc_year, mode='markers',
                                         marker_size=4, marker_color=colors[iy-firstyear-n_years], 
                                         name=str(iy), showlegend=False),
                              row=1, col=3)
            else:
                fig.add_trace(go.Scatter(x=vc_year.index, y=vc_year, mode='lines',
                                         line_width=1, line_color=colors[iy-firstyear-n_years], 
                                         name=str(iy), showlegend=False),
                              row=1, col=3)                
        if n_years > 1:
            df_exp_vc[period_label] = df_exp_vc[list(map(str,years))].mean(axis=1).round(2).values # average of all years in df_train_sel
            if points_on:
                fig.add_trace(go.Scatter(x=df_exp_vc.index, y=df_exp_vc[period_label], mode='markers',
                                         marker_size=8, marker_color='royalblue',
                                         name=period_label, showlegend=False),
                              row=1, col=3)
            else:
                fig.add_trace(go.Scatter(x=df_exp_vc.index, y=df_exp_vc[period_label], mode='lines',
                                         line_width=5, line_color='royalblue',
                                         name=period_label, showlegend=False),
                              row=1, col=3)
        fig.add_trace(go.Scatter(x=df_exp_vc.index, y=df_exp_vc[label], mode='markers', 
                                 marker_color='black', marker_size=10, 
                                 marker_symbol='circle', name=label, showlegend=False),
                      row=1, col=3)
    else:
        fig = empty_subplot(fig, 'No hourly data available', 1, 3)
        
    # Add logo Empa
    fig = add_logo(fig, 0.36, 0.0, 0.075, 0.2)
    if res != 'monthly':
        fig = add_logo(fig, 0.0, 0.0, 0.075, 0.2)
        fig = add_logo(fig, 0.72, 0.0, 0.075, 0.2)
    
    # Formatting (borders, labels, etc.)
    meta = metadata[metadata['gaw_id']==cod]
    units = '(ppm)' if param=='co2' else '(ppb)'
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, tickfont_size=14)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, tickfont_size=14)
    fig.update_xaxes(title_text='Month', ticktext=month_names, tickvals=df_exp_sc.index, 
                     range=[0,13], showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text=param.upper() + ' mole fraction ' + units,
                     row=1, col=2)
    if res != 'monthly':
        fig.update_xaxes(title_text='Time UTC', tickformat='%H:%M', row=1, col=1)
        fig.update_yaxes(title_text=param.upper() + ' mole fraction ' + units,
                         row=1, col=1)
        fig.update_xaxes(title_text='Month', ticktext=month_names, tickvals=df_exp_vc.index, 
                         range=[0,13], showgrid=False, row=1, col=3)
        fig.update_yaxes(title_text='Standard deviation of hourly ' + param.upper() + 
                             ' mole fraction ' + units,
                         row=1, col=3)
    fig.update_layout(title={'text':meta['name'].item() + ' (' + cod + ')', 
                             'xanchor':'center', 'x':0.5,
                             'yanchor':'top', 'y':0.98},
                      legend={'itemclick':False, 'itemdoubleclick':False},
                      margin={'t':75}, hovermode='x unified', 
                      width=1600, height=600)
    
    return fig, {'display':'block'}


@app.callback(Output('collapse-3', 'is_open'),
              [Input('info-button-3', 'n_clicks')],
              [State('collapse-3', 'is_open')])
def toggle_collapse_3(n, is_open):
    if n:
        return not is_open
    return is_open
    
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)