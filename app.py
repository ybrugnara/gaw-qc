# -*- coding: utf-8 -*-
"""
Dashboard for GAW QC tool

Historical and near-real time CAMS data are read from a sqlite database
Data to analyze are uploaded by the user in the form of a xls or csv file

@author: bryu
"""

from dash import Dash, dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_daq as daq
import plotly.graph_objects as go
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from pandas.core.tools.datetimes import guess_datetime_format
from datetime import datetime, timedelta
import statsmodels.api as sm
from pyod.models.lof import LOF
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import base64
from io import StringIO, BytesIO



# Path to database
inpath = 'test.db'

# Sub-LOF parameters
window_size = 6
n_neighbors = 100

# SARIMAX parameters
p_conf = 0.01 # confidence level probability for the shaded area in the plot

# CAMS-based anomaly score parameter (in hours)
window_size_cams = 50

# Threshold parameters (thr0 + incr * strictness level)
thr0_lof = 0.994
incr_lof = 0.002
thr0_cams = 0.985
incr_cams = 0.005

# Define Sub-LOF instance
subLOF = LOF(n_neighbors=n_neighbors, metric='euclidean')



# Define functions

def read_data(db_file, gaw_id, v, h, last_time):
    """ read data for the target station from the database
    :param db_file: Path of database file
    :param gaw_id: GAW id of the target station
    :param v: Variable (one of ch4, co, co2, o3)
    :param h: Height from the ground
    :param last_time: Last timestamp of the period to analyze
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
        out_cams.rename(columns={'value':v+'_cams', 'value_tc':'tc_'+v}, inplace=True)
        out_cams.drop(columns=['id','series_id'], inplace=True)
        out = pd.merge(out_gaw, out_cams, how='outer', on='time')
        
    out['time'] = pd.to_datetime(out['time'], format='ISO8601')
    out['hour'] = out['time'].dt.hour
    out['doy'] = out['time'].dt.dayofyear
    out['trend'] = np.arange(out.shape[0])
    out.set_index('time', inplace=True)
    out = out[out.index <= last_time]
        
    return out, res


def filter_data(df, res, thr_h, thr_m, w):
    """ assign NaN to hourly/monthly means that are based on an insufficient number of measurements
    :param df: Data frame produced by read_data (time as index)
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
        df.loc[df['n_meas']<thr_m, par] = np.nan
        
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
        print('Could not recognize time format')
        return []
    df_up['Time'] = df_up['Time'].dt.round('H') # round to nearest hour
    if tz != 'UTC':
        df_up['Time'] = to_utc(df_up['Time'], tz) # convert time to UTC
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
    :param tz: Local timezone as 'UTC+xx:xx' or 'UTC-xx:xx'
    :param reverse: Set to True to covert from UTC to local
    :return: Series of converted times
    """
    k = -1 if reverse else 1
    if tz[3] == '-':    
        series = series + k*timedelta(hours=int(tz[4:6]), minutes=int(tz[7:9]))
    elif tz[3] == '+':
        series = series - k*timedelta(hours=int(tz[4:6]), minutes=int(tz[7:9]))
    return series


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


def downscaling(df_train, df_val, par, w):
    """ dowscaling algorithm for CAMS forecasts; the anomaly score is calculated as the moving median of the prediction error
    :param df_train: Data frame of training data (containing both measurements and CAMS data, with time as index)
    :param df_val: Same as df_train but for the target period
    :param par: Variable to debias (defines the column names)
    :param w: Size of the moving window used to calculate the anomaly score (integer)
    :return: Series of downscaled data for the target period (hourly and monthly), series of the anomaly score
    """
    RF = RandomForestRegressor(criterion='squared_error', n_estimators=200, 
                               n_jobs=-1, max_depth=15, max_samples=None, 
                               max_features=6, min_samples_leaf=40, random_state=42)
    df_train_cams = df_train.dropna()
    df_val_cams = df_val.drop(par,axis=1).dropna()
    RF.fit(df_train_cams.drop(par,axis=1).to_numpy(), df_train_cams[par].to_numpy())
    y_pred = RF.predict(df_val_cams.to_numpy())
    y_pred = pd.Series(y_pred, index=df_val_cams.index).reindex(index=df_val.index)
    
    # Monthly data (for SARIMA plot)  
    y_pred_mon = y_pred.groupby(pd.Grouper(freq='1M',label='left')).mean()
    y_pred_mon.index = y_pred_mon.index + timedelta(days=1)

    # Anomaly score
    y_pred_all = RF.predict(pd.concat([df_train_cams.drop(par,axis=1), df_val_cams]).to_numpy())
    y_pred_all = pd.Series(y_pred_all, index=np.concatenate((df_train_cams.index,df_val_cams.index)))
    errors = y_pred_all - pd.concat([df_train_cams[par], df_val.loc[df_val.index.isin(df_val_cams.index), par]])
    errors = errors.reindex(index=np.concatenate((df_train.index,df_val.index)))
    diff_series = errors - np.median(errors[:df_train.shape[0]].dropna())
    anom_score = diff_series.rolling(w, min_periods=int(w/2)).median()
    
    return y_pred, y_pred_mon, anom_score


def empty_plot(msg): 
    """ return an empty plot containing a text message
    :param msg: Message to print in the plot area
    :return: Figure object
    """
    fig = go.Figure()
    fig.update_layout(
        xaxis =  {'visible':False},
        yaxis = {'visible':False},
        annotations = [{'text':msg,
                        'xref':'paper',
                        'yref':'paper',
                        'showarrow':False,
                        'font':{'size':28}}])
    
    return fig
            


### DASH APP ###
app = Dash(__name__)

metadata = read_meta(inpath)
gaw_stations = metadata[['name','gaw_id']].set_index('gaw_id') \
                 .sort_values(by='name').to_dict()['name']

app.layout = html.Div([
    
    # Input form
    html.Div([
        html.Div([
            dcc.Dropdown(gaw_stations, 
                         id='station-dropdown', value='JFJ', #placeholder='Select station',
                         style={'font-size':'16px', 'font-family':'Arial'}),
            ], style={'width':'20%', 'padding-left':'50px', 'margin-bottom':'10px'}),
        html.Div([
            dcc.Dropdown(id='param-dropdown', placeholder='Select parameter',
                         style={'font-size':'16px', 'font-family':'Arial'}),
            ], style={'width':'10%', 'padding-left':'50px', 'margin-bottom':'10px'}),
        html.Div([
            dcc.Dropdown(id='height-dropdown', placeholder='Select height (m)',
                         style={'font-size':'16px', 'font-family':'Arial'}),
            ], style={'width':'10%', 'padding-left':'50px', 'margin-bottom':'10px'}),
        html.Div([
            dcc.Dropdown(['UTC',
                          'UTC-11:00','UTC-10:00','UTC-09:00','UTC-08:00','UTC-07:00',
                          'UTC-06:00','UTC-05:00','UTC-04:00','UTC-03:00','UTC-02:00',
                          'UTC-01:00','UTC+01:00','UTC+02:00','UTC+03:00',
                          'UTC+04:00','UTC+05:00','UTC+06:00','UTC+07:00','UTC+08:00',
                          'UTC+09:00','UTC+10:00','UTC+11:00','UTC+12:00'],
                         id='timezone-dropdown', value='UTC', #placeholder='Select time zone',
                         style={'font-size':'16px', 'font-family':'Arial'}),
                ], style={'width':'10%', 'padding-left':'50px', 'margin-bottom':'10px'}),
        html.Div([
            dcc.Upload(children=html.Button('Upload File'),
                #children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                id='upload-data'),
            html.Label([], id='upload-label',
                       style={'font-size':'16px', 'font-family':'Arial', 'color':'red', 'padding-left':'10px'})
            ], style={'display':'flex', 'padding-left':'50px', 'margin-top':'25px', 'margin-bottom':'50px'}),
        html.Hr()]),
        
    # Hourly plot
    html.Div([
        html.Div([
            dcc.Loading(
                id='loading-1a',
                type='circle',
                children=dcc.Graph(id='graph-sublof')
                ),
            dcc.Loading(
                id='loading-1b',
                type='circle',
                children=dcc.Graph(id='pie-chart')
                )
            ], style={'display':'flex', 'padding-left':'50px'}),
        html.Div([
            html.Div([
                html.Label('Toggle debiased CAMS forecasts', 
                           style={'font-size':'14px', 'font-family':'Arial', 'padding-right':'20px'}),
                daq.BooleanSwitch(id='cams-switch-1', on=True)
                ], style={'display':'flex', 'padding-left':'50px'}),
            html.Div([
                html.Label('Strictness', style={'font-size':'14px', 'font-family':'Arial'}),
                daq.Slider(
                    min=1,
                    max=3,
                    step=1,
                    value=2,
                    marks={1: {'label':'Most strict', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           3: {'label':'Least strict', 'style':{'font-size':'16px', 'font-family':'Arial'}}},
                    id='threshold-slider',
                    updatemode='drag',
                    handleLabel={'showCurrentValue':True, 'label':'level', 'style':{'font-family':'Arial'}},
                    size=300
                    )
                ], style={'padding-left':'200px'}),
            html.Div([
                html.Button('Export to CSV', id='btn_csv'),
                dcc.Download(id='export-csv'),
                ], style={'padding-left':'200px'}),
            ], style={'display':'flex', 'padding-left':'50px', 'margin-bottom':'75px'})]),
    
    # Monthly plot
    html.Div([    
        html.Div([
                html.Div([
                dcc.Loading(
                    id='loading-2',
                    type='circle',
                    children=dcc.Graph(id='graph-sarima')
                    )
                ]),
                html.Div([
                        html.Label('Type of trend', 
                                   style={'font-size':'14px', 'font-family':'Arial', 
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
                            labelStyle={'font-size':'14px', 'font-family':'Arial'}
                            )
                ], style={'margin-top':'100px'})
        ], style={'display':'flex', 'padding-left':'50px', 'margin-top':'100px'}),
        html.Div([
            html.Div([
                html.Label('Toggle debiased CAMS forecasts', 
                           style={'font-size':'14px', 'font-family':'Arial', 'padding-right':'20px'}),
                daq.BooleanSwitch(id='cams-switch-2', on=True)
                ], style={'display':'flex', 'padding-left':'50px'}),
            html.Div([
                html.Label('Number of years to use', style={'font-size':'14px', 'font-family':'Arial'}),
                dcc.Slider(
                    min=4,
                    max=10,
                    step=1,
                    value=7,
                    marks={4: {'label': '4', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           5: {'label': '5', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           6: {'label': '6', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           7: {'label': '7', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           8: {'label': '8', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           9: {'label': '9', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           10: {'label': '10', 'style':{'font-size':'16px', 'font-family':'Arial'}}},
                    id='length-slider',
                    updatemode='drag'
                    )
                ], style={'padding-left':'150px', 'width':'350px'}),
            html.Div([
                html.Button('Export to CSV', id='btn_csv_monthly'),
                dcc.Download(id='export-csv-monthly'),
                ], style={'padding-left':'200px'}),
            ], style={'display':'flex', 'padding-left':'50px', 'margin-bottom':'75px'})]),
    
    # Diurnal cycle plot
    html.Div([    
        html.Div([
                dcc.Loading(
                    id='loading-3',
                    type='circle',
                    children=dcc.Graph(id='graph-diurnal-cycle')
                    )
                ], style={'padding-left':'250px', 'margin-top':'50px'}),
        html.Div([
                html.Label('Number of years to use', style={'font-size':'14px', 'font-family':'Arial'}),
                dcc.Slider(
                    min=1,
                    max=7,
                    step=1,
                    value=4,
                    marks={1: {'label':'1', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           2: {'label':'2', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           3: {'label':'3', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           4: {'label':'4', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           5: {'label':'5', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           6: {'label':'6', 'style':{'font-size':'16px', 'font-family':'Arial'}},
                           7: {'label':'7', 'style':{'font-size':'16px', 'font-family':'Arial'}}},
                    id='n-years',
                    updatemode='drag'
                    )
                ], style={'padding-left':'475px', 'width':'350px', 'margin-bottom':'50px'})]),
    
    # Data storage
    html.Div([
            dcc.Loading(
                id='loading-data',
                type='circle',
                fullscreen=True,
                children=dcc.Store(id='input-data', data=[], storage_type='memory')
                )
            ])
    
])



@callback(Output('param-dropdown', 'options'),
          Output('param-dropdown', 'value'),
          Input('station-dropdown', 'value'))
def update_variables(stat):
    if stat is None:
        raise PreventUpdate
        
    meta = read_series(inpath, stat)
    pars = np.sort(np.unique(meta['variable']))
    
    return list(pars), None


@callback(Output('height-dropdown', 'options'),
          Output('height-dropdown', 'value'),
          Output('upload-label', 'children'),
          Output('cams-switch-1', 'on'),
          Output('cams-switch-2', 'on'),
          Output('upload-data', 'contents'),
          Input('station-dropdown', 'value'),
          Input('param-dropdown', 'value'),
          Input('height-dropdown', 'value'))
def update_heights(stat, par, hei):
    if (stat is None) | (par is None):
        raise PreventUpdate
        
    meta = read_series(inpath, stat)
    heights = np.sort(np.unique(meta['height'][meta['variable']==par]))
    statname = metadata.loc[metadata['gaw_id']==stat, 'name']
    
    if hei in heights:
        val = hei
    elif len(heights) == 1:
        val = heights.item()
    else:
        val = None
        
    if val != None:
        lbl = '<<< Please upload the data for ' + par.upper() + ' at ' + \
            statname + ' (' + str(val) + ' m)'
    else:
        lbl = []
        
    if par == 'co2':
        switch = False
    else:
        switch = True
    
    return list(heights), val, lbl, switch, switch, None


@callback(Output('input-data', 'data'),
          Output('upload-label', 'children', allow_duplicate=True),
          Input('param-dropdown', 'value'),
          Input('station-dropdown', 'value'),
          Input('height-dropdown', 'value'),
          Input('timezone-dropdown', 'value'),
          Input('upload-data', 'contents'),
          State('upload-data', 'filename'),
          prevent_initial_call=True)
def update_data(par, cod, hei, tz, content, filename):
    if (cod is None) | (par is None) | (hei is None) | (content is None):
        raise PreventUpdate
        
    # Parse uploaded data
    df_up = parse_data(content, filename, par, tz)
    time0 = df_up.index[0]
    time1 = df_up.index[-1]
    
    # Read data from database and remove values based on insufficient measurements
    df, res = read_data(inpath, cod, par, hei, time1)
    df = filter_data(df, res, 0.25, 15, 5)
    
    # Calculate monthly means of uploaded data
    if res == 'monthly':
        df_up = df_up.groupby(pd.Grouper(freq='1M',label='left')).mean()
        df_up.index = df_up.index + timedelta(days=1)
    
    # Merge uploaded data with historical data
    if par == 'co2':
        df = pd.concat([df, df_up])
        df = df.groupby(df.index).last() # drop duplicated times (keep the last instance)
    else:
        df.loc[df.index.isin(df_up.index), par] = df_up.loc[df_up.index.isin(df.index), par]
    df = df[df.index <= time1]
    
    # Calculate monthly means of merged data
    if res == 'hourly':
        df_mon = df[[par]].groupby(pd.Grouper(freq='1M',label='left')).mean()      
        if df.index[0] > df_mon.index[1]: 
            df_mon = df_mon.iloc[1:,:] # exclude first month if it has less than one day of data
        if df.index[-1]-timedelta(days=2) < df_mon.index[-1]: 
            df_mon = df_mon.iloc[:-1,:] # exclude last month if it has less than one day of data
        df_mon.index = df_mon.index + timedelta(days=1)
    else:
        df_mon = df[[par]].copy()
        
    # Split into training and 'validation' sets
    df_train = df[df.index < time0]
    df_val = df[df.index >= time0]

    # Downscale/debias CAMS
    empty_df = df_val.drop(index=df_val.index)
    if par == 'co2':        
        y_pred, y_pred_mon, anom_score = [empty_df, empty_df, empty_df]
    elif res == 'monthly':
        y_pred, anom_score = [empty_df, empty_df]
        y_pred_mon = debias(df_train, df_val[par+'_cams'], par)
    else:
        y_pred, y_pred_mon, anom_score = downscaling(df_train, df_val, par, window_size_cams)
        y_pred_mon = y_pred_mon[y_pred_mon.index.isin(df_mon.index)]
        if y_pred.index[0] > y_pred_mon.index[1]-timedelta(days=1): 
            y_pred_mon = y_pred_mon.iloc[1:] # exclude first month if it has less than one day of data

    # Apply Sub-LOF on training and 'validation' periods
    if res == 'hourly':
        score_train = run_sublof(df_train[par], subLOF, window_size)
        score_val = run_sublof(df_val[par], subLOF, window_size)
    else:
        score_train = score_val = pd.Series([])
    
    out = [par, cod, res,
           df.to_json(date_format='iso', orient='columns'),
           df_mon.to_json(date_format='iso', orient='columns'),
           df_train.to_json(date_format='iso', orient='columns'),
           df_val.to_json(date_format='iso', orient='columns'),
           y_pred.to_json(date_format='iso', orient='index'),
           y_pred_mon.to_json(date_format='iso', orient='index'),
           anom_score.to_json(date_format='iso', orient='index'),
           score_train.to_json(date_format='iso', orient='index'),
           score_val.to_json(date_format='iso', orient='index')]  
    
    return out, []


@callback(
    Output("export-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True)
def export_csv(n_clicks):
    if outfile == None:
        raise PreventUpdate
        
    return dcc.send_data_frame(df_exp.to_csv, outfile, index=False)


@callback(
    Output("export-csv-monthly", "data"),
    Input("btn_csv_monthly", "n_clicks"),
    prevent_initial_call=True)
def export_csv_monthly(n_clicks):
    if outfile_monthly == None:
        raise PreventUpdate
        
    return dcc.send_data_frame(df_exp_mon.to_csv, outfile_monthly, index=False)


@callback(Output('graph-sublof', 'figure'),
          Output('pie-chart', 'figure'),
          Input('cams-switch-1', 'on'),
          Input('threshold-slider', 'value'),
          Input('input-data', 'data'))
def update_figure_1(on, selected_q, input_data):
    if input_data == []:
        raise PreventUpdate
    
    param, cod, res, df, df_mon, df_train, df_val, y_pred, \
        y_pred_mon, anom_score, score_train, score_val = input_data
        
    if res == 'monthly': # if data are monthly write a message in place of the plots
        return empty_plot('No hourly data available'), empty_plot('No hourly data available')

    df_train = pd.read_json(StringIO(df_train), orient='columns')
    df_val = pd.read_json(StringIO(df_val), orient='columns')
    y_pred = pd.read_json(StringIO(y_pred), orient='index', typ='series')
    anom_score = pd.read_json(StringIO(anom_score), orient='index', typ='series')
    score_train = pd.read_json(StringIO(score_train), orient='index')
    score_val = pd.read_json(StringIO(score_val), orient='index')
    
    # Define filename for export file
    global outfile
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
    global df_exp
    df_exp = pd.DataFrame({'Time':df_val.index, 
                           'Value':df_val[param],
                           'Prediction CAMS':np.nan,
                           'Flag1':0, 
                           'Flag2':np.nan,
                           'Strictness':selected_q})
    df_exp.loc[df_exp['Time'].isin(flags_yellow.index), 'Flag1'] = 1
    df_exp.loc[df_exp['Time'].isin(flags_red.index), 'Flag1'] = 2
    if on & (param!='co2'):
        df_exp['Prediction CAMS'] = y_pred
    
    # Flag data after CAMS
    if on & (param != 'co2'):
        df_exp['Flag2'] = 0
        threshold_cum = thr0_cams + incr_cams*selected_q
        thr_cum_yellow = np.quantile(anom_score[:df_train.shape[0]].dropna(), 
                                     [1-threshold_cum, threshold_cum]) * 2
        anom_score_val = pd.Series(anom_score[-df_val.shape[0]:], index=df_val.index)
        flags_cum_yellow = (anom_score_val < thr_cum_yellow[0]) | \
            (anom_score_val > thr_cum_yellow[1])
        flags_cum_yellow = pd.Series(df_val.index[flags_cum_yellow])
        for fl in flags_cum_yellow: # flags must be extended to the entire window used by cumulative_score
            time0 = max(fl-timedelta(hours=window_size_cams-1), df_val.index[0])
            additional_times = pd.Series(pd.date_range(time0, fl, freq='H'))
            flags_cum_yellow = pd.concat([flags_cum_yellow, additional_times], ignore_index=True)
        flags_cum_yellow.drop_duplicates(inplace=True)
        flags_cum_yellow.sort_values(inplace=True)
        flags_cum_red = flags_cum_yellow[flags_cum_yellow.isin(flags_yellow.index)] # two yellow flags => red flag
        df_pie.loc[df_pie['color']=='red', 'value'] += len(flags_cum_red)
        df_pie.loc[df_pie['color']=='yellow', 'value'] += (len(flags_cum_yellow) - len(flags_cum_red))
        df_pie.loc[df_pie['color']=='green', 'value'] -= len(flags_cum_yellow)
        df_exp.loc[df_exp['Time'].isin(flags_cum_yellow), 'Flag2'] = 1
    
    # Plot time series
    fig = go.Figure(layout=go.Layout(width=1200, height=600))
    fig.add_trace(go.Scatter(x=df_val.index, y=df_val[param], mode='lines',
                             hoverinfo='skip', name='Measurements', showlegend=True))
    i_yellow = (df_exp['Flag1'] == 1) & (df_exp['Flag2'] != 1)
    i_red = (df_exp['Flag1'] + df_exp['Flag2'].fillna(0)) > 1
    fig.add_trace(go.Scatter(x=df_exp.loc[i_yellow,'Time'], y=df_exp.loc[i_yellow,'Value'], mode='markers', 
                             marker_size=8, marker_color='yellow', name='Yellow flags', showlegend=True))
    fig.add_trace(go.Scatter(x=df_exp.loc[i_red,'Time'], y=df_exp.loc[i_red,'Value'], mode='markers', 
                             marker_size=8, marker_color='red', name='Red flags', showlegend=True))
    # Plot CAMS
    if on & (param != 'co2'):
        fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines', 
                                 hoverinfo='skip', line_color='orange', 
                                 name='Prediction (CAMS)', showlegend=True))
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
                fig.add_vrect(x0=t_flag, x1=time1, fillcolor='gold', opacity=0.3, line_width=0)
    elif on & (param == 'co2'):
        fig.add_annotation(x=0.5, xref='paper', y=1.07, yref='paper', 
                           text='(CAMS data not available for CO2)', 
                           font=dict(family='Arial',size=16,color='darkred'), 
                           showarrow=False)
    
    # Add border
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # Labels and formatting            
    meta = metadata[metadata['gaw_id']==cod]            
    fig.update_layout(title={'text':meta['name'].item() + ' (' + cod + '): hourly means', 
                             'xanchor':'center', 'x':0.5},
                      legend={'yanchor':'top', 'y':0.99, 'xanchor':'left', 'x':0.01},
                      xaxis_title='Time UTC',
                      yaxis_title=param.upper() + ' mole fraction ' + 
                                      ('(ppm)' if param=='co2' else '(ppb)'),
                      xaxis_range=[df_val.index[0],df_val.index[-1]], 
                      yaxis_range=[df_val[param].min()-5,df_val[param].max()+5])
    
    # Plot pie chart
    pie = go.Figure(layout=go.Layout(width=400, height=400),
                    data=[go.Pie(labels=df_pie['name'], values=df_pie['value'], 
                                 marker={'colors':df_pie['color']})])
    
    return fig, pie


@callback(Output('graph-sarima', 'figure'),
          Input('cams-switch-2', 'on'),
          Input('trend-radio', 'value'),
          Input('trend-radio', 'options'),
          Input('length-slider', 'value'),
          Input('input-data', 'data'))
def update_figure_2(on, selected_trend, label_trend, max_length, input_data):
    if input_data == []:
        raise PreventUpdate
        
    param, cod, res, df, df_mon, df_train, df_val, y_pred, \
        y_pred_mon, anom_score, score_train, score_val = input_data
    df_mon = pd.read_json(StringIO(df_mon), orient='columns')
    df_val = pd.read_json(StringIO(df_val), orient='columns')
    y_pred_mon = pd.read_json(StringIO(y_pred_mon), orient='index', typ='series')
    label_trend = pd.DataFrame(label_trend)
    label_trend = label_trend['label'][label_trend['value']==selected_trend].item()
    
    # Define filename for export file
    global outfile_monthly
    outfile_monthly = cod + '_' + param + '_' + \
        str(df_val.index[0])[:10].replace('-','') + '-' + \
        str(df_val.index[-1])[:10].replace('-','') + '_' + \
        str(max_length) + '_' + label_trend.replace(' ','').lower() + '_monthly.csv'
    
    # Extract the number of months included in the submitted period (months to predict)
    mtp = len(df_val.index.month.unique())
    
    # Extract period
    length = np.min([df_mon.shape[0], max_length*12 + mtp])
    df_sar = df_mon.iloc[-length:]
    df_sar.index = pd.DatetimeIndex(df_sar.index.values, freq='MS')
    df_sar_to_predict = df_sar[-mtp:]
    
    # Fit SARIMA model
    sar = sm.tsa.statespace.SARIMAX(df_sar[:-mtp], 
                                    order=(1,0,0), 
                                    seasonal_order=(0,1,1,12), 
                                    trend=selected_trend,
                                    enforce_stationarity=False).fit(disp=False)
    # Make forecast
    fcst, confidence_int = forecast_sm(sar, mtp, p_conf, df_sar.index)
    
    # Flags
    flags = df_sar_to_predict[(df_sar_to_predict[param] > confidence_int['upper '+param]) | 
                   (df_sar_to_predict[param] < confidence_int['lower '+param])]
    
    # Create data frame for export
    global df_exp_mon
    df_exp_mon = pd.DataFrame({'Time':df_sar_to_predict.index, 
                               'Value':df_sar_to_predict[param],
                               'N measurements':df_val[param].groupby(df_val.index.month).count().values,
                               'Prediction':fcst,
                               'Prediction CAMS':np.nan,
                               'Flag':0, 
                               'Years used':int((length-mtp)/12),
                               'Trend':label_trend})
    df_exp_mon.loc[df_exp_mon['Time'].isin(flags.index), 'Flag'] = 1
    if on & (param != 'co2'):
        df_exp_mon['Prediction CAMS'] = y_pred_mon
    
    # Plot
    fig = go.Figure(layout=go.Layout(width=1200, height=600))
    fig.add_trace(go.Scatter(x=df_sar.index, y=df_sar[param], 
                             mode='lines+markers', marker_color='black', marker_size=6, 
                             line_color='black', name='Monthly means', showlegend=True))
    if mtp > 1: # plot a line + filled area
        fig.add_trace(go.Scatter(x=fcst.index, y=fcst, 
                                 mode='lines', line=dict(color='orange',width=1.5), 
                                 name='Prediction', showlegend=True))
        fig.add_trace(go.Scatter(x=confidence_int.index, 
                                 y=confidence_int['lower '+param], 
                                 mode='lines', line=dict(color='orange',width=0.1), 
                                 showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=confidence_int.index, 
                                 y=confidence_int['upper '+param],
                                 mode='lines', line=dict(color='orange',width=0.1), 
                                 fill='tonextx', hoverinfo='skip',
                                 name=str(int(100*(1-p_conf)))+'% confidence range', 
                                 showlegend=True))
    else: # plot a point + error bars
        confidence_int['upper '+param] -= fcst
        fig.add_trace(go.Scatter(x=fcst.index, y=fcst, 
                                 marker=dict(color='orange',size=6),
                                 error_y=dict(type='data',visible=True,
                                              array=confidence_int['upper '+param]),
                                 name='Prediction', showlegend=True))
    if flags.shape[0] > 0:
        fig.add_trace(go.Scatter(x=flags.index, y=flags[param], mode='markers',
                                 marker_color='red', marker_size=20, marker_symbol='circle-open', 
                                 marker_line_width=3, name='Flags', showlegend=True))
    if on & (param != 'co2'):
        fig.add_trace(go.Scatter(x=y_pred_mon.index, y=y_pred_mon, mode='markers', 
                                 marker_color='royalblue', marker_size=8, marker_symbol='star', 
                                 name='Prediction (CAMS)', showlegend=True))
    elif on & (param == 'co2'):
        fig.add_annotation(x=0.5, xref='paper', y=1.07, yref='paper', 
                           text='(CAMS data not available for CO2)', 
                           font=dict(family='Arial',size=16,color='darkred'), 
                           showarrow=False)
    # Add border
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # Labels and formatting 
    meta = metadata[metadata['gaw_id']==cod] 
    fig.update_layout(title={'text':meta['name'].item() + ' (' + cod + '): monthly means', 
                             'xanchor':'center', 'x':0.5},
                      legend={'yanchor':'top', 'y':0.99, 
                              'xanchor':'left', 'x':0.01, 'traceorder':'normal'}, 
                      xaxis_title='Time UTC',
                      yaxis_title=param.upper() + ' mole fraction ' + 
                                      ('(ppm)' if param=='co2' else '(ppb)'))
    
    return fig


@callback(Output('graph-diurnal-cycle', 'figure'),
          Input('n-years', 'value'),
          Input('timezone-dropdown', 'value'),
          Input('input-data', 'data'))
def update_figure_3(n_years, tz, input_data):
    if input_data == []:
        raise PreventUpdate
    
    param, cod, res, df, df_mon, df_train, df_val, y_pred, \
        y_pred_mon, anom_score, score_train, score_val = input_data
        
    if res == 'monthly': # if data are monthly write a message in place of the plot
        return empty_plot('No hourly data available')
        
    df = pd.read_json(StringIO(df), orient='columns')
    df_val = pd.read_json(StringIO(df_val), orient='columns')
    
    # Convert to local time
    if tz != 'UTC':
        new_index = to_utc(pd.Series(df.index), tz, reverse=True)
        df.set_index(new_index, inplace=True)
        new_index = to_utc(pd.Series(df_val.index), tz, reverse=True)
        df_val.set_index(new_index, inplace=True)
    
    # Deal with periods spanning two calendar years by shifting the data before 01.01 by one year
    lastj = df_val.index.dayofyear[-1]
    if df_val.index.dayofyear[0] > lastj:
        new_index = pd.Series(df.index)
        new_index[df.index.dayofyear>lastj] = \
            new_index[df.index.dayofyear>lastj] + timedelta(days=365)
        df.set_index(new_index, inplace=True)
    
    # Select data to be plotted
    lastyear = df_val.index.year[-1]
    firstyear = lastyear - n_years
    df_train_recent = df[(df.index.year>=firstyear) & 
                         (df.index.dayofyear.isin(df_val.index.dayofyear.unique()))]
    
    # Calculate diurnal cycle and number of available days per hour
    dc_val = df_val[[param]].groupby(df_val.index.hour).mean()
    dc_n = df_val[[param]].groupby(df_val.index.hour).count()
    n_days = np.unique(df_val.index.date).size
    
    # Plot
    fig = go.Figure(layout=go.Layout(width=800, height=600))
    x_labels = pd.date_range('2020-01-01', periods=24, freq='h')
    colors = ['#f0f921', '#fdb42f','#ed7953', '#cc4778', '#9c179e', '#5c01a6', '#0d0887'] # plasma palette
    for iy in range(firstyear,firstyear+n_years):
        df_train_year = df_train_recent[df_train_recent.index.year==iy]
        dc_train_year = df_train_year[[param]].groupby(df_train_year.index.hour).mean()
        fig.add_trace(go.Scatter(x=x_labels, y=dc_train_year[param], mode='lines',
                      line_width=1.5, line_color=colors[iy-firstyear-n_years], 
                      name=iy, showlegend=True))
    fig.add_trace(go.Scatter(x=x_labels, y=dc_val[param], mode='markers', 
                             marker_color='black', marker_size=10, marker_symbol='circle', 
                             name=str(lastyear), showlegend=True))
    # Add number of measurements
    fig.add_annotation(x=x_labels[0]-timedelta(hours=1), xref='x', y=1.05, yref='paper',
                       text='N:', font=dict(family='Arial',size=12), showarrow=False)
    for i in range(24):
        c = 'magenta' if dc_n.iloc[i].item() < n_days/2 else 'darkgreen'
        fig.add_annotation(x=x_labels[i], xref='x', y=1.05, yref='paper', 
                           text=dc_n.iloc[i].item(), font=dict(family='Arial',size=12,color=c), 
                           showarrow=False)
    # Add border
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # Labels and formatting 
    meta = metadata[metadata['gaw_id']==cod]
    fig.update_layout(title={'text':meta['name'].item() + ' (' + cod + '): diurnal cycle', 
                             'xanchor':'center', 'x':0.5},
                      hovermode='x unified',
                      xaxis_tickformat='%H:%M',
                      xaxis_title='Time ' + tz,
                      yaxis_title=param.upper() + ' mole fraction ' + 
                                      ('(ppm)' if param=='co2' else '(ppb)'))
    
    return fig
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)