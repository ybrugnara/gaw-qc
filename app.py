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
from flask_caching import Cache
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from pandas.core.tools.datetimes import guess_datetime_format
from datetime import date, datetime, timedelta
import calendar
import pytz
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

# Maximum number of years that can be processed for the third panel
n_years_max = 7 

# Minimum date to use CAMS data
min_date_cams = datetime(2020,1,1)

# Sub-LOF parameters
window_size = 5
n_neighbors = 100

# SARIMAX parameter (confidence level probability for the shaded area in the plot)
p_conf = 0.01

# Minimum number of months required to fit SARIMAX
min_months_sarima = 24

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

def read_data(db_file, gaw_id, v, h, time1):
    """ read data for the target station from the database
    :param db_file: Path of database file
    :param gaw_id: GAW id of the target station
    :param v: Variable (one of ch4, co, co2, o3)
    :param h: Height from the ground
    :param time1: latest time to analyze
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
            if (v != 'co2') & (time1 >= min_date_cams):
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
            if (v != 'co2') & (time1 >= min_date_cams):
                cur.execute('SELECT * FROM cams_monthly WHERE ' + \
                                'series_id=?', (series_id,))
                data_cams = cur.fetchall()
                names_cams = [x[0] for x in cur.description]
            res = 'monthly'
            
    out_gaw = pd.DataFrame(data_gaw, columns=names_gaw)
    out_gaw.rename(columns={'value':v}, inplace=True)
    
    if (v == 'co2') | (time1 < min_date_cams):
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
        df.loc[(df['n_meas']>0) & (df['n_meas']<thr_h*running_max), par] = np.nan
    else:
        df.loc[(df['n_meas']>0) & (df['n_meas']<thr_m), par] = np.nan
        
    return df


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
    :return: List of two dates
    """
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute('SELECT id FROM series WHERE ' + \
                        'gaw_id=? AND variable=? AND height=?', 
                    (gaw_id, v, h))
        series_id = cur.fetchone()
        
        if series_id is None:
            return None, None
        
        series_id = series_id[0]
        cur.execute('SELECT MIN(time),MAX(time) FROM gaw_hourly WHERE ' + \
                        'series_id=?', (series_id,))
        dates = cur.fetchall()[0]
        
        if dates[0] is None:
            cur.execute('SELECT MIN(time),MAX(time) FROM gaw_monthly WHERE ' + \
                            'series_id=?', (series_id,))
            dates = cur.fetchall()[0]     
            last_day = calendar.monthrange(int(dates[1][:4]),int(dates[1][5:7]))[1]
            return dates[0]+'-01', dates[1]+'-'+str(last_day)
        
        else:   
            return dates[0][:10], dates[1][:10]
        
        
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
    df_up.sort_values(['Time'], inplace=True) # sort chronologically
    df_up.set_index('Time', inplace=True)
    if tz != 'UTC': # convert time to UTC
        df_up.index = df_up.index.shift(-int(tz[3:6]), freq='H') 
    
    # Deal with decimal separator and missing values
    try:
        df_up[par] = pd.to_numeric(df_up[par].replace(',', '.', regex=True))
    except:
        print('Could not convert data column to numeric')
        return []
    df_up.loc[df_up[par]<0, par] = np.nan # assign NaN to all negative values
    df_up = add_missing(df_up) # fill missing periods with NaNs
    df_up['n_meas'] = np.nan
    
    return df_up


def write_log(log_file, row):
    """ write log about user activity
    :param db_file: Path of log file
    :param row: Row to add to log (list)
    :return: None
    """
    log = pd.DataFrame(dict(time=row[0], gaw_id=row[1], variable=row[2], height=row[3],
                            start=row[4], end=row[5], upload=row[6], loading_time=row[7]),
                       index=[0])
    h = True if len(glob.glob(log_file)) == 0 else False      
    log.to_csv(log_file, header=h, index=False, mode='a', date_format='%Y-%m-%d')


def aggregate_scores(scores, w_size): 
    """ assign score to each point as the average of the scores of the windows it is in
    :param scores: Series produced by run_sublof
    :param w_size: Window size (integer)
    :return: Series of aggregated scores
    """
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
    scores = pd.DataFrame(model.decision_scores_, index=times)
    scores = scores.reindex(series.index)
    scores = aggregate_scores(scores, win)
    
    return scores


def add_missing(timeseries):
    """ fill missing times with NaN for hourly data
    :param timeseries: Data frame with hourly resolution (time as index)
    :return: Filled in data frame
    """
    return timeseries.resample('H').asfreq()


def monthly_means(timeseries, n_min, t0, t1):
    """ Calculate monthly means from hourly data (require at least n_min values per month)
    Data at the beginning or end of a month that is partly outside the analzed period are excluded
    :param timeseries: Data frame with one variable and time as index
    :param n_min: Integer
    :param t0: Start time of data to analyze
    :param t1: End time of data to analyze 
    :return: Data frame of monthly means
    """
    tmonth0 = datetime(t0.year, t0.month, 1, 0)
    tmonth1 = datetime(t1.year, t1.month, calendar.monthrange(t1.year,t1.month)[1], 23)
    to_drop = ((timeseries.index>=tmonth0) & (timeseries.index<t0)) | \
        ((timeseries.index<=tmonth1) & (timeseries.index>t1))
    tmp = timeseries.drop(index=timeseries.index[to_drop])
    out = tmp.groupby(pd.Grouper(freq='1M',label='left')).mean().round(2)
    n = tmp.groupby(pd.Grouper(freq='1M',label='left')).count()
    out[n<n_min] = np.nan
    out.index = out.index + timedelta(days=1) # first day of the month as index
    
    return out


def forecast_sm(df_m, n, max_l, sel_t): 
    """ make prediction using a SARIMA model
    :param df_m: Data frame with monthly data (one variable and time as index)
    :param n: Number of time steps to predict
    :param max_l: Maximum length in years of the training period
    :param sel_t: Type of trend (character string)
    :return: Four data frames with monthly data
    """
    # Extract periods and parameter name
    length = np.min([df_m.shape[0], max_l*12 + n])
    par = df_m.columns[0]
    df_sar = df_m[par].iloc[-length:]
    df_sar_to_predict = df_sar.iloc[-n:]
    
    if df_sar.iloc[:-n].count() >= min_months_sarima:    
        # Define and fit SARIMA model
        sar = sm.tsa.statespace.SARIMAX(df_sar[:-n], 
                                        order=(1,0,0), 
                                        seasonal_order=(0,1,1,12), 
                                        trend=sel_t,
                                        enforce_stationarity=False).fit(disp=False)        
        # Make forecast
        future_fcst = sar.get_forecast(n)
        confidence_int = future_fcst.conf_int(alpha=p_conf)
        confidence_int.index = df_sar.index[-n:]
        fcst = future_fcst.predicted_mean
        fcst.index = df_sar.index[-n:]
        fcst = fcst.round(2)
        confidence_int = confidence_int.round(2)
        
    else:
        fcst = pd.Series(n*[np.nan], index=df_sar.index[-n:])
        confidence_int = pd.DataFrame({'upper '+par:n*[np.nan], 'lower '+par:n*[np.nan]},
                                      index=df_sar.index[-n:])
        
    # Flags
    flags = df_sar_to_predict[(df_sar_to_predict > confidence_int['upper '+par]) | 
                   (df_sar_to_predict < confidence_int['lower '+par])]
    
    return df_sar, fcst, confidence_int, flags


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


def downscaling(df_train, df_test, par, w, model):
    """ dowscaling algorithm for CAMS forecasts; the anomaly score is calculated as the moving median of the prediction error
    :param df_train: Data frame of training data (containing both measurements and CAMS data, with time as index)
    :param df_test: Same as df_train but for the target period
    :param par: Variable to debias (defines the column names)
    :param w: Size of the moving window used to calculate the anomaly score (integer)
    :param model: Instance of a sklearn regression model
    :return: Series of downscaled data for the target period (hourly and monthly), series of the anomaly score
    """
    df_train_cams = df_train.dropna()
    df_test_cams = df_test.drop(par,axis=1).dropna()
    model.fit(df_train_cams.drop(par,axis=1).to_numpy(), df_train_cams[par].to_numpy())
    y_pred = model.predict(df_test_cams.to_numpy())
    y_pred = pd.Series(y_pred, index=df_test_cams.index).reindex(index=df_test.index)
    
    # Monthly data (for SARIMA plot)  
    y_pred_mon = y_pred.groupby(pd.Grouper(freq='1M',label='left')).mean()
    y_pred_n = y_pred.groupby(pd.Grouper(freq='1M',label='left')).count()
    y_pred_mon[y_pred_n<n_min] = np.nan
    y_pred_mon.index = y_pred_mon.index + timedelta(days=1)

    # Anomaly score
    x_to_predict = pd.concat([df_train_cams.drop(par,axis=1), df_test_cams]).sort_index()
    y_pred_all = model.predict(x_to_predict.to_numpy())
    y_pred_all = pd.Series(y_pred_all, index=x_to_predict.index)
    y_to_compare = pd.concat([df_train_cams[par], 
                              df_test.loc[df_test.index.isin(df_test_cams.index), par]]).sort_index()
    errors = y_pred_all - y_to_compare
    all_times = pd.concat([df_train, df_test]).sort_index().index
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


def add_header(df, par, is_new, cams_on, time1):
    """ Add header with data credits to export
    :param df: Data frame to export
    :param par: Variable (one of ch4, co, co2, o3)
    :param is_new: Whether the analzed data were uploaded by the user (boolean)
    :param cams_on: Whether CAMS data are exported (boolean)
    :param time1: latest time to analyze
    :return: String of comma-separated data ready for export
    """
    units = 'ppm' if par == 'co2' else 'ppb'
    header = '# Mole fraction of ' + par.upper() + ' in ' + units + '.'
    if not is_new:
        header += ' Measurements data source: '
        if par == 'o3':
            header += 'World Data Centre for Reactive Gases (WDCRG - www.gaw-wdcrg.org).'
        else:
            header += 'World Data Centre for Greenhouse Gases (WDCGG - gaw.kishou.go.jp).'
    if cams_on & (par != 'co2') & (time1 >= min_date_cams):
        header += ' CAMS data source: Copernicus Atmosphere Data Store (ADS - atmosphere.copernicus.eu/data).'
    header += ' File created by the gaw-qc app on ' + \
        datetime.now(pytz.timezone('UTC')).isoformat(sep=' ', timespec='seconds') + '.'
    
    out = header + '\r\n' + df.to_csv(index=False)
    
    return out
        


### DASH APP ###

app = Dash(external_stylesheets=[dbc.themes.MATERIA], 
           title='GAW-qc', update_title='Loading...')

cache = Cache(app.server, 
              config={ # note that filesystem cache doesn't work on systems with ephemeral filesystems like Heroku
                      'CACHE_TYPE': 'filesystem',
                      'CACHE_DIR': 'cache',   
                      'CACHE_THRESHOLD': 50 # should be equal to maximum number of users on the app at a single time
    })

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
                            ' in the framework of the Global Atmosphere Watch Programme of the World Meteorological Organization.',
                            ' Technical support was provided by the Scientific IT department at Empa.'
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
        # Store input parameters (also used to signal callbacks when to fire)
        dcc.Loading(
            id='loading-data',
            type='circle',
            fullscreen=False, 
            children=dcc.Store(id='input-data', data=[], storage_type='memory')
            ),
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
            html.P([
                'The first panel shows time series of hourly means.',
                ' The yellow and red circles indicate anomalous outliers;',
                ' the shaded areas indicate periods of systematic biases with respect to CAMS-derived predictions.',
                ' Click on the Help button for more information.'
                ])
            ], id='div-text-1', style={'display':'none'}),
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
                    dbc.Button('Export to CSV', id='btn_csv_hourly', size='sm'),
                    dcc.Download(id='export-csv-hourly')
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
            html.P([
                'The second panel shows time series of monthly means, including a few years of historical data.',
                ' The red circles indicate anomalous outliers.',
                ' Click on the Help button for more information.'
                ])
            ], id='div-text-2', style={'display':'none'}),
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
                                ' assumed by the model can be selected by the user (top-right corner).',
                                html.Br(),
                                ' The SARIMA prediction will not be shown if less than two years of past data are available.'
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
            html.P([
                'The third panel shows comparisons with other years for diurnal, seasonal, and variability cycle.',
                ' The analyzed data is represented by black dots.',
                ' Click on the Help button for more information.'
                ])
            ], id='div-text-3', style={'display':'none'}),
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
                    ], style={'padding-left':'345px'}),
                html.Div([
                    dbc.Button('Export to CSV', id='btn_csv_vc', size='sm'),
                    dcc.Download(id='export-csv-vc'),
                    ], style={'padding-left':'345px'})
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
        
        ], style={'margin-top':'50px', 'margin-bottom':'50px'}),
    
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
        
    d0, d1 = read_start_end(inpath, cod, par.lower(), hei)
    
    if d1 is None:
        return None, None, None, 'hide'
    
    return datetime.strptime(d0, '%Y-%m-%d'), datetime.strptime(d1, '%Y-%m-%d'), \
        date(int(d1[:4]), 1, 1), 'auto'
        
        
@callback(Output('timezone-dropdown', 'value', allow_duplicate=True),
          Input('timezone-dropdown', 'value'),
          Input('upload-data', 'contents'),
          prevent_initial_call=True)
def update_tz(tz, content):
    if (tz is not None) | (content is None):
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
def wait_message(cod, par, hei, date1, tz, content):
    if (cod is None) | (par is None) | (hei is None) | \
        ((date1 is None) & ((tz is None) | (content is None))):
        return '', {'display':'none'}, 'auto'
    else:
        return 'Please wait - Dashboard is loading...', \
            {'font-size':'18px', 'font-weight': 'bold', 'margin-bottom':'25px'}, \
            'hide'
            
            
@callback(Output('input-data', 'data', allow_duplicate=True), # reset dcc.Store to avoid firing all callbacks at once when an input parameter is changed
          Input('label-wait', 'children'),
          prevent_initial_call=True)
def reset_data(lab):
    if lab == '':
        raise PreventUpdate
    
    return []
           

# Read data from database and perform the most expensive computations
# Output is saved in the cache of the server: the function is not executed again unless the input changes            
@cache.memoize()
def get_data(cod, par, hei, date0, date1, tz, content, filename):
    par = par.lower()
    
    if content is None:
        time0 = datetime.strptime(date0, '%Y-%m-%d')
        time1 = datetime.strptime(date1, '%Y-%m-%d')
        is_new = False
    else:
        # Parse uploaded data
        df_up = parse_data(content, filename, par, tz)
        if not isinstance(df_up, pd.DataFrame): # file could not be read
            return []
        time0 = df_up.index[0]        
        time1 = df_up.index[-1]      
        if tz != 'UTC':
            ltime0 = time0 + timedelta(hours=int(tz[3:6]))
            ltime1 = time1 + timedelta(hours=int(tz[3:6]))
        else:
            ltime0 = time0
            ltime1 = time1
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
        if content is not None:
            df_up = df_up[df_up.index >= time0]
    
    # Read data and remove values based on insufficient measurements
    df_all, res = read_data(inpath, cod, par, hei, time1)
    df_all = filter_data(df_all, res, 0.25, 15, 5)
    
    # Calculate monthly means of uploaded data
    if (res == 'monthly') & (content is not None):
        df_up = monthly_means(df_up, n_min, ltime0, ltime1)
            
    # Merge uploaded data with historical data
    if content is not None:
        df_all.loc[(df_all.index>=time0) & (df_all.index<=time1), par] = np.nan
        if (par == 'co2') | (time1 < min_date_cams):
            df_all = pd.concat([df_all, df_up])
            df_all = df_all.groupby(df_all.index).last().sort_index() # drop duplicated times (keep the last instance)
        else:
            df_all.loc[df_all.index.isin(df_up.index), par] = df_up.loc[df_up.index.isin(df_all.index), par]
    
    # Calculate monthly means of merged data (require at least n_min measurements per month)
    if res == 'hourly':
        df_mon = monthly_means(df_all[[par]], n_min, 
                               time0 if content is None else ltime0, 
                               time1 if content is None else ltime1)
        if (par != 'co2') & (time1 >= min_date_cams):
            df_mon[par+'_cams'] = monthly_means(df_all[[par+'_cams']], n_min, 
                                                time0 if content is None else ltime0, 
                                                time1 if content is None else ltime1)
    else:
        df_mon = df_all[[par]].copy()
        if (par != 'co2') & (time1 >= min_date_cams):
            df_mon[par+'_cams'] = df_all[par+'_cams']
        
    # Define training and test sets
    if content is None:
        df_test = df_all[(df_all.index>=time0) & (df_all.index<time1+timedelta(days=1))].drop(columns='n_meas')
    else:
        df_test = df_all[(df_all.index>=time0) & (df_all.index<=time1)].drop(columns='n_meas')
    df_train = df_all.copy()
    df_train.loc[df_train.index.isin(df_test.index), par] = np.nan
    df_train = df_train.drop(columns='n_meas')
    if df_test[par].count() == 0:
        return None

    # Downscale/debias CAMS
    empty_df = df_test.drop(index=df_test.index)
    if (par == 'co2') | (time1 < min_date_cams):        
        y_pred, y_pred_mon, anom_score = [empty_df, empty_df, empty_df]
    elif res == 'monthly':
        y_pred, anom_score = [empty_df, empty_df]
        y_pred_mon = debias(df_train, df_test[par+'_cams'], par).round(2)
    else:
        y_pred, y_pred_mon, anom_score = downscaling(df_train, df_test, par, window_size_cams, ml_model)
        y_pred_mon = y_pred_mon[y_pred_mon.index.isin(df_mon.index)]
        y_pred = y_pred.round(2)
        y_pred_mon = y_pred_mon.round(2)
        anom_score_train = anom_score[anom_score.index.isin(df_train.dropna().index)]
            
    # Apply Sub-LOF on training and test periods
    if res == 'hourly':
        score_train = run_sublof(df_train[par], subLOF, window_size)
        score_test = run_sublof(df_test[par], subLOF, window_size)
    else:
        score_train = score_test = pd.Series([])
        
    # Calculate flagging thresholds for the three strictness levels
    thresholds = pd.DataFrame({'LOF':[], 'CAMS lower':[], 'CAMS upper':[], 
                               'range lower':[], 'range upper':[]})
    if res == 'hourly':      
        p_thrs_lof = thr0_lof + incr_lof * np.array([1,2,3])
        thresholds['LOF'] = np.quantile(score_train.dropna(), p_thrs_lof)
    else:
        thresholds['LOF'] = [np.nan, np.nan, np.nan]
    
    if (par != 'co2') & (res == 'hourly') & (time1 >= min_date_cams):
        p_thrs_cams = thr0_cams + incr_cams * np.array([1,2,3])
        thresholds['CAMS lower'] = np.quantile(anom_score_train.dropna(), 1-p_thrs_cams) * 2
        thresholds['CAMS upper'] = np.quantile(anom_score_train.dropna(), p_thrs_cams) * 2
    
    if res == 'hourly':
        half_range = (df_train[par].max() - df_train[par].min()) / 2
        thresholds['range lower'] = df_train[par].min() - half_range
        thresholds['range upper'] = df_train[par].max() + half_range
    
    thresholds.set_index(np.array([1,2,3]), inplace=True)
    
    # Prepare data for monthly plot
    mtp = len(df_test.index.month.unique()) # months to predict
    if res == 'hourly':
        n_meas = df_test[par].groupby(pd.Grouper(freq='1M',label='left')).count()
        n_meas.index = n_meas.index + timedelta(days=1)
    else:
        n_meas = df_all['n_meas']
    df_monplot = df_mon[df_mon.index <= df_test.index[-1]].copy() # exclude data after target period
    n_meas = n_meas[n_meas.index <= df_test.index[-1]]
    df_monplot['n'] = np.nan
    df_monplot.loc[df_monplot.index.isin(n_meas.index),'n'] = n_meas
    if content is not None:
        if time0.month != ltime0.month: # remove first month to forecast when when tz > UTC
            mtp -= 1
        elif time1.month != ltime1.month: # remove last month to forecast when when tz < UTC
            df_monplot.drop(index=df_monplot.index[-1], inplace=True)
            mtp -= 1
    
    # Prepare data for cycle plots
    years = df_all[par].dropna().index.year.unique()
    lastyear = years[-1]
    firstyear = lastyear - n_years_max
    years = years[years >= firstyear]
    dc = pd.DataFrame(columns=years)
    vc = pd.DataFrame(columns=years)
    if res == 'hourly':
        df_vc = df_all.loc[df_all.index.year>=firstyear, par].copy()
        if df_test.index.year[0] < df_test.index.year[-1]: # shift data before 01.01 by one year to calculate the diurnal cycle by year
            new_index = pd.Series(df_all.index)
            lastj = df_test.index.dayofyear[-1]
            new_index[df_all.index.dayofyear>lastj] = new_index[df_all.index.dayofyear>lastj] + timedelta(days=365)
            df_all.set_index(new_index, inplace=True)
        df_dc = df_all.loc[(df_all.index.year>=firstyear) & 
                           (df_all.index.dayofyear.isin(df_test.index.dayofyear.unique())), par]
        if content is not None:
            if tz != 'UTC': 
                df_dc.index = df_dc.index.shift(int(tz[3:6]), freq='H') # convert back to local time
        # Calculate diurnal cycle and variability cycle
        for iy in years:
            df_year = df_dc[df_dc.index.year==iy]
            dc[iy] = df_year.groupby(df_year.index.hour).mean().round(2)
            dc[str(iy)+'_n'] = df_year.groupby(df_year.index.hour).count()
            df_year = df_vc[df_vc.index.year==iy]
            vc_year = df_year.groupby(df_year.index.month).std().round(2)
            to_nan = df_mon.loc[df_mon.index.year==iy, par].isna().values
            vc_year[to_nan] = np.nan
            vc[iy] = vc_year
    
    # Wrap everything together and convert to json format for storage
    test_cols = [par] if (par == 'co2') | (time1 < min_date_cams) else [par, par+'_cams']
    out = [par, cod, res, is_new, mtp, time0, time1, lastyear,
           df_test[test_cols].to_json(date_format='iso', orient='columns'),
           df_mon.to_json(date_format='iso', orient='columns'),
           df_monplot.to_json(date_format='iso', orient='columns'),
           y_pred.to_json(date_format='iso', orient='index'),
           y_pred_mon.to_json(date_format='iso', orient='index'),
           anom_score.to_json(date_format='iso', orient='index'),
           score_test.to_json(date_format='iso', orient='index'),
           thresholds.to_json(orient='columns'),
           dc.to_json(orient='columns'),
           vc.to_json(orient='columns')]
    
    return out


@callback(Output('input-data', 'data', allow_duplicate=True), # this fires the dashboard
          Output('date-range', 'start_date', allow_duplicate=True), # reset start date
          Output('date-range', 'end_date', allow_duplicate=True), # reset end date
          Output('upload-data', 'contents', allow_duplicate=True), # reset uploaded file
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
        
    start_time = datetime.now()
    data = get_data(cod, par, hei, date0, date1, tz, content, filename)
    stop_time = datetime.now()
    
    if data is None:
        return None, None, None, None
    
    out = [cod, par, hei, date0, date1, tz, content, filename]
    
    # Write log
    if data != []:
        param, cod, res, is_new, mtp, time0, time1 = data[:7]
        write_log(logpath, 
                  [datetime.isoformat(start_time, sep=' ', timespec='seconds'),
                   cod, par, hei, time0, time1, int(is_new),
                   (stop_time-start_time).seconds])
    
    return out, None, None, None


# Flag hourly data for plot and export
def process_hourly(param, df_test, anom_score, score_test, thresholds, cams_on, selected_q, time1):
    # Flag data after Sub-LOF
    df_test['Flag LOF'] = 0
    thr_yellow = thresholds.loc[selected_q,'LOF']
    thr_red = thr_yellow * 2
    flags_yellow = df_test[(score_test[0]>thr_yellow) & (score_test[0]<=thr_red)]
    flags_red = df_test[score_test[0]>thr_red]
    df_test.loc[df_test.index.isin(flags_yellow.index), 'Flag LOF'] = 1
    
    # Flag extreme outliers that exceed historical records by half the historical range (red flags only)
    flags_red = pd.concat([flags_red, df_test[(df_test[param]>thresholds.loc[1,'range upper']) |
                                              (df_test[param]<thresholds.loc[1,'range lower'])]])
    flags_red = flags_red.groupby(flags_red.index).first() # drop duplicates
    df_test.loc[df_test.index.isin(flags_red.index), 'Flag LOF'] = 2
    
    # Flag data after CAMS (yellow only; two yellow flags [LOF+CAMS] trigger a red flag)
    if cams_on & (param != 'co2') & (time1 >= min_date_cams):
        df_test['Flag CAMS'] = 0
        anom_score_test = anom_score[anom_score.index.isin(df_test.index)]
        flags_cum_yellow = (anom_score_test < thresholds.loc[selected_q,'CAMS lower']) | \
            (anom_score_test > thresholds.loc[selected_q,'CAMS upper'])
        flags_cum_yellow = pd.Series(df_test.index[flags_cum_yellow])
        for fl in flags_cum_yellow: # flags must be extended to the entire window used by cumulative_score
            t0 = max(fl-timedelta(hours=window_size_cams-1), df_test.index[0])
            additional_times = pd.Series(pd.date_range(t0, fl, freq='H'))
            flags_cum_yellow = pd.concat([flags_cum_yellow, additional_times], ignore_index=True)
        flags_cum_yellow.drop_duplicates(inplace=True)
        flags_cum_yellow.sort_values(inplace=True)
        df_test.loc[df_test.index.isin(flags_cum_yellow), 'Flag CAMS'] = 1
        df_test.loc[df_test[param].isna(), 'Flag CAMS'] = 0
        i_yellow = (df_test['Flag LOF'] == 1) & (df_test['Flag CAMS'] != 1)
        i_red = (df_test['Flag LOF'] + df_test['Flag CAMS']) > 1
    else:
        flags_cum_yellow = None
        i_yellow = df_test['Flag LOF'] == 1
        i_red = df_test['Flag LOF'] == 2
        
    return df_test, flags_cum_yellow, i_yellow, i_red


@callback(Output('graph-hourly', 'figure'),
          Output('div-switch', 'style'),
          Output('div-hourly', 'style'),
          Output('div-hourly-settings', 'style'),
          Output('div-hourly-info', 'style'),
          Output('div-text-1', 'style'),
          Input('points-switch', 'on'),
          Input('cams-switch-1', 'on'),
          Input('threshold-slider', 'value'),
          Input('bin-slider', 'value'),
          Input('input-data', 'data'))
def update_figure_1(points_on, cams_on, selected_q, bin_size, input_data):   
    if input_data == []:
        raise PreventUpdate
        
    if input_data is None:     
        return empty_plot('No data available in the selected period - Try choosing a longer period'), \
            {'display':'none'}, {'display':'block'}, {'display':'none'}, {'display':'none'}, \
            {'display':'none'}
    
    # Get input data from browser memory
    cod, param, hei, date0, date1, tz, content, filename = input_data
    
    # Get data from cache
    cache_data = get_data(cod, param, hei, date0, date1, tz, content, filename)
    
    if cache_data == []:
        return empty_plot('Impossible to read uploaded file - Try to change the time format or check the wiki'), \
            {'display':'none'}, {'display':'block'}, {'display':'none'}, {'display':'none'}, \
            {'display':'none'}
            
    param, cod, res, is_new, mtp, time0, time1, lastyear, df_test, df_mon, df_monplot, \
        y_pred, y_pred_mon, anom_score, score_test, thresholds, dc, vc = cache_data        
        
    if res == 'monthly':
        return empty_plot('No hourly data available'), {'display':'flex'}, {'display':'block'}, \
            {'display':'none'}, {'display':'none'}, {'display':'flex', 'margin-top':'50px'}
        
    df_test = pd.read_json(df_test, orient='columns')
    y_pred = pd.read_json(y_pred, orient='index', typ='series')
    anom_score = pd.read_json(anom_score, orient='index', typ='series')
    score_test = pd.read_json(score_test, orient='index')
    thresholds = pd.read_json(thresholds, orient='columns')
    
    # Prepare data for hourly plot
    df_test, flags_cum_yellow, i_yellow, i_red = \
        process_hourly(param, df_test, anom_score, score_test, thresholds, cams_on, selected_q, time1)
    
    # Create data frame for pie chart
    df_pie = pd.DataFrame({'color':['green','yellow','red'],
                           'name':['normal','anomalous','very anomalous'],
                           'value':[df_test[param].count()-i_red.sum()-i_yellow.sum(), 
                                    i_yellow.sum(), i_red.sum()]})
    
    # Define plot panels
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'rowspan': 2}, {'type': 'domain'}], [None, {}]],
                        column_widths=[0.75, 0.25], row_heights=[0.5, 0.5],
                        horizontal_spacing=0.075, vertical_spacing=0.075,
                        subplot_titles=(' ','',''))
    
    # Plot measurements
    if points_on:
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test[param], mode='markers', marker_color='black',
                                 marker_size=3, hoverinfo='skip', name='Measurements'),
                      row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test[param], mode='lines', line_color='black',
                                 line_width=2, hoverinfo='skip', name='Measurements'),
                      row=1, col=1)

    # Plot CAMS
    if cams_on & (param != 'co2') & (time1 >= min_date_cams):
        if points_on:
            fig.add_trace(go.Scatter(x=df_test.index, y=df_test[param+'_cams'], mode='markers',
                                     hoverinfo='skip', marker_color='silver', marker_size=2,
                                     name='CAMS'),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='markers', 
                                     hoverinfo='skip', marker_color='orange', marker_size=2.5,
                                     name='CAMS + ML'),
                          row=1, col=1)            
        else:
            fig.add_trace(go.Scatter(x=df_test.index, y=df_test[param+'_cams'], mode='lines', 
                                     hoverinfo='skip', line_color='silver', line_width=0.75,
                                     name='CAMS'),
                          row=1, col=1) 
            fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines', 
                                     hoverinfo='skip', line_color='orange', line_width=1.5,
                                     name='CAMS + ML'),
                          row=1, col=1)           
            
        # Plot shaded areas for significant differences
        if len(flags_cum_yellow) > 0:
            flags_cum = (flags_cum_yellow - df_test.index[0]).astype('int64')/3.6e12 # convert to number of hours from time 0
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
                t_flag = df_test.index[0] + timedelta(hours=fl)
                if i == i_blocks.size-1:
                    n = np.sum(flags_cum_diff[last_pos+1:])
                else:
                    n = np.sum(flags_cum_diff[i_blocks[i]+1:i_blocks[i+1]])
                    last_pos = i_blocks[i+1]
                t1 = t_flag + timedelta(hours=n)
                fig.add_vrect(x0=t_flag, x1=t1, fillcolor='gold', opacity=0.3, line_width=0,
                              row=1, col=1)
        
    # Plot flags
    fig.add_trace(go.Scatter(x=df_test.index[i_yellow], y=df_test.loc[i_yellow,param], 
                             mode='markers', marker_symbol='circle-open', marker_line_width=3,
                             marker_size=8, marker_color='yellow', name='Yellow flags (LOF)'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_test.index[i_red], y=df_test.loc[i_red,param], 
                             mode='markers', marker_symbol='circle-open', marker_line_width=3,
                             marker_size=8, marker_color='red', name='Red flags (LOF)'),
                  row=1, col=1)
    
    # Plot pie chart
    fig.add_trace(go.Pie(labels=df_pie['name'], values=df_pie['value'], 
                         marker={'colors':df_pie['color']}, showlegend=False),
                  row=1, col=2)
    
    # Plot histogram
    fig.add_trace(go.Histogram(x=df_test[param], histnorm='probability', 
                               name='Measurements', marker_color='black',
                               xbins=dict(size=bin_size), showlegend=False),
                  row=2, col=2)
    if cams_on & (param != 'co2') & (time1 >= min_date_cams):
        fig.add_trace(go.Histogram(x=df_test[param+'_cams'], histnorm='probability', 
                                   name='CAMS', marker_color='silver',
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
        {'display':'flex', 'padding-left':'25px'}, \
        {'display':'flex', 'margin-top':'50px'}
        
        
@callback(Output('export-csv-hourly', 'data'),
          Output('btn_csv_hourly', 'n_clicks'),
          Input('btn_csv_hourly', 'n_clicks'),
          Input('cams-switch-1', 'on'),
          Input('threshold-slider', 'value'),
          Input('input-data', 'data'),
          prevent_initial_call=True)
def export_csv_hourly(n_clicks, cams_on, selected_q, input_data):
    if (n_clicks == 0) or (n_clicks is None):
        raise PreventUpdate
    
    # Get input data from browser memory
    cod, param, hei, date0, date1, tz, content, filename = input_data
    
    # Get data from cache                      
    param, cod, res, is_new, mtp, time0, time1, lastyear, df_test, df_mon, df_monplot, \
        y_pred, y_pred_mon, anom_score, score_test, thresholds, dc, vc = \
            get_data(cod, param, hei, date0, date1, tz, content, filename)
    
    df_test = pd.read_json(df_test, orient='columns')
    y_pred = pd.read_json(y_pred, orient='index', typ='series')
    anom_score = pd.read_json(anom_score, orient='index', typ='series')
    score_test = pd.read_json(score_test, orient='index')
    thresholds = pd.read_json(thresholds, orient='columns')
    
    # Get flags
    df_test, flags_cum_yellow, i_yellow, i_red = \
        process_hourly(param, df_test, anom_score, score_test, thresholds, cams_on, selected_q, time1)  
        
    # Define filename for export file
    outfile = cod + '_' + param + '_' + \
        str(df_test.index[0])[:10].replace('-','') + '-' + \
        str(df_test.index[-1])[:10].replace('-','') + '_' + \
        str(selected_q) + '_hourly.csv'
        
    # Create data frame for export
    df_exp = pd.DataFrame({'Time':df_test.index, 
                           'Value':df_test[param],
                           'CAMS':np.nan,
                           'CAMS + ML':np.nan,
                           'Flag LOF':df_test['Flag LOF'], 
                           'Flag CAMS':np.nan,
                           'Strictness':selected_q})
    if cams_on & (param != 'co2') & (time1 >= min_date_cams):
        df_exp['CAMS'] = df_test[param+'_cams']
        df_exp['CAMS + ML'] = y_pred
        df_exp['Flag CAMS'] = df_test['Flag CAMS']
    else:
        df_exp.drop(columns=['CAMS','CAMS + ML','Flag CAMS'], inplace=True)
    export = add_header(df_exp, param, is_new, cams_on, time1)

    return dict(content=export, filename=outfile), 0


@app.callback(Output('collapse-1', 'is_open'),
              [Input('info-button-1', 'n_clicks')],
              [State('collapse-1', 'is_open')])
def toggle_collapse_1(n, is_open):
    if n:
        return not is_open
    return is_open


# Fit SARIMA model and flag monthly data for plot and export
def process_monthly(df_monplot, mtp, max_length, selected_trend, param):
    df_monplot['prediction'] = np.nan
    df_monplot['upper'] = np.nan
    df_monplot['lower'] = np.nan
    df_monplot['flag'] = False
    
    # Make forecast with SARIMA
    df_sar, fcst, confidence_int, flags = \
        forecast_sm(df_monplot, mtp, max_length, selected_trend)
    df_monplot['prediction'] = fcst
    df_monplot['upper'] = confidence_int['upper '+param]
    df_monplot['lower'] = confidence_int['lower '+param]
    df_monplot.loc[df_monplot.index.isin(flags.index), 'flag'] = True
    
    return df_monplot, df_sar, fcst, confidence_int, flags


@callback(Output('graph-monthly', 'figure'),
          Output('div-monthly', 'style'),
          Output('div-text-2', 'style'),
          Input('points-switch', 'on'),
          Input('cams-switch-2', 'on'),
          Input('trend-radio', 'value'),
          Input('length-slider', 'value'),
          Input('input-data', 'data'))
def update_figure_2(points_on, cams_on, selected_trend, max_length, input_data):
    if input_data == []:
        raise PreventUpdate
              
    if input_data is None:
        return empty_plot(''), {'display':'none'}, {'display':'none'}

    # Get input data from browser memory
    cod, param, hei, date0, date1, tz, content, filename = input_data
    
    # Get data from cache
    cache_data = get_data(cod, param, hei, date0, date1, tz, content, filename)

    if cache_data == []:
        return empty_plot(''), {'display':'none'}, {'display':'none'}
            
    param, cod, res, is_new, mtp, time0, time1, lastyear, df_test, df_mon, df_monplot, \
        y_pred, y_pred_mon, anom_score, score_test, thresholds, dc, vc = cache_data   
        
    df_mon = pd.read_json(df_mon, orient='columns')
    df_monplot = pd.read_json(df_monplot, orient='columns')
    y_pred_mon = pd.read_json(y_pred_mon, orient='index', typ='series')
    
    # Prepare data for monthly plot
    df_monplot, df_sar, fcst, confidence_int, flags = \
        process_monthly(df_monplot, mtp, max_length, selected_trend, param)
    
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
    if cams_on & (param != 'co2') & (time1 >= min_date_cams):
        df_cams = df_monplot[param+'_cams'].iloc[-mtp:]
        fig.add_trace(go.Scatter(x=y_pred_mon.index, y=y_pred_mon, mode='markers', 
                                  marker_color='royalblue', marker_size=11, marker_symbol='star', 
                                  name='CAMS + ML'))
        fig.add_trace(go.Scatter(x=df_cams.index, y=df_cams, mode='markers', 
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
    
    return fig, {'display':'block'}, {'display':'flex', 'margin-top':'75px'}


@callback(Output('export-csv-monthly', 'data'),
          Output('btn_csv_monthly', 'n_clicks'),
          Input('btn_csv_monthly', 'n_clicks'),
          Input('cams-switch-2', 'on'),
          Input('trend-radio', 'value'),
          Input('trend-radio', 'options'),
          Input('length-slider', 'value'),
          Input('input-data', 'data'),
          prevent_initial_call=True)
def export_csv_monthly(n_clicks, cams_on, selected_trend, label_trend, max_length, input_data):
    if (n_clicks == 0) or (n_clicks is None):
        raise PreventUpdate

    # Get input data from browser memory
    cod, param, hei, date0, date1, tz, content, filename = input_data
    
    # Get data from cache                      
    param, cod, res, is_new, mtp, time0, time1, lastyear, df_test, df_mon, df_monplot, \
        y_pred, y_pred_mon, anom_score, score_test, thresholds, dc, vc = \
            get_data(cod, param, hei, date0, date1, tz, content, filename)
        
    df_mon = pd.read_json(df_mon, orient='columns')
    df_monplot = pd.read_json(df_monplot, orient='columns')
    y_pred_mon = pd.read_json(y_pred_mon, orient='index', typ='series')
    
    label_trend = pd.DataFrame(label_trend)
    label_trend = label_trend['label'][label_trend['value']==selected_trend].item()
    
    # Get SARIMA prediction and flags
    df_monplot, df_sar, fcst, confidence_int, flags = \
        process_monthly(df_monplot, mtp, max_length, selected_trend, param)
        
    # Define filename for export file
    outfile = cod + '_' + param + '_' + \
        str(df_monplot.index[-mtp])[:7].replace('-','') + '-' + \
        str(df_monplot.index[-1])[:7].replace('-','') + '_' + \
        str(max_length) + '_' + label_trend.replace(' ','').lower() + '_monthly.csv'
      
    # Create data frame for export
    length = np.min([df_mon.shape[0], max_length*12 + mtp])
    df_exp = pd.DataFrame({'Time':df_monplot.index[-mtp:], 
                            'Value':df_monplot[param].iloc[-mtp:],
                            'N measurements':df_monplot['n'].iloc[-mtp:],
                            'SARIMA (best estimate)':df_monplot['prediction'].iloc[-mtp:],
                            'SARIMA (lower limit)':df_monplot['lower'].iloc[-mtp:],
                            'SARIMA (upper limit)':df_monplot['upper'].iloc[-mtp:],
                            'CAMS':np.nan,
                            'CAMS + ML':np.nan,
                            'Flag':0, 
                            'Years used':int((length-mtp)/12),
                            'Trend':label_trend})
    df_exp.loc[df_exp['Time'].isin(df_monplot.index[df_monplot['flag']]), 'Flag'] = 1
    if cams_on & (param != 'co2') & (time1 >= min_date_cams):
        df_exp['CAMS'] = df_monplot[param+'_cams'].iloc[-mtp:]
        df_exp['CAMS + ML'] = y_pred_mon
    else:
        df_exp.drop(columns=['CAMS','CAMS + ML'], inplace=True)
    export = add_header(df_exp, param, is_new, cams_on, time1)

    return dict(content=export, filename=outfile), 0


@app.callback(Output('collapse-2', 'is_open'),
              [Input('info-button-2', 'n_clicks')],
              [State('collapse-2', 'is_open')])
def toggle_collapse_2(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(Output('graph-cycles', 'figure'),
          Output('div-cycles', 'style'),
          Output('div-text-3', 'style'),
          Input('points-switch', 'on'),
          Input('n-years', 'value'),
          Input('input-data', 'data'))
def update_figure_3(points_on, n_years, input_data):
    if input_data == []:
        raise PreventUpdate
              
    if input_data is None:
        return empty_plot(''), {'display':'none'}, {'display':'none'}
    
    # Get input data from browser memory
    cod, param, hei, date0, date1, tz, content, filename = input_data
    
    # Get data from cache
    cache_data = get_data(cod, param, hei, date0, date1, tz, content, filename)

    if cache_data == []:
        return empty_plot(''), {'display':'none'}, {'display':'none'}
            
    param, cod, res, is_new, mtp, time0, time1, lastyear, df_test, df_mon, df_monplot, \
        y_pred, y_pred_mon, anom_score, score_test, thresholds, dc, vc = cache_data
        
    df_test = pd.read_json(df_test, orient='columns')
    df_mon = pd.read_json(df_mon, orient='columns')
    df_monplot = pd.read_json(df_monplot, orient='columns')
    dc = pd.read_json(dc, orient='columns')
    vc = pd.read_json(vc, orient='columns')
     
    # Choose years
    firstyear = lastyear - n_years
    targetyear = df_test.index.year[-1]
    years = vc.columns[vc.columns>=firstyear]
    years_for_mean = years[years!=targetyear]
    
    # Calculate multiyear averages
    if res != 'monthly':
        dc['multiyear'] = dc[list(map(str,years_for_mean))].mean(axis=1).round(2).values
        vc.columns = vc.columns.astype(str)
        vc['multiyear'] = vc[list(map(str,years_for_mean))].mean(axis=1).round(2).values
    df_mon_new = df_monplot.iloc[-mtp:,:]
    df_mon_sel = df_mon[df_mon.index.year>=firstyear].copy()
    df_mon_my = df_mon_sel[~df_mon_sel.index.isin(df_mon_new.index)].copy()
    sc = df_mon_my[[param]].groupby(df_mon_my.index.month).mean().round(2)
    
    # Define plot panels and line colors
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.075,
                        subplot_titles=('Diurnal cycle', 'Seasonal cycle', 'Variability cycle'))
    colors = ['#f0f921', '#fdb42f','#ed7953', '#cc4778', '#9c179e', '#5c01a6', '#0d0887'] # plasma palette
    label = 'Your data' if is_new else 'Selected period'
    period_label = 'Mean  ' + str(np.min(years_for_mean)) + '-' + str(np.max(years_for_mean))
    n_years_for_mean = len(years_for_mean)
       
    # Plot diurnal cycle
    if res != 'monthly':
        x_labels = pd.date_range('2020-01-01', periods=24, freq='h')
        i_col = len(colors) - len(years_for_mean)
        for iy in years_for_mean:
            if points_on:
                fig.add_trace(go.Scatter(x=x_labels, y=dc[str(iy)], mode='markers',
                                          marker_size=4, marker_color=colors[i_col], 
                                          customdata=dc[str(iy)+'_n'],
                                          hovertemplate='%{y} (N=%{customdata})',
                                          name=str(iy), showlegend=False),
                              row=1, col=1)                   
            else:
                fig.add_trace(go.Scatter(x=x_labels, y=dc[str(iy)], mode='lines',
                                          line_width=1, line_color=colors[i_col], 
                                          customdata=dc[str(iy)+'_n'],
                                          hovertemplate='%{y} (N=%{customdata})',
                                          name=str(iy), showlegend=False),
                              row=1, col=1)
            i_col += 1
        if n_years_for_mean > 1:
            if points_on:
                fig.add_trace(go.Scatter(x=x_labels, y=dc['multiyear'], mode='markers',
                                          marker_size=8, marker_color='royalblue',
                                          name=period_label, showlegend=False),
                              row=1, col=1)                
            else:
                fig.add_trace(go.Scatter(x=x_labels, y=dc['multiyear'], mode='lines',
                                          line_width=5, line_color='royalblue',
                                          name=period_label, showlegend=False),
                              row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=dc[str(targetyear)], mode='markers', 
                                  marker_color='black', marker_size=10, marker_symbol='circle',
                                  customdata=dc[str(targetyear)+'_n'],
                                  hovertemplate='%{y} (N=%{customdata})',
                                  name=label, showlegend=False),
                      row=1, col=1)
    else:
        fig = empty_subplot(fig, 'No hourly data available', 1, 1)
    
    # Plot seasonal cycle
    i_col = len(colors) - len(years_for_mean)
    for iy in years_for_mean:
        df_mon_year = df_mon_sel[df_mon_sel.index.year==iy]
        if points_on:
            fig.add_trace(go.Scatter(x=df_mon_year.index.month, y=df_mon_year[param], mode='markers',
                                      marker_size=4, marker_color=colors[i_col], 
                                      name=str(iy)),
                          row=1, col=2)
        else:
            fig.add_trace(go.Scatter(x=df_mon_year.index.month, y=df_mon_year[param], mode='lines',
                                      line_width=1, line_color=colors[i_col], 
                                      name=str(iy)),
                          row=1, col=2)
        i_col += 1
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
    fig.add_trace(go.Scatter(x=df_mon_new.index.month, y=df_mon_new[param], mode='markers', 
                              marker_color='black', marker_size=10, 
                              marker_symbol='circle', name=label),
                  row=1, col=2)
    
    # Plot seasonal cycle of variability
    if res != 'monthly':
        vc_test = vc.loc[vc.index.isin(df_test.index.month[df_test.index.year==targetyear]), str(targetyear)]
        if df_test.index.year[0] < df_test.index.year[-1]: # test period is across two calendar years
            vc_test_add = vc.loc[vc.index.isin(df_test.index.month[df_test.index.year==targetyear-1]), str(targetyear-1)]
            vc_test = pd.concat([vc_test_add, vc_test]).iloc[-mtp:]
        i_col = len(colors) - len(years_for_mean)
        for iy in years_for_mean:
            if points_on:
                fig.add_trace(go.Scatter(x=vc.index, y=vc[str(iy)], mode='markers',
                                          marker_size=4, marker_color=colors[i_col], 
                                          name=str(iy), showlegend=False),
                              row=1, col=3)
            else:
                fig.add_trace(go.Scatter(x=vc.index, y=vc[str(iy)], mode='lines',
                                          line_width=1, line_color=colors[i_col], 
                                          name=str(iy), showlegend=False),
                              row=1, col=3)
            i_col += 1                
        if n_years > 1:
            if points_on:
                fig.add_trace(go.Scatter(x=vc.index, y=vc['multiyear'], mode='markers',
                                          marker_size=8, marker_color='royalblue',
                                          name=period_label, showlegend=False),
                              row=1, col=3)
            else:
                fig.add_trace(go.Scatter(x=vc.index, y=vc['multiyear'], mode='lines',
                                          line_width=5, line_color='royalblue',
                                          name=period_label, showlegend=False),
                              row=1, col=3)
        fig.add_trace(go.Scatter(x=vc_test.index, y=vc_test, mode='markers', 
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
    fig.update_xaxes(title_text='Month', ticktext=month_names, tickvals=np.arange(1,13), 
                      range=[0,13], showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text=param.upper() + ' mole fraction ' + units,
                      row=1, col=2)
    if res != 'monthly':
        fig.update_xaxes(title_text='Time ' + 'UTC' if tz is None else tz, tickformat='%H:%M', row=1, col=1)
        fig.update_yaxes(title_text=param.upper() + ' mole fraction ' + units,
                          row=1, col=1)
        fig.update_xaxes(title_text='Month', ticktext=month_names, tickvals=np.arange(1,13), 
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
    
    return fig, {'display':'block'}, {'display':'flex', 'margin-top':'75px'}


@callback(Output('export-csv-dc', 'data'),
          Output('btn_csv_dc', 'n_clicks'),
          Input('btn_csv_dc', 'n_clicks'),
          Input('n-years', 'value'),
          Input('input-data', 'data'),
          prevent_initial_call=True)
def export_csv_dc(n_clicks, n_years, input_data):
    if (n_clicks == 0) or (n_clicks is None):
        raise PreventUpdate
        
    # Get input data from browser memory
    cod, param, hei, date0, date1, tz, content, filename = input_data
    
    # Get data from cache                      
    param, cod, res, is_new, mtp, time0, time1, lastyear, df_test, df_mon, df_monplot, \
        y_pred, y_pred_mon, anom_score, score_test, thresholds, dc, vc = \
            get_data(cod, param, hei, date0, date1, tz, content, filename)
        
    if res == 'monthly':
        raise PreventUpdate
        
    df_test = pd.read_json(df_test, orient='columns')
    dc = pd.read_json(dc, orient='columns')
    vc = pd.read_json(vc, orient='columns')
    
    # Choose years
    firstyear = lastyear - n_years
    targetyear = df_test.index.year[-1]
    years = vc.columns[vc.columns>=firstyear]
    years_for_mean = years[years!=targetyear]
    
    # Calculate multiyear average
    dc['multiyear'] = dc[list(map(str,years_for_mean))].mean(axis=1).round(2).values
           
    # Define filename for export
    outfile = cod + '_' + param + '_' + \
        str(firstyear) + '-' + str(lastyear) + '_diurnal-cycle.csv'
    
    # Create data frame for export
    label = 'Your data' if is_new else 'Selected period'
    period_label = 'Mean  ' + str(np.min(years_for_mean)) + '-' + str(np.max(years_for_mean))
    n_years_for_mean = len(years_for_mean)
    df_exp = pd.DataFrame({'Hour':dc.index, 
                            period_label:dc['multiyear'],
                            label:dc[str(targetyear)],
                            'N days':dc[str(targetyear)+'_n']})
    for iy in years_for_mean:
        df_exp[str(iy)] = dc[str(iy)]
    if n_years_for_mean == 1: # remove multi-year average if there is only one year
        df_exp.drop(columns=[period_label], inplace=True)
    export = add_header(df_exp, param, False, False, time1)

    return dict(content=export, filename=outfile), 0


@callback(Output('export-csv-sc', 'data'),
          Output('btn_csv_sc', 'n_clicks'),
          Input('btn_csv_sc', 'n_clicks'),
          Input('n-years', 'value'),
          Input('input-data', 'data'),
          prevent_initial_call=True)
def export_csv_sc(n_clicks, n_years, input_data):
    if (n_clicks == 0) or (n_clicks is None):
        raise PreventUpdate
        
    # Get input data from browser memory
    cod, param, hei, date0, date1, tz, content, filename = input_data
    
    # Get data from cache                      
    param, cod, res, is_new, mtp, time0, time1, lastyear, df_test, df_mon, df_monplot, \
        y_pred, y_pred_mon, anom_score, score_test, thresholds, dc, vc = \
            get_data(cod, param, hei, date0, date1, tz, content, filename)
    
    df_test = pd.read_json(df_test, orient='columns')
    df_mon = pd.read_json(df_mon, orient='columns')
    df_monplot = pd.read_json(df_monplot, orient='columns')
    vc = pd.read_json(vc, orient='columns')
    
    # Choose years
    firstyear = lastyear - n_years
    targetyear = df_test.index.year[-1]
    years = vc.columns[vc.columns>=firstyear]
    years = years[years!=targetyear]
    
    # Calculate multiyear average
    df_mon_new = df_monplot.iloc[-mtp:,:]
    df_mon_sel = df_mon[df_mon.index.year>=firstyear].copy()
    df_mon_my = df_mon_sel[~df_mon_sel.index.isin(df_mon_new.index)].copy()
    sc = df_mon_my[[param]].groupby(df_mon_my.index.month).mean().round(2)
        
    # Define filename for export 
    outfile = cod + '_' + param + '_' + \
        str(firstyear) + '-' + str(lastyear) + '_seasonal-cycle.csv'
        
    # Create data frame for export
    label = 'Your data' if is_new else 'Selected period'
    period_label = 'Mean  ' + str(np.min(years)) + '-' + str(np.max(years))
    n_years_for_mean = len(years)
    df_exp = pd.DataFrame({'Month':np.arange(1,13), 
                            period_label:np.nan,
                            label:np.nan})
    df_exp.set_index(df_exp['Month'], inplace=True)
    df_exp.loc[df_mon_new.index.month, label] = df_mon_new[param].values
    df_exp.loc[sc.index, period_label] = sc[param]   
    for iy in years:
        df_mon_year = df_mon_sel[df_mon_sel.index.year==iy]
        df_exp[str(iy)] = np.nan
        df_exp.loc[df_mon_year.index.month, str(iy)] = df_mon_year[param].values
    if n_years_for_mean == 1: # remove multi-year average if there is only one year
        df_exp.drop(columns=[period_label], inplace=True)
    export = add_header(df_exp, param, False, False, time1)

    return dict(content=export, filename=outfile), 0


@callback(Output('export-csv-vc', 'data'),
          Output('btn_csv_vc', 'n_clicks'),
          Input('btn_csv_vc', 'n_clicks'),
          Input('n-years', 'value'),
          Input('input-data', 'data'),
          prevent_initial_call=True)
def export_csv_vc(n_clicks, n_years, input_data):
    if (n_clicks == 0) or (n_clicks is None):
        raise PreventUpdate
        
    # Get input data from browser memory
    cod, param, hei, date0, date1, tz, content, filename = input_data
    
    # Get data from cache                      
    param, cod, res, is_new, mtp, time0, time1, lastyear, df_test, df_mon, df_monplot, \
        y_pred, y_pred_mon, anom_score, score_test, thresholds, dc, vc = \
            get_data(cod, param, hei, date0, date1, tz, content, filename)
        
    if res == 'monthly':
        raise PreventUpdate
        
    df_test = pd.read_json(df_test, orient='columns')
    vc = pd.read_json(vc, orient='columns')
    
    # Choose years
    firstyear = lastyear - n_years
    targetyear = df_test.index.year[-1]
    years = vc.columns[vc.columns>=firstyear]
    years_for_mean = years[years!=targetyear]
        
    # Calculate multiyear average
    vc.columns = vc.columns.astype(str)
    vc['multiyear'] = vc[list(map(str,years_for_mean))].mean(axis=1).round(2).values
            
    # Define filename for export
    outfile = cod + '_' + param + '_' + \
        str(firstyear) + '-' + str(lastyear) + '_variability-cycle.csv'
      
    # Create data frame for export
    label = 'Your data' if is_new else 'Selected period'
    period_label = 'Mean  ' + str(np.min(years_for_mean)) + '-' + str(np.max(years_for_mean))
    n_years_for_mean = len(years_for_mean)
    vc_test = vc.loc[vc.index.isin(df_test.index.month[df_test.index.year==targetyear]), str(targetyear)]
    if df_test.index.year[0] < df_test.index.year[-1]: # test period is across two calendar years
        vc_test_add = vc.loc[vc.index.isin(df_test.index.month[df_test.index.year==targetyear-1]), str(targetyear-1)]
        vc_test = pd.concat([vc_test_add, vc_test]).iloc[-mtp:]      
    df_exp = pd.DataFrame({'Month':np.arange(1,13), 
                            period_label:vc['multiyear'],
                            label:vc_test})
    df_exp.set_index(df_exp['Month'], inplace=True)
    for iy in years_for_mean:
        df_exp[str(iy)] = vc[str(iy)]
    if n_years_for_mean == 1: # remove multi-year average if there is only one year
        df_exp.drop(columns=[period_label], inplace=True)
    export = add_header(df_exp, param, False, False, time1)

    return dict(content=export, filename=outfile), 0


@app.callback(Output('collapse-3', 'is_open'),
              [Input('info-button-3', 'n_clicks')],
              [State('collapse-3', 'is_open')])
def toggle_collapse_3(n, is_open):
    if n:
        return not is_open
    return is_open
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)