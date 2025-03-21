import base64
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO, StringIO
from pandas.core.tools.datetimes import guess_datetime_format
from pathlib import Path
from typing import Literal, NoReturn

from gaw_qc.log_utils.decorators import log_function
logger = logging.getLogger(__name__)


@dataclass
class UserInput:
    """Data submitted by the user
    cod: GAW ID
    par: gas species
    hei: index of sampling height (height can have duplicates - hence an index)
    date_start: initial date
    date_end: final date
    tz: time zone
    content: content of uploaded file
    filename: name of uploaded file
    """    
    cod: str
    par: str
    hei: int
    date_start: str
    date_end: str
    tz: str
    content: str | None
    filename: str | None

    @log_function(logger)
    def parse_data(self) -> pd.DataFrame | list[NoReturn]:
        """Parse the data uploaded by the user
        """
        content_type, content_string = self.content.split(",")
        decoded = base64.b64decode(content_string)
        fn = Path(self.filename)
        param = self.par.lower()
        
        # Read into a data frame
        try:
            if fn.suffix in [".csv", ".txt"]:
                df_up = pd.read_csv(
                    StringIO(decoded.decode("utf-8")),
                    sep=None,
                    #header=None,
                    #skiprows=1,
                    #usecols=[0, 1],
                    quoting=3,
                    #names=["Time", param],
                    engine="python",
                )
            else:
                df_up = pd.read_excel(
                    BytesIO(decoded),
                    #header=None,
                    #skiprows=1,
                    #usecols=[0, 1],
                    #names=["Time", param],
                )
        except:
            logger.error("Could not recognize file format of uploaded file")
            return []

        # Get rid of quotes
        df_up.replace('"', "", regex=True, inplace=True)

        # Rename columns
        df_up.rename(
            columns={df_up.columns[0]: "Time", df_up.columns[1]: param},
            inplace=True
        )

        # Deal with time format
        fmt = guess_datetime_format(df_up["Time"].iloc[0])
        try:
            df_up["Time"] = df_up["Time"].apply(datetime.strptime, args=(fmt,))
        except:
            try:
                if fmt[1] == "m":  # change mdy format to dmy
                    fmt = fmt.replace("d", "m")
                    fmt = list(fmt)
                    fmt[1] = "d"
                    fmt = "".join(fmt)
                df_up["Time"] = df_up["Time"].apply(datetime.strptime, args=(fmt,))
            except:
                logger.error("Could not recognize time format of uploaded file")
                return []
        logger.info(f"Time format of uploaded file is {fmt}")
        df_up["Time"] = df_up["Time"].dt.round("H")  # round to nearest hour
        df_up.sort_values(["Time"], inplace=True)  # sort chronologically
        df_up.set_index("Time", inplace=True)
        if self.tz != "UTC":  # convert time to UTC
            df_up.index = df_up.index.shift(-int(self.tz[3:6]), freq="H")
    
        # Deal with decimal separator
        for c in df_up.columns[1:]:
            try:
                df_up[c] = pd.to_numeric(df_up[c].replace(",", ".", regex=True))
            except:
                logger.error("Could not convert data column to numeric")
                return []

        # Deal with missing values
        df_up.loc[df_up[param] < 0, param] = np.nan  # assign NaN to negative values
        df_up = df_up.resample("H").asfreq()  # fill missing periods with NaN

        # Add empty column for number of measurements
        df_up["n_meas"] = np.nan
    
        return df_up
    
    
@dataclass
class ProcessedData:
    """Data used to produce the dashboard (stored on server cache)
    cod: GAW ID
    par: gas species
    hei: index of sampling height (height can have duplicates - hence an index)
    res: time resolution
    is_new: whether data were uploaded by the user
    time_start: start time of target period (UTC)
    time_end: end time of tarted period (UTC)
    offset: difference from UTC in hours
    last_year: last year to show in the 3rd panel
    test_data: hourly measurements to be inspected, together with CAMS and CAMS+ data (converted to json)
    monthly_data: monthly means of measurements (converted to json)
    monthly_data_plot: data to be plotted in the 2nd panel and 3rd panel (converted to json)
    cams_plus_mon: monthly means of downscaled CAMS forecasts (converted to json)
    anom_score_cams: anomaly score from difference CAMS+ - measurements (converted to json)
    anom_score_lof: anomaly score from Sub-LOF (converted to json)
    thresholds: table of thresholds to define flags (converted to json)
    diurnal_cycle: data for plot of diurnal cycle (converted to json)
    seasonal_cycle: data for plot of seasonal cycle (converted to json)
    var_cycle: data for plot of variability cycle (converted to json)
    """
    cod: str
    par: str
    hei: int
    res: Literal["hourly", "monthly"]
    is_new: bool
    time_start: datetime
    time_end: datetime
    offset: float
    last_year: int
    test_data: str
    monthly_data: str
    monthly_data_plot: str
    cams_plus_mon: str
    anom_score_cams: str
    anom_score_lof: str
    thresholds: str
    diurnal_cycle: str
    seasonal_cycle: str
    var_cycle: str


@dataclass
class PlottingData:
    """
    Data used for the cycles plots and exports (third panel)
    label: label of test data for legend
    period_label: label of multi-year period for legend
    y_title: title of y axis
    years: years that are plotted other than the target year (as strings)
    dc: diurnal cycle data
    sc: seasonal cycle data
    vc: variability cycle data
    """
    label: str
    period_label: str
    y_title: str
    years_for_mean: pd.Index
    diurnal_cycle: pd.DataFrame
    seasonal_cycle: pd.DataFrame
    var_cycle: pd.DataFrame
