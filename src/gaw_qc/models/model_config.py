from pydantic_settings import BaseSettings
from datetime import datetime
from gaw_qc.db.variables import GawVars


class ModelSettings(BaseSettings):
    # Minimum number of hourly values required to calculate monthly means
    n_min: int = 300

    # Maximum number of years that can be processed for the third panel
    n_years_max: int = 7
    
    # Date of earliest CAMS data to use
    min_date_cams: datetime = datetime(2020,1,1)

    # Sub-LOF parameters
    window_size: int = 3
    n_neighbors: int = 100

    # SARIMAX parameters
    sarima_order: list[int] = [1, 0, 0]
    sarima_sorder: list[int] = [0, 1, 1, 12]
    sarima_stationarity: bool = False
    p_conf: float = 0.01

    # Minimum number of months required to fit SARIMAX
    min_months_sarima: int = 36

    # Minimum and maximum number of months used to train the ML model
    # NB: reducing the maximum limit may improve performance
    # NB: the maximum also applies to Sub-LOF
    min_months_ml: int = 12
    max_months_ml: int = 120

    # CAMS-based anomaly score parameter (minimum flaggable window size in hours)
    window_size_cams: int = 50

    # Threshold parameters (thr0 + incr * strictness level)
    n_levels: int = 3
    thr0_lof: float = 0.99
    incr_lof: float = 0.003
    thr0_cams: float = 0.97
    incr_cams: float = 0.01

    # CAMS additional features to use for downscaling
    # NB: an empty string implies no downscaling
    cams_vars: dict[GawVars, str] = {
        GawVars.ch4: "value_tc, u10, v10, t2m, bcod, pm10, tcwv",
        GawVars.co: "value_tc, u10, v10, t2m, bcod, mslp, pm10, tcwv",
        GawVars.o3: "value_tc, u10, v10, t2m, bcod, mslp, pm10, tcwv",
        GawVars.co2: "u10, v10, t2m, bcod, mslp, pm10, tcwv",
        GawVars.n2o: "",
    }


def get_cams_vars(v: GawVars) -> str:
    """
    Get comma-separated string of additional CAMS features that are used
    by the downscaling algorithm
    """
    return ModelSettings().cams_vars[GawVars[v]]
