from datetime import datetime

from pandas import DataFrame, to_datetime


def parse_timestamps(df: DataFrame) -> tuple[datetime, datetime]:
    return tuple(to_datetime(df.unstack().values, unit="s", utc=True).to_pydatetime())
