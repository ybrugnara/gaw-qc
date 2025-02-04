import calendar
import pandas as pd
from datetime import datetime, timedelta


def limit_to_one_year(start_time: datetime, end_time: datetime) -> datetime:
    """
    Limit the dates in the interval between `start_time` and
    `end_time` to one year.
    """
    end_date = datetime.strftime(end_time, "%m-%d")
    main_year = start_time.year if end_date < "02-29" else end_time.year
    diff = end_time - start_time
    n_days = 366 if calendar.isleap(main_year) else 365
    if diff > timedelta(days=n_days):
        new_start = end_time - timedelta(days=n_days) + timedelta(hours=1)
    else:
        new_start = start_time

    return new_start


def limit_to_max_length(
        times: pd.DatetimeIndex, max_len: int, start_time: datetime, end_time: datetime
) -> tuple[datetime, datetime]:
    """
    Limit the length of a time index to `max_len` hours that must
    contain a target period between `start_time` and `end_time`.
    Priority is given to more recent times, unless the target period
    ends before the half point of the index.
    :return: start and end time of the index to keep
    """
    times_outside_target = times[(times < start_time) | (times > end_time)]
    n = len(times_outside_target)
    if n <= max_len:
        out = (times[0], times[-1])
    elif end_time < times[int(len(times)/2)]:
        out = (times_outside_target[0], times_outside_target[max_len])
    else:
        out = (times_outside_target[-max_len], times_outside_target[-1])

    return out
