import logging
from sqlalchemy.engine import Engine
import pandas as pd
from gaw_qc.log_utils.decorators import log_function


logger = logging.getLogger(__name__)


@log_function(logger)
def get_station_stats(engine: Engine) -> list[dict]:
    logger.info(f"Getting metadata for stations")
    q = """
        with hourly_counts as (
    select series_id, count(*) as count, max(time) as last_time, min(time) as first_time from gaw_hourly
    group by series_id
    ),
    monthly_counts as (
    select series_id, count(*) as count,  max(time) as last_time, min(time) as first_time from gaw_monthly
    group by series_id
    ),
    monthly_cams_counts as (
    select series_id, count(*) as count,  max(time) as last_time, min(time) as first_time from cams_monthly
    group by series_id
    ),
    hourly_cams_counts as (
    select series_id, count(*) as count,  max(time) as last_time, min(time) as first_time from cams_hourly
    group by series_id
    )

    select
    h.series_id, h.count as hourly_count, st.name, st.gaw_id, s.variable, s.height,
    max(coalesce(h.first_time, 0),
    coalesce(m.first_time, 0)) as first_time,
    max(coalesce(h.last_time, 0),
    coalesce(0, m.last_time)) as last_time,
    max(coalesce(mc.first_time, 0), coalesce(hc.first_time, 0)) as first_cams_time,
    coalesce(mc.last_time, hc.last_time) as last_cams_time
    from hourly_counts h
    left join monthly_counts m on h.series_id = m.series_id
    left join monthly_cams_counts mc on h.series_id = mc.series_id
    left join hourly_cams_counts hc on h.series_id = hc.series_id
    join series s on h.series_id = s.id
    join stations st on s.gaw_id = st.gaw_id;
    """
    results = pd.read_sql(q, engine)
    return results.to_dict(orient="records")
