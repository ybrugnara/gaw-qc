from dataclasses import dataclass
from datetime import datetime
from gaw_qc.db.variables import GawVars


@dataclass
class DataLogRecord:
    station: str
    parameter: GawVars
    height: int
    start: datetime
    end: datetime
    new_data: bool
    duration: int
