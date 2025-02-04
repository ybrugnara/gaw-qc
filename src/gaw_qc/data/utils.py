from dataclasses import fields
from datetime import datetime
from gaw_qc.data.classes import PlottingData, ProcessedData, UserInput
from typing import Type


def list_to_class(
        l: list[str | bool | float | int | datetime | None], 
        C: Type[PlottingData | ProcessedData | UserInput],
) -> PlottingData | ProcessedData | UserInput:
    """
    Insert list of arguments of a dataclass into an instance of that dataclass
    NB: arguments must be in the correct order
    """
    field_names = [field.name for field in fields(C)]
    d = dict(zip(field_names, l))        
    return C(**d)
