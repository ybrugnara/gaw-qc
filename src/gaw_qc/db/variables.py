from enum import Enum


class GawVars(Enum):
    ch4 = "ch4"
    co = "co"
    co2 = "co2"
    o3 = "o3"


class GawUnits(Enum):
    ch4 = "ppb"
    co = "ppb"
    co2 = "ppm"
    o3 = "ppb"


class GawSources(Enum):
    ch4 = "World Data Centre for Greenhouse Gases (WDCGG - gaw.kishou.go.jp)"
    co = "World Data Centre for Greenhouse Gases (WDCGG - gaw.kishou.go.jp)"
    co2 = "World Data Centre for Greenhouse Gases (WDCGG - gaw.kishou.go.jp)"
    o3 = "World Data Centre for Reactive Gases (WDCRG - www.gaw-wdcrg.org)"
