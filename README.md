# gaw-qc

Demo version of the Empa dashboard for GAW stations.

The dashboard requires the upload of one csv or xls file with hourly mole fraction measurements of either carbon dioxide (CO2) or methane (CH4) for the station of Jungfraujoch. Two such files are provided in the folder `examples` (note that the timestamps are in UTC).

In alternative, one can analyze the historical data present in the database (period 2015-2023).

### How to run
Once downloaded, the easiest and recommended way to run the app is by using [Docker](https://www.docker.com) (see also `README.Docker.md`).

### Acknowledgments
This demo makes use of data provided by the [World Data Centre for Greenhouse Gases (WDCGG)](https://gaw.kishou.go.jp) and the [Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/).

The code was developed at Empa in the framework of the Global Atmosphere Watch (GAW) programme with the support of the Federal Office of Meteorology and Climatology MeteoSwiss.
