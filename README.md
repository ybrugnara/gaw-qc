![](https://github.com/ybrugnara/gaw-qc/blob/main/src/gaw_qc/assets/logos/gaw-qc_logo.png)

This is an abridged version of the GAW-QC web app developed at Empa.

The app allows the user to upload of one csv or xls file with hourly mole fraction measurements made at a Global Atmosphere Watch (GAW) station. One such file is provided in the folder `examples` (note that the timestamps are in UTC). In alternative, one can analyze the historical data available in the database. The main goal is to facilitate the detection of anomalous measurements through data-driven algorithms based on the archives of measurements and numerical forecasts by the Copernicus Atmosphere Monitoring Service (CAMS).

More information is available in the wiki of this repository.

### How to run
Once downloaded, the easiest and recommended way to run the app is by using [Docker](https://www.docker.com).

From a terminal go to the directory where the Dockerfile is located an run:
`docker compose up --build`.

The application will be available at http://localhost:8000.

### Database
The data are stored in a [SQLite](https://www.sqlite.org/) database. This repository includes a minimal database with only one station (Jungfraujoch) and one variable (methane).

![](https://github.com/ybrugnara/gaw-qc/blob/main/src/gaw_qc/assets/images/gaw_db.png)

### Customization
Several settings can be changed in dedicated Python and CSS scripts:

- `src/gaw_qc/config/app_config.py` contains general settings such as the path of the database or the graphical theme of the app
- `src/gaw_qc/models/model_config.py` contains the model hyper-parameters for anomaly detection and downscaling of the CAMS forecasts
- `src/gaw_qc/plotting/aesthetics.py` contains several graphical parameters used in the plots
- `src/gaw_qc/assets/css/style.css` contains the CSS rules used to design the layout

### Contributors
This software was developed at [Empa](https://www.empa.ch) by Yuri Brugnara (Laboratory for Air Pollution / Environmental Technology) and Simone Baffelli (Scientific IT) in the framework of the [Global Atmosphere Watch (GAW) programme](https://www.empa.ch/gaw).

### Acknowledgments
This repository includes data provided by the [World Data Centre for Greenhouse Gases (WDCGG)](https://gaw.kishou.go.jp) and the [Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/).

The GAW programme at Empa is supported by the Federal Office of Meteorology and Climatology MeteoSwiss.

To use the latest full version of GAW-QC visit https://www.empa.ch/gaw.

![](https://github.com/ybrugnara/gaw-qc/blob/main/src/gaw_qc/assets/logos/Logo_Empa.png)
![](https://github.com/ybrugnara/gaw-qc/blob/main/src/gaw_qc/assets/logos/wmo-gaw.png)

