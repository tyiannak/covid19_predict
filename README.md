# covid19_predict
A simple Python predictor for covid19 data

## Methodology
TODO

## Install dependencies
`pip3 install -r requirements.txt`

## Get data

We get the data from [Our World in Data](https://ourworldindata.org). In particular, the total deaths and total cases csv files are downloaded as follow:

`
wget https://covid.ourworldindata.org/data/total_deaths.csv
`

`
wget https://covid.ourworldindata.org/data/total_cases.csv
`

Also, the `populations.csv` file contains the 2013 populations for all countries (a bit outdated, but this is just used for normalization of the COVID19 data). Of course more recent data are more by welcome by contributors :-) 

## Run predictions for a list of selected countries of interest

`
python3 main.py -c greece italy germany "united kingdom" "united states"
`
