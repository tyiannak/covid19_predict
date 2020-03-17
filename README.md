# covid19_predict
A simple Python predictor for covid19 data. 
Note: this code uses simple ML and feature extraction techniques and it does not take into consideration any other parameters (weather-related, socio-economic etc). It's role is for 100% educational purposes for demonstrating how ML can be used to predict time sequence values.

## Methodology
 * Total number of cases (per day) are first normalized by country population.
 * Training data is created by extracting the following features for each data point (day) of the dataset: 1st derivative, 2nd derivative and deltas from days -2 and -3. Same for number of deaths. As a "ground truth" (target) value we use the true relative case increase (new cases) in the next day.
 * An SVM linear regressor is trained using the data of the previous tep.
 * SVM is used to predict (normalized) new cases for tomorrow and the next day (denormalization also needs to applied so that the final outcome is in number of new cases)
 * Predictions are shown in the terminal. The two plots show (a) total cases and deaths (and normalized) (b) new cases and predicted values (simulated results using cross validation)

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
