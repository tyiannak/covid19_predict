import csv
import numpy as np
import argparse
import plotly
import plotly.graph_objs as go
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor


def load_data():
    dates_c = []
    dates_d = []
    with open("total_cases.csv", 'r') as fp:
        reader = csv.reader(fp)
        count = 0
        for row in reader:
            if count == 0:
                countries_c = [r.lower() for r in row[1:]]
                cases = [[] for r in row[1:]]
            else:
                dates_c.append(row[0])
                for ir, r in enumerate((row[1:])):
                    if r:
                        cases[ir].append(int(r))
                    else:
                        cases[ir].append(0)
            count += 1

    with open("total_deaths.csv", 'r') as fp:
        reader = csv.reader(fp)
        count = 0
        for row in reader:
            if count == 0:
                countries_d = [r.lower() for r in row[1:]]
                deaths = [[] for r in row[1:]]
            else:
                dates_d.append(row[0])
                for ir, r in enumerate((row[1:])):
                    if r:
                        deaths[ir].append(int(r))
                    else:
                        deaths[ir].append(0)
            count += 1

    if dates_c == dates_d and countries_c == countries_d:
        count = 0
        countries_with_pop = []
        populations = []
        with open("populations.csv", 'r') as fp:
            reader = csv.reader(fp)
            for row in reader:
                if count > 0:
                    if row[0].lower() in countries_c:
                        countries_with_pop.append(row[0].lower())
                        if row[1]:
                            populations.append(float(row[1]))
                        else:
                            populations.append(10**6)  # if no population found
                count += 1
        cases_final, deaths_final = [], []
        for c in countries_with_pop:
            cases_final.append(cases[countries_d.index(c)])
            deaths_final.append(deaths[countries_d.index(c)])

        return dates_c, countries_with_pop, populations, \
               cases_final, deaths_final
    else:
        print("ERROR in data consistency! "
              "The two csvs contain different number of rows or columns! ")


def feature_extraction(countries, cases, deaths, pop, sel_countries_train):
    features, target_cases, target_deaths = [], [], []
    for iS, s in enumerate(sel_countries_train):
#        print(s)
        # get data ...
        cas = cases[countries.index(s)]
        dea = deaths[countries.index(s)]
        # and normalized data for current country
        cas_norm = [10**6 * c / pop[countries.index(s)] for c in cas]
        dea_norm = [10**6 * d / pop[countries.index(s)] for d in dea]

        # get features
        start_window = 6
        for i in range(start_window, len(cas)-1):
            feature_vector = [
                cas_norm[i] - cas_norm[i - 1],
                cas_norm[i] - cas_norm[i - 2],
                cas_norm[i] - 2 * cas_norm[i - 1] + cas_norm[i - 2],
                dea_norm[i] - dea_norm[i - 1],
                dea_norm[i] - dea_norm[i - 2],
                dea_norm[i] - 2 * dea_norm[i - 1] + dea_norm[i - 2],
                # TODO add life exp here and other demographics
            ]
            features.append(feature_vector)
            target_cases.append(cas_norm[i + 1] - cas_norm[i])
            target_deaths.append(dea_norm[i + 1] - dea_norm[i])
#            print(feature_vector, target_cases[-1], target_deaths[-1])
    features = np.array(features)
    target_cases = np.array(target_cases)
    target_deaths = np.array(target_deaths)

    return features, target_cases, target_deaths


def train_model(countries, cases, deaths, pop, train_countries):
    tr_features, tr_cases, tr_deaths = feature_extraction(countries,
                                                          cases,
                                                          deaths, pop,
                                                          train_countries)

    print(tr_features.shape)
#    to_keep = (tr_features.any(axis=1))
#    tr_features = tr_features[to_keep]
#    tr_cases = tr_cases[to_keep]
#    tr_deaths = tr_deaths[to_keep]
#    print(tr_features.shape)
    clf_c = svm.SVR(C=1, kernel="linear")
#    clf_c = KNeighborsRegressor(n_neighbors=19)
    clf_c.fit(tr_features, tr_cases)
    clf_d = svm.SVR(C=1, kernel="linear")
#    clf_d = KNeighborsRegressor(n_neighbors=19)
    clf_d.fit(tr_features, tr_deaths)

    return clf_c, clf_d


def test_model(model, countries, cases, deaths, pop, test_countries):
    te_features, te_cases, te_deaths = feature_extraction(countries,
                                                          cases,
                                                          deaths, pop,
                                                          test_countries)
    cases_pred = model.predict(te_features)
    error = np.mean(np.abs(cases_pred - te_cases))
    print("Error: {0:.3f}".format(error))
    return cases_pred, te_cases


def plot_countries(dates, countries, cases, deaths, selected_countries, pop):
    subplot_titles = []
    for c in selected_countries:
        subplot_titles.append(c + " - cases")
        subplot_titles.append(c + " - deaths")
        subplot_titles.append(c + " - NORM cases (per mil)")
        subplot_titles.append(c + " - NORM deaths (per mil)")

    figs = plotly.subplots.make_subplots(rows=len(selected_countries), cols=4,
                                         subplot_titles=subplot_titles)

    # train the model with ALL countries
    print("Training global model : ")
    svm_global_c, svm_global_d = train_model(countries, cases, deaths,
                                             pop, countries)

    for iS, s in enumerate(selected_countries):
        print(s)
        cas = cases[countries.index(s)]
        dea = deaths[countries.index(s)]
        cas_norm = [10**6 * c / pop[countries.index(s)] for c in cas]
        dea_norm = [10**6 * d / pop[countries.index(s)] for d in dea]
        figs.append_trace(go.Scatter(x=dates, y=cas, showlegend=False),
                          iS + 1, 1)
        figs.append_trace(go.Scatter(x=dates, y=dea, showlegend=False),
                          iS + 1, 2)
        figs.append_trace(go.Scatter(x=dates, y=cas_norm, showlegend=False),
                          iS + 1, 3)
        figs.append_trace(go.Scatter(x=dates, y=dea_norm, showlegend=False),
                          iS + 1, 4)

        # plot predictions
        feature_vector = [
            cas_norm[-1] - cas_norm[-2],
            cas_norm[-1] - cas_norm[-3],
            cas_norm[-1] - 2 * cas_norm[-2] + cas_norm[-3],
            dea_norm[-1] - dea_norm[-2],
            dea_norm[-1] - dea_norm[-3],
            dea_norm[-1] - 2 * dea_norm[-2] + dea_norm[-3],
            # TODO add life exp here and other demographics
        ]
        pred_c = svm_global_c.predict([feature_vector])[0] * \
                 pop[countries.index(s)] / (10**6)
        pred_d = svm_global_d.predict([feature_vector])[0] * \
               pop[countries.index(s)] / (10**6)
        print("Predicted new cases today {0:.1f}".format(pred_c))
        print("Predicted new deaths today {0:.1f}".format(pred_d))

        # attention: use predict before de-nnormalization
        new_c = cas_norm[-1] + svm_global_c.predict([feature_vector])[0]
        new_d = dea_norm[-1] + svm_global_d.predict([feature_vector])[0]
        feature_vector = [
            new_c - cas_norm[-1],
            new_c - cas_norm[-2],
            new_c - 2 * cas_norm[-1] + cas_norm[-2],
            new_d - dea_norm[-1],
            new_d - dea_norm[-2],
            new_d - 2 * dea_norm[-1] + dea_norm[-2],
            # TODO add life exp here and other demographics
        ]
        pred_c = svm_global_c.predict([feature_vector])[0] * \
                 pop[countries.index(s)] / (10**6)
        pred_d = svm_global_d.predict([feature_vector])[0] * \
               pop[countries.index(s)] / (10**6)
        print("Predicted new cases tomorrow {0:.1f}".format(pred_c))
        print("Predicted new deaths tomorrow {0:.1f}".format(pred_d))


    plotly.offline.plot(figs, filename="temp.html", auto_open=True)


def validate(countries, cases, deaths, populations):
    # Training model for cross validation
    sel_countries_1 = [d for d in countries[1::2]]
    sel_countries_2 = [d for d in countries[0::2]]

    svm_cases_1, svm_deaths_1 = train_model(countries, cases, deaths, populations,
                                            sel_countries_1)
    svm_cases_2, svm_deaths_2 = train_model(countries, cases, deaths, populations,
                                            sel_countries_2)

    n_cols = 5
    figs = plotly.subplots.make_subplots(rows=
                                         int((len(countries) / n_cols)) + 1,
                                         cols=n_cols,
                                         subplot_titles=
                                         ["new cases per million in " + s
                                          for s in sel_countries_1 +
                                          sel_countries_2])
    for iS, s in enumerate(sel_countries_1):
        cases_pred_1, cases_true_1 = test_model(svm_cases_1, countries, cases,
                                                deaths, populations, [s])
        mark_prop1 = dict(color='rgba(80, 50, 250, 0.9)',
                          line=dict(color='rgba(150, 180, 80, 1)', width=3))
        mark_prop2 = dict(color='rgba(250, 150, 80, 0.9)',
                          line=dict(color='rgba(150, 180, 80, 1)', width=3))
        figs.append_trace(go.Scatter(x=dates, y=cases_pred_1, name="pred",
                                     marker=mark_prop1),
                          int(iS / n_cols) + 1, iS % n_cols + 1)
        figs.append_trace(go.Scatter(x=dates, y=cases_true_1, name="true",
                                     marker=mark_prop2),
                          int(iS / n_cols) + 1, iS % n_cols + 1 )
    for iS, s in enumerate(sel_countries_2):
        cases_pred_2, cases_true_2 = test_model(svm_cases_2, countries, cases,
                                                deaths, populations, [s])
        mark_prop1 = dict(color='rgba(80, 50, 250, 0.9)',
                          line=dict(color='rgba(150, 180, 80, 1)', width=3))
        mark_prop2 = dict(color='rgba(250, 150, 80, 0.9)',
                          line=dict(color='rgba(150, 180, 80, 1)', width=3))
        figs.append_trace(go.Scatter(x=dates, y=cases_pred_2, name="pred",
                                     marker=mark_prop1),
                          int((iS+len(sel_countries_1)) / n_cols) + 1, (iS+len(sel_countries_1)) % n_cols + 1)
        figs.append_trace(go.Scatter(x=dates, y=cases_true_2, name="true",
                                     marker=mark_prop2),
                          int((iS+len(sel_countries_1)) / n_cols) + 1, (iS+len(sel_countries_1)) % n_cols + 1 )

    plotly.offline.plot(figs, filename="temp2.html", auto_open=True)


def parse_arguments():
    covid = argparse.ArgumentParser(description="")
    covid.add_argument("-c", "--countries", nargs="+")
    covid.add_argument("--chromagram", action="store_true",
                                  help="Show chromagram")
    return covid.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sel_countries = args.countries
    important_countries = ["united states", "china", "germany", "italy", "greece",
                           "spain", "france", "united kingdom", "japan",
                           "south korea", "taiwan", "austria", "netherlands",
                           "canada", "denmark", "ireland"]

    # read the data:
    dates, countries_init, populations_init, cases_init, deaths_init = load_data()

    cases = [c for ic, c in enumerate(cases_init)
             if countries_init[ic] in important_countries]
    deaths = [c for ic, c in enumerate(deaths_init)
              if countries_init[ic] in important_countries]
    populations = [c for ic, c in enumerate(populations_init)
                   if countries_init[ic] in important_countries]
    countries = [c for ic, c in enumerate(countries_init)
                   if countries_init[ic] in important_countries]

    print(len(cases), len(important_countries), len(deaths))


    # get only countries that exist in the data
    sel_countries_final = [s for s in sel_countries if s in countries]

    # plot selected data:
    plot_countries(dates, countries, cases, deaths, sel_countries_final,
                   populations)


    # validate
    validate(countries, cases, deaths, populations)