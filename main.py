import csv
import numpy as np
import argparse
import plotly
import plotly.graph_objs as go

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
        return dates_c, countries_with_pop, populations, cases_final, \
               deaths_final
    else:
        print("ERROR in data consistency! "
              "The two csvs contain different number of rows or columns! ")


def plot_countries(dates, countries, cases, deaths, selected_countries, pop):
    subplot_titles = []
    for c in selected_countries:
        subplot_titles.append(c + " - cases")
        subplot_titles.append(c + " - deaths")
        subplot_titles.append(c + " - NORM cases (per mil)")
        subplot_titles.append(c + " - NORM deaths (per mil)")

    figs = plotly.subplots.make_subplots(rows=len(selected_countries), cols=4,
                                         subplot_titles=subplot_titles)
    for iS, s in enumerate(selected_countries):
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

    plotly.offline.plot(figs, filename="temp.html", auto_open=True)


def parse_arguments():
    covid = argparse.ArgumentParser(description="")
    covid.add_argument("-c", "--countries", nargs="+")
    covid.add_argument("--chromagram", action="store_true",
                                  help="Show chromagram")
    return covid.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sel_countries = args.countries

    # read the data:
    dates, countries, populations, cases, deaths = load_data()
    # get only countries that exist in the data
    sel_countries_final = [s for s in sel_countries if s in countries]

    # plot selected data:
    plot_countries(dates, countries, cases, deaths, sel_countries_final, populations)


