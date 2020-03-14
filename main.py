import csv
import sys
import time
import numpy as np
import scipy
import argparse
import os


def load_data():
    cases = []
    deaths = []
    dates_c = []
    dates_d = []
    with open("total_cases.csv", 'r') as file:
        reader = csv.reader(file)
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

    with open("total_deaths.csv", 'r') as file:
        reader = csv.reader(file)
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
        return dates_c, countries_d, cases, deaths

    else:
        print("ERROR in data consistency! "
              "The two csvs contain different number of rows or columns! ")

def load_data2():
    with open('total_cases.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        kept2 = [row for row in reader]
    print(kept2)

def parse_arguments():
    covid = argparse.ArgumentParser(description="")
    covid.add_argument("-c", "--countries")
    covid.add_argument("--chromagram", action="store_true",
                                  help="Show chromagram")
    return covid.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dates, countries, cases, deaths = load_data()
    print(cases[countries.index("italy")])

