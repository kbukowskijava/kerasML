# Stworzona dwuwarstwowa sieć neuronowa, z wyliczeniem błędu RMSE dla zbioru treningowego/testowego i podsumowanie błędu dla zbioru testowego - tp/tn/fn/fp, recall.
# Dane mogą być sztucznie wygenerowane np: waga-wzrost albo wiek-liczba ludności,
# tak naprawdę dowolny zbiór danych.
# Można pracować na gotowych danych dostępnych w internecie.
# Do tego jakieś wykresy. Proszę również o wycenę takiego projektu.

##################################################################
# DATASETS:
# name of the dataset: weather.csv
# description: This dataset contains 14 different features such as air temperature, atmospheric pressure, and humidity. These were collected every 10 minutes, beginning in 2003.
#             For efficiency, you will use only the data collected between 2009 and 2016.
# name of the dataset: books.csv
# description: More than 30,000 books in spanish of different narratives with image and ISBN and other features.
#             Given in three different formats: CSV, XLSX and JSON.
###################################################################


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import windowGenerator as wg
import logging
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 8
json_file_path = "config.json"  # ścieżka do pliku konfiguracyjnego

# załadowanie pliku JSON do dict'a config
with open(json_file_path) as json_file:
    config = json.load(json_file)


def read_csv(csvFilePath):
    return pd.read_csv(csvFilePath)


def show_tpr_graphs(dataFrame):
    try:
        logging.info("Przygotowywanie wykresów...")
        plot_cols = ['T (degC)', 'p (mbar)', 'rh (%)']
        plot_features = dataFrame[plot_cols]
        plot_features.index = dataFrame['Date Time']
        _ = plot_features.plot(subplots=True)
        plot_features = dataFrame[plot_cols][:480]
        plot_features.index = dataFrame['Date Time'][:480]
        _ = plot_features.plot(subplots=True)
        plt.show()
        logging.info("Wykresy wygenerowane poprawnie!")
    except Exception as e:
        print(f"Pojawił się błąd podczas rysowania wykresów danych treningowych. Powód: {e}")


def main():
    # odczytanie danych do trenowania z pliku CSV
    logging.info("odczytanie danych do trenowania z pliku CSV")
    train_df = read_csv("weather.csv")

    # odfiltrowanie najważniejszych kolumn
    # filterArray = ['title', 'isbn']
    logging.info("odfiltrowanie najważniejszych kolumn")
    filterArray = ['Date Time', 'p (mbar)', 'T (degC)', 'rh (%)', 'VPmax (mbar)']
    train_df = train_df.filter(items=filterArray)
    train_df['Date Time'] = pd.to_datetime(train_df['Date Time'])
    print(train_df.dtypes)
    logging.info(
        f"Załadowano dane z plik CSV.\n Dataframe head:\n{train_df.head(config['amountOfHeadRows'])}\nSprawdzenie poprawności filtrowania...")

    # sprawdzenie poprawności filtrowania:
    if train_df.shape[1] == len(filterArray):
        logging.info("Odfiltrowywanie zakończone sukcesem!")
    else:
        raise Exception('Sprawdz, czy nie ma literowek w filtrach <filterArray>')

    show_tpr_graphs(train_df)

    # zadaniem nominalnym tego zbioru jest przewidzenie temperatury
    # zatem niezbędne będzie oddzielenie cech od etykiet


if __name__ == '__main__':
    main()
