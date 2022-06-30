# Stworzona dwuwarstwowa sieć neuronowa, z wyliczeniem błędu RMSE dla zbioru treningowego/testowego i podsumowanie błędu dla zbioru testowego - tp/tn/fn/fp, recall.
# Dane mogą być sztucznie wygenerowane np: waga-wzrost albo wiek-liczba ludności,
# tak naprawdę dowolny zbiór danych.
# Można pracować na gotowych danych dostępnych w internecie.
# Do tego jakieś wykresy. Proszę również o wycenę takiego projektu.

##################################################################
# DATASETS:
# name of the dataset: weather.csv
# description: This dataset contains 14 different features such as air temperature, atmospheric pressure, and humidity. These were collected every 10 minutes, beginning in 2003.
#              For efficiency, you will use only the data collected between 2009 and 2016.
# name of the dataset: books.csv
# description: More than 30,000 books in spanish of different narratives with image and ISBN and other features.
#              Given in three different formats: CSV, XLSX and JSON.
###################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
import seaborn as sns
import windowGenerator as wg
import baseline as base

# wstępne ustawienia wykresów
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 8
json_file_path = "config.json"  # ścieżka do pliku konfiguracyjnego

MAX_EPOCHS = 20  # maksymalna liczba powtórzeń trenowania

# załadowanie pliku JSON do dict'a config
with open(json_file_path) as json_file:
    config = json.load(json_file)


def compile_and_fit(model, window, patience=2):
    # aby ułatwić kompilowanie i uczenie, przygotowana została funkcja
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def print_versions():
    print("Tensorflow version: " + tf.__version__)
    print("Keras version: " + tf.keras.__version__)
    print("Pandas version: " + pd.__version__)


def read_csv(csvFilePath):
    return pd.read_csv(csvFilePath)


def show_tpr_graphs(dataFrame):
    try:
        logging.info("Przygotowywanie wykresów...")
        plot_cols = ['T (degC)', 'p (mbar)', 'rh (%)']
        plot_features = dataFrame[plot_cols]
        plot_features.index = dataFrame['Date Time']
        _ = plot_features.plot(subplots=True)
        # wyrysowanie wykresu dla ostatnich 288 (48h) przypadków - dane są pobierane co 10 minut
        plot_features = dataFrame[plot_cols][:288]
        plot_features.index = dataFrame['Date Time'][:288]
        _ = plot_features.plot(subplots=True)
        plt.show()
        logging.info("Wykresy wygenerowane poprawnie!")
    except Exception as e:
        print(f"Pojawił się błąd podczas rysowania wykresów danych treningowych. Powód: {e}")


def main():
    print_versions()
    # odczytanie danych do trenowania z pliku CSV
    logging.info("odczytanie danych do trenowania z pliku CSV")
    train_df = read_csv("weather.csv")

    # odfiltrowanie najważniejszych kolumn
    logging.info("odfiltrowanie najważniejszych kolumn")
    filterArray = ['Date Time', 'p (mbar)', 'T (degC)', 'rh (%)', 'VPmax (mbar)']
    train_df = train_df.filter(items=filterArray)
    train_df['Date Time'] = pd.to_datetime(train_df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    logging.info(
        f"Załadowano dane z plik CSV.\n Dataframe head:\n{train_df.head(config['amountOfHeadRows'])}\nSprawdzenie poprawności filtrowania...")

    # sprawdzenie poprawności filtrowania:
    if train_df.shape[1] == len(filterArray):
        logging.info("Odfiltrowywanie zakończone sukcesem!")
    else:
        raise Exception('Sprawdz, czy nie ma literowek w filtrach <filterArray>')

    show_tpr_graphs(train_df)

    # zadaniem nominalnym tego zbioru jest przewidzenie temperatury na określony przez użytkownika okres
    # zatem niezbędne będzie oddzielenie cech od etykiet

    filterArray = ['Date Time', 'T (degC)']
    data_to_train = train_df.filter(items=filterArray)
    timestamps_for_data_training = data_to_train['Date Time'].map(pd.Timestamp.timestamp)
    data_to_train.head()

    # jako dane pogodowe, mają wyraźną dzienną i roczną okresowość
    # można uzyskać użyteczne sygnały, używając przekształceń sinus i cosinus, aby wyczyścić sygnały „Pora dnia” i „Pora roku”
    day = 24 * 60 * 60
    year = (365.2425) * day

    data_to_train['Day sin'] = np.sin(timestamps_for_data_training * (2 * np.pi / day))
    data_to_train['Day cos'] = np.cos(timestamps_for_data_training * (2 * np.pi / day))
    data_to_train['Year sin'] = np.sin(timestamps_for_data_training * (2 * np.pi / year))
    data_to_train['Year cos'] = np.cos(timestamps_for_data_training * (2 * np.pi / year))

    plt.plot(np.array(data_to_train['Day sin'])[:228])
    plt.plot(np.array(data_to_train['Day cos'])[:228])
    plt.xlabel('Time [h]')
    plt.title('Time of day signal')
    plt.show()

    # transformacja fouriera
    fft = tf.signal.rfft(data_to_train['T (degC)'])
    f_per_dataset = np.arange(0, len(fft))

    n_samples_h = len(data_to_train['T (degC)'])
    hours_per_year = 24 * 365.2524
    years_per_dataset = n_samples_h / (hours_per_year)

    f_per_year = f_per_dataset / years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 400000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')
    plt.show()

    # wykonanie podziału danych na 70%, 20%, 10% dla zestawów treningowych, walidacyjnych i testowych
    column_indices = {name: i for i, name in enumerate(data_to_train.columns)}

    del data_to_train['Date Time']
    n = len(data_to_train)
    train_df = data_to_train[0:int(n * 0.7)]
    val_df = data_to_train[int(n * 0.7):int(n * 0.9)]
    test_df = data_to_train[int(n * 0.9):]

    num_features = data_to_train.shape[1]

    # Ważne jest, aby skalować funkcje przed uczeniem sieci neuronowej. Normalizacja jest powszechnym sposobem wykonywania tego skalowania: odejmij średnią i podziel przez odchylenie #standardowe każdej cechy.
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (data_to_train - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(data_to_train.keys(), rotation=90)
    plt.show()

    single_step_window = wg.StoreModelData(input_width=1, label_width=1, shift=1, train_df=train_df, val_df=val_df,
                                           test_df=test_df, label_columns=['T (degC)'])

    baseline = base.Baseline(label_index=column_indices['T (degC)'])

    baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}

    val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

    dense_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(dense_model, single_step_window)

    val_performance['Dense'] = dense_model.evaluate(single_step_window.val)
    performance['Dense'] = dense_model.evaluate(single_step_window.test, verbose=0)
    print(history)

if __name__ == '__main__':
    main()
