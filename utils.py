import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import classification_report

CARDINALS = [
    "N", "NNE", "NE", "ENE", 
    "E", "ESE", "SE", "SSE", 
    "S", "SSW", "SW", "WSW", 
    "W", "WNW", "NW", "NNW"
]

DEFAULT_VALS = {
    "Month": np.nan,
    "Location": np.nan,
    "MinTemp": 10,
    "MaxTemp": 20,
    "Rainfall": -1,
    "RainfallADJ": -1, 
    "Evaporation": -1,
    "Sunshine": -1,
    "WindGustDir": -2,
    "WindGustDir_x_comp": -2,
    "WindGustDir_y_comp": -2,
    "WindGustSpeed": 40,
    "WindDir9am": -2,
    "WindDir9am_x_comp": -2,
    "WindDir9am_y_comp": -2,
    "WindDir3pm": -2,
    "WindDir3pm_x_comp": -2,
    "WindDir3pm_y_comp": -2,
    "WindSpeed9am": 10,
    "WindSpeed3pm": 10,
    "Humidity9am": 70,
    "Humidity3pm": 60,
    "Pressure9am": 1015,
    "Pressure3pm": 1015,
    "Cloud9am": 4,
    "Cloud3pm": 4,
    "Temp9am": 15,
    "Temp3pm": 20,
    "RainToday": -1,
    "lat": 25.2744,
    "lng": 133.7751
}


LOCATIONS = {
    "BadgerysCreek": {"lat": -33.8829, "lng": 150.7609},
    "Cobar": {"lat": -31.4958, "lng": 145.8389},
    "Dartmoor": {"lat": -37.9144, "lng": 141.2730},
    "MelbourneAirport": {"lat": -37.6690, "lng": 144.8410},
    "MountGinini": {"lat": -35.5294, "lng": 148.7723},
    "Nhil": {"lat": -36.3328, "lng": 141.6503},
    "NorahHead": {"lat": -33.2833, "lng": 151.5667},
    "NorfolkIsland": {"lat": -29.0408, "lng": 167.9547},
    "PearceRAAF": {"lat": -31.6676, "lng": 116.0292},
    "PerthAirport": {"lat": -31.9385, "lng": 115.9672},
    "SalmonGums": {"lat": -32.9815, "lng": 121.6438},
    "SydneyAirport": {"lat": -33.9399, "lng": 151.1753},
    "Tuggeranong": {"lat": -35.4244, "lng": 149.0888},
    "Uluru": {"lat": -25.3444, "lng": 131.0369},
    "Walpole": {"lat": -34.9777, "lng": 116.7338},
    "Watsonia": {"lat": -37.7080, "lng": 145.0830},
    "Williamtown": {"lat": -32.8150, "lng": 151.8428},
    "Witchcliffe": {"lat": -34.0261, "lng": 115.1003}
}


def evaluate_binary_clf(X_test, y_test, model, **kwargs):
    y_prob = model.predict_proba(X_test)
    if "y_pred" not in kwargs:
        y_pred = model.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(8, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    skplt.metrics.plot_roc(y_test, y_prob, figsize=(10, 6), ax=ax1)
    skplt.metrics.plot_precision_recall(y_test, y_prob, figsize=(10, 6), ax=ax2)
    ax1.grid()
    ax2.grid()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    skplt.metrics.plot_cumulative_gain(y_test, y_prob, figsize=(10, 6), ax=ax1)
    skplt.metrics.plot_lift_curve(y_test, y_prob, figsize=(10, 6), ax=ax2)
    skplt.metrics.plot_ks_statistic(y_test, y_prob, figsize=(10, 6))
    plt.grid()
    print(classification_report(y_test, y_pred))
    
    
def get_location_data():
    cities = pd.read_csv(
        "./simplemaps_worldcities_basicv1.73/worldcities.csv",
        usecols=["city", "lat", "lng", "country"]
    )
    cities = (cities[cities["country"] == "Australia"]
              .drop(columns=["country"])
              .reset_index(drop=True)
              .sort_values(by="city", ignore_index=True))

    cities["city"] = cities["city"].astype("string").str.replace(" ", "")

    locations = pd.DataFrame(LOCATIONS).transpose().reset_index().rename(columns={"index": "city"})
    locations = pd.concat([cities, locations])
    locations.reset_index(inplace=True, drop=True)
    assert locations.loc[273]['city'] == "Richmond"
    assert round(locations.loc[273]['lng']) == 143
    locations.drop([273], inplace=True)  # this is not the Richmond with the weather station
    locations.drop_duplicates(subset=['city'], keep=False)
    locations.reset_index(inplace=True, drop=True)
    return locations
    

def feature_transformer(df):
    df['MinTemp'] = np.minimum(df['MinTemp'], df['Temp9am'])
    df['MinTemp'] = np.minimum(df['MinTemp'], df['Temp3pm'])
    df['MaxTemp'] = np.maximum(df['MaxTemp'], df['Temp9am'])
    df['MaxTemp'] = np.maximum(df['MaxTemp'], df['Temp3pm'])
    df["FEAT1"] = df["MaxTemp"] - df["MinTemp"]
    df["FEAT2"] = (df["Temp9am"] - df["MinTemp"])/(df["MaxTemp"] - df["MinTemp"])
    df["FEAT3"] = (df["Temp3pm"] - df["MinTemp"])/(df["MaxTemp"] - df["MinTemp"])
    df["FEAT4"] = df["Pressure3pm"] - df["Pressure9am"]
    df["FEAT5"] = df["Humidity3pm"] - df["Humidity9am"]
    df["FEAT6"] = np.log(df["Rainfall"] + 3)*np.exp(df["Sunshine"]/4)
    df["FEAT7"] = (df["Evaporation"]*df["Sunshine"]).clip(-250, 250)
    df["FEAT8"] = df["WindGustSpeed"]*df["WindGustDir_x_comp"]
    df["FEAT9"] = df["WindGustSpeed"]*df["WindGustDir_y_comp"]
    df["FEAT10"] = df["WindSpeed9am"]*df["WindDir9am_x_comp"]
    df["FEAT11"] = df["WindSpeed3pm"]*df["WindDir3pm_x_comp"]
    df["FEAT12"] = df["WindSpeed9am"]*df["WindDir9am_y_comp"]
    df["FEAT13"] = df["WindSpeed3pm"]*df["WindDir3pm_y_comp"]
    df["FEAT14"] = (df["Pressure3pm"]*df["Humidity3pm"]/10000) + df["Cloud3pm"]
    df["FEAT15"] = (df["Rainfall"] - df["Evaporation"])/(df["Sunshine"] + 2)
    df["FEAT16"] = df["Temp3pm"]*df["Cloud3pm"]/10
    return df