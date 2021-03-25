import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import classification_report
from sklearn.linear_model import Ridge
from tensorflow import keras

CARDINALS = [
    "N", "NNE", "NE", "ENE", 
    "E", "ESE", "SE", "SSE", 
    "S", "SSW", "SW", "WSW", 
    "W", "WNW", "NW", "NNW"
]

DEFAULT_VALS = {  # rough estimate/good default replacements for unknown value
    "Month": np.nan,
    "Location": np.nan,
    "MinTemp": 10,
    "MaxTemp": 20,
    "Rainfall": 0,
    "RainfallADJ": 0, 
    "Evaporation": 0,
    "Sunshine": 0,
    "WindGustDir": -2,
    "WindGustDir_x_comp": 0,
    "WindGustDir_y_comp": 0,
    "WindGustSpeed": 40,
    "WindDir9am": 0,
    "WindDir9am_x_comp": 0,
    "WindDir9am_y_comp": 0,
    "WindDir3pm": 0,
    "WindDir3pm_x_comp": 0,
    "WindDir3pm_y_comp": 0,
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
    "lng": 133.7751,
    "FEAT1": 10,
    "FEAT2": 0.5,
    "FEAT3": 1,
    "FEAT4": 0,
    "FEAT5": -20,
    "FEAT8": 0,
    "FEAT9": 0,
    "FEAT10": 0,
    "FEAT11": 0,
    "FEAT12": 0,
    "FEAT13": 0,
    "FEAT14": 0,
    "FEAT15": 1,
    "FEAT16": 1,
    "FEAT17": 0
    
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
    y_prob = kwargs.get("y_prob", model.predict_proba(X_test))
    y_pred = kwargs.get("y_pred", model.predict(X_test))
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


def process_data(df, cardinal_map):
    if "RISK_MM" in df:
        df.drop(columns=["RISK_MM"], inplace=True)
    df.dropna(subset=["RainTomorrow"], inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Location"] = df["Location"].astype("string")
    
    month = df["Date"].dt.month + df["Date"].dt.day/30
    df["Month_x_comp"] = np.cos(month*np.pi/6)
    df["Month_y_comp"] = np.sin(month*np.pi/6)
    
    for col in ("WindGustDir", "WindDir9am", "WindDir3pm"):
        df[col] = df[col].map(cardinal_map)
        df[f"{col}_x_comp"] = np.cos(df[col])
        df[f"{col}_y_comp"] = np.sin(df[col])
        df.drop(columns=[col], inplace=True)

    df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1})
    df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    rain_adj_factor = (df
                       .groupby("Location")
                       .apply(lambda d: d.dropna(subset=['Rainfall'])['Date'].diff().dt.days)
                       .droplevel(0)
                       .fillna(1)
                       .rename("RainADJFactor"))
    df = df.merge(rain_adj_factor, left_index=True, right_index=True, how='left')
    df["RainADJFactor"] = df["RainADJFactor"].fillna(1)
    df["Rainfall"] = df["Rainfall"]/df["RainADJFactor"]
    df.drop(columns=["RainADJFactor"], inplace=True)

    evap_adj_factor = (df
                       .groupby("Location")
                       .apply(lambda d: d.dropna(subset=['Evaporation'])['Date'].diff().dt.days)
                       .droplevel(0)
                       .fillna(1)
                       .rename("EvapADJFactor"))
    df = df.merge(evap_adj_factor, left_index=True, right_index=True, how='left')
    df["EvapADJFactor"] = df["EvapADJFactor"].fillna(1)
    df["Evaporation"] = df["Evaporation"]/df["EvapADJFactor"]
    df.drop(columns=["EvapADJFactor"], inplace=True)

    locations = get_location_data()
    df = df.merge(locations, how="left", left_on="Location", right_on="city").drop(columns=["city"])

    return df
    

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
    df["FEAT14"] = df["Humidity3pm"]/(df["Humidity9am"] + 0.01).clip(0, 5)
    df["FEAT15"] = df["Pressure3pm"]/df["Pressure9am"]
    df["FEAT16"] = np.log((df["WindSpeed3pm"]/(df["WindSpeed9am"]+1)) + 1)
    df["FEAT17"] = np.log10(df["Rainfall"] + 1)
    # df["FEAT14"] = (df["Pressure3pm"]*df["Humidity3pm"]/10000) + df["Cloud3pm"]
    return df


def walk_forward_extrapolate(s):
    """
    Filling nans in a pandas series using a 120-window mean and
    a running standard deviation. Applying on a pandas series
    to use in a groupby operation.
    """
    for position in range(len(s)):
        if pd.isna(s.iloc[position]):
            stdev = np.random.normal(0, s.iloc[: position].std())
            s.iloc[position] = s.iloc[max(position-120, 0): position].mean() + stdev
    return s


def running_zero_count_with_reset(s):
    counter = 0
    new_series = []
    for position in range(len(s)):
        if s.iloc[position] == 0:
            counter += 1
        else:
            counter = 0
        new_series.append(counter)
    new_series = pd.Series(new_series)
    new_series.index = s.index
    return np.exp(0.01*new_series) - 1


def scaler(df):
    """
    Normalising a dataframe using an expanding mean and standard deviation
    to avoid look-ahead bias.
    """
    location = df.pop("Location")
    df = (df - df.expanding().mean())/(df.expanding().std().fillna(0) + 0.01)
    df.drop(df.head(1).index, inplace=True)
    location.drop(location.head(1).index, inplace=True)
    df = df.merge(location, left_index=True, right_index=True)
    return df


def plot_target_on_feature(df, feature, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    df.set_index("Date")[feature].plot(linewidth=0.5, ax=ax)
    df.query("RainTomorrow==1").plot.scatter(x="Date", y=feature, color="red", s=30, ax=ax)
    ax.grid()
    title = title if title is not None else feature
    ax.set_title(title)
    plt.show()
    return fig, ax


def generate_nan_prediction(X_train, X_test, feature, alpha=0.1, n_samples=730):
    """
    Using an L2-regularised linear model to model a feature for 
    filling missing values
    """

    lat_scale = 50
    lng_scale = 200
    temp_scale = 50
    predictors = X_train[["Location", "lat", "lng", "Month_x_comp", "Month_y_comp", "Temp9am"]].head(n_samples).copy()
    predictors.update(predictors.groupby("Location").ffill())  # to help include more locations without look-ahead
    for col in predictors:
        predictors[col] = predictors[col].fillna(DEFAULT_VALS.get(col, np.nan))
    predictors["lat"] /= lat_scale
    predictors["lng"] /= lng_scale
    predictors["Temp9am"] /= temp_scale

    model = Ridge(alpha=alpha, random_state=0).fit(
        predictors.drop(columns=["Location"]),
        X_train.groupby("Location")[feature].ffill().fillna(DEFAULT_VALS.get(feature, np.nan)).head(n_samples)
    )

    print("Model:")
    print(f"{feature} = {round(model.coef_[0]/lat_scale, 3)}LAT"
          f" + {round(model.coef_[1]/lng_scale, 3)}LNG"
          f" + {round(model.coef_[2], 3)}Month_x"
          f" + {round(model.coef_[3], 3)}Month_y"
          f" + {round(model.coef_[4]/temp_scale, 3)}Temp9am"
          f" + {round(model.intercept_, 3)}")
    
    prediction_x_train = (
        model.intercept_
        + model.coef_[0]*X_train["lat"]/lat_scale
        + model.coef_[1]*X_train["lng"]/lng_scale
        + model.coef_[2]*X_train["Month_x_comp"]
        + model.coef_[3]*X_train["Month_y_comp"]
        + model.coef_[4]*X_train["Temp9am"]/temp_scale
    )
    
    prediction_x_test = (
        model.intercept_
        + model.coef_[0]*X_test["lat"]/lat_scale
        + model.coef_[1]*X_test["lng"]/lng_scale
        + model.coef_[2]*X_test["Month_x_comp"]
        + model.coef_[3]*X_test["Month_y_comp"]
        + model.coef_[4]*X_test["Temp9am"]/temp_scale
    )
    
    
    return prediction_x_train, prediction_x_test