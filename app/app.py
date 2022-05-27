

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.fertilizer import fertilizer_adv
import requests
import config
import pickle
import io
import torch
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import metrics
from sklearn import utils
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None




app = Flask(__name__)



@ app.route('/')
def home():
    title = 'Smart Farming - Home'
    return render_template('index.html', title=title)

@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Smart Farming - Crop Recommendation'
    return render_template('crop.html', title=title)


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Smart Farming - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

@ app.route('/yield-prediction')
def yield_prediction():
    title = 'Smart Farming - Yield Prediction'
    return render_template('yield.html', title=title)


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Smart Farming - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")
        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)



@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Smart Farming - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_adv[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)



def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


df = pd.read_csv('Data/crop_production.csv')

Q1 = df.quantile(0)
Q3 = df.quantile(1)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

target ='Production'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

@ app.route('/yield-predict', methods=['POST'])
def yield_predict():
    title = 'Smart Farming - Yield Prediction'

    if request.method == 'POST':
        
        area = float(request.form['area'])
        Crop = request.form.get("crop")
        state = request.form.get("state")
        year = float(request.form['year'])
        city = request.form.get("city")
        season = request.form.get('season')
        data = np.array([[state, city, year, season, Crop, area]])

        if data != None:
            from sklearn.svm import SVR
            svr=SVR(kernel='poly',epsilon=1.0)
            my_prediction = svr.predict(data)
            final_prediction = my_prediction[0]

            return render_template('yield-result.html', y_prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)




if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    