

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.fertilizer import fertilizer_adv
import requests
import config
import pickle
import io
import torch
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

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']



def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

df = pd.read_csv('Data/crop_recommendation.csv')

Q1 = df.quantile(0)
Q3 = df.quantile(1)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

target ='label'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
model = pipeline.fit(X_train, y_train)
y_pred_rf = model.predict(X_test)
# mcc_rf = matthews_corrcoef(X_test, y_pred_rf)
# print (mcc_rf[0])

pipeline1 = make_pipeline(StandardScaler(),DecisionTreeClassifier())
model = pipeline1.fit(X_train, y_train)
y_pred_dt = model.predict(X_test)
# mcc_dt = matthews_corrcoef(X_test, y_pred_dt)

pipeline2 = make_pipeline(StandardScaler(),SVC())
model = pipeline2.fit(X_train, y_train)
y_pred_svm = model.predict(X_test)
# mcc_svm = matthews_corrcoef(X_test, y_pred_svm)


pipeline4 = make_pipeline(StandardScaler(),GaussianNB())
model = pipeline4.fit(X_train, y_train)
y_pred_nb = model.predict(X_test)
# mcc_nb = matthews_corrcoef(X_test, y_pred_nb)

pipeline5 = make_pipeline(StandardScaler(),KNeighborsClassifier())
model = pipeline5.fit(X_train, y_train)
y_pred_knn = model.predict(X_test)
# mcc_knn = matthews_corrcoef(X_test, y_pred_knn)

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


yield_prediction_model_path = 'models/RandomForest.pkl'
yield_prediction_model = pickle.load(
    open(yield_prediction_model_path, 'rb'))


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



@ app.route('/yield-predict', methods=['POST'])
def yield_predict():
    title = 'Smart Farming - Yield Prediction'

    if request.method == 'POST':
        
        area = float(request.form['area'])
        Crop = request.form.ge("crop")
        state = request.form.ge("district")
        city = request.form.get("city")
        data = np.array([[area, Crop, state, city,]])

        if data != None:
            my_prediction = yield_prediction_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('yield-result.html', y_prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)




if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    
