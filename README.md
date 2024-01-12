# HOW TO USE TENSORFLOW 2.X TO PREDICT BIKE RENTAL USING ANNs (Regression)
<p align="center">
  <img src="Screenshot 2024-01-11 141555.png">
</p>

## 0) INTRODUCTION: PROJECT STATEMENT

<p align="center">
  <img src="Screenshot 2024-01-12 162543.png">
</p>
In pursuit of enhancing bike rental management, this project focuses on predicting rental usage by leveraging key environmental factors such as temperature, humidity, and wind speed. By harnessing the power of machine learning, the objective is to construct a multi-layer perception network. This sophisticated neural network architecture will enable the system to analyze intricate patterns within the input variables, providing a comprehensive understanding of their impact on bike rental trends. The ultimate goal is to develop a predictive model that not only captures the nuances of weather-related influences but also contributes to more efficient and informed decision-making in the bike rental domain. Through the implementation of advanced algorithms, this initiative aims to optimize resource allocation and enhance user experience in the dynamic realm of bike-sharing services.

<p align="center">
  <img src="Screenshot 2024-01-11 142901.png">
</p>

### Data Reference:

1. SuperDataScience
2. Laboratory of Artificial Intelligence and Decision Support (LIAAD), University of Porto INESC Porto, Campus da FEUP Rua Dr. Roberto Frias, 378 4200 - 465 Porto, Portugal

### Data Description:

1. instant: record index
2. dteday : date
3. season : season (1:springer, 2:summer, 3:fall, 4:winter)
4. yr : year (0: 2011, 1:2012)
5. mnth : month ( 1 to 12)
6. hr : hour (0 to 23)
7. holiday : wether day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
8. weekday : day of the week
9. workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
weathersit :

a) Clear, Few clouds, Partly cloudy

b) Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist

c) Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds

d)  Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

1. temp : Normalized temperature in Celsius. The values are divided to 41 (max)
2. hum: Normalized humidity. The values are divided to 100 (max)
3. windspeed: Normalized wind speed. The values are divided to 67 (max)
4. casual: count of casual users
5. registered: count of registered users
6. cnt: count of total rental bikes including both casual and registered

## I) IMPORT LIBRARIES

<p align="center">
  <img src="Screenshot 2024-01-11 142847.png">
</p>

## II) IMPORT DATASETS

We will need to mount your drive using the following commands:

For more information regarding mounting, please check this out: https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory

from google.colab import drive
drive.mount('/content/drive')


Mounted at /content/drive
We have to include the full link to the csv file containing your dataset

bike = pd.read_csv('/content/drive/My Drive/bike_sharing_daily.csv')

bike


<p align="center">
  <img src="Screenshot 2024-01-12 171329.png">
</p>

bike.info()
<p align="center">
  <img src="Screenshot 2024-01-12 172215.png">
</p>

## III) CLEAN UP DATASET

sns.heatmap(bike.isnull())
<p align="center">
  <img src="Screenshot 2024-01-11 142826.png">
</p>

bike = bike.drop(labels = ['instant'], axis = 1)

<p align="center">
  <img src="Screenshot 2024-01-11 142736.png">
</p>

bike.index = pd.DatetimeIndex(bike.dteday)

bike
<p align="center">
  <img src="Screenshot 2024-01-11 142547.png">
</p>


## IV) VISUALIZE DATASET

bike['cnt'].asfreq('W').plot(linewidth = 3)

plt.title('Bike Usage Per week')

plt.xlabel('Week')

plt.ylabel('Bike Rental')

<p align="center">
  <img src="Screenshot 2024-01-11 142521.png">
</p>

bike['cnt'].asfreq('M').plot(linewidth = 3)

plt.title('Bike Usage Per Month')

plt.xlabel('Month')

plt.ylabel('Bike Rental')

<p align="center">
  <img src="Screenshot 2024-01-11 142459.png">
</p>


bike['cnt'].asfreq('Q').plot(linewidth = 3)

plt.title('Bike Usage Per Quarter')

plt.xlabel('Quarter')

plt.ylabel('Bike Rental')

<p align="center">
  <img src="Screenshot 2024-01-11 142438.png">
</p>

sns.pairplot(bike)
<p align="center">
  <img src="Screenshot 2024-01-11 142257.png">
</p>

<p align="center">
  <img src="Screenshot 2024-01-11 142413.png">
</p>

X_numerical = bike[['temp', 'hum', 'windspeed', 'cnt']]

X_numerical
<p align="center">
  <img src="Screenshot 2024-01-11 142212.png">
</p>


sns.pairplot(X_numerical)
<p align="center">
  <img src="Screenshot 2024-01-12 180450.png">
</p>
[ ]
  1
sns.heatmap(X_numerical.corr(), annot =True)
<p align="center">
  <img src="Screenshot 2024-01-11 142017.png">
</p>

## V) CREATE TRAINING AND TESTING DATASET
[ ]
  1
X_cat = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]
[ ]
  1
X_cat
<p align="center">
  <img src="NLP1.png">
</p>

[ ]
  1
  2
  3
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
[ ]
  1
X_cat.shape
account_circle
(731, 32)
[ ]
  1
X_cat = pd.DataFrame(X_cat)
[ ]
  1
X_numerical
account_circle

[ ]
  1
X_numerical = X_numerical.reset_index()
[ ]
  1
X_all = pd.concat([X_cat, X_numerical], axis = 1)
[ ]
  1
X_all
account_circle

[ ]
  1
X_all = X_all.drop('dteday', axis = 1)
[ ]
  1
X_all
account_circle

[ ]
  1
  2
X = X_all.iloc[:, :-1].values
y = X_all.iloc[:, -1:].values
[ ]
  1
X.shape
account_circle
(731, 35)
[ ]
  1
y.shape
account_circle
(731, 1)
[ ]
  1
  2
  3
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = scaler.fit_transform(y)
[ ]
  1
  2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
[ ]
  1
X_train.shape
account_circle
(584, 35)
[ ]
  1
X_test.shape
account_circle
(147, 35)
## VI) TRAIN THE MODEL
[ ]
  1
  2
  3
  4
  5
  6
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(35, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

[ ]
  1
model.summary()
account_circle
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 100)               3600      
                                                                 
 dense_1 (Dense)             (None, 100)               10100     
                                                                 
 dense_2 (Dense)             (None, 100)               10100     
                                                                 
 dense_3 (Dense)             (None, 1)                 101       
                                                                 
=================================================================
Total params: 23,901
Trainable params: 23,901
Non-trainable params: 0
_________________________________________________________________
[ ]
  1
model.compile(optimizer='Adam', loss='mean_squared_error')
[ ]
  1
epochs_hist = model.fit(X_train, y_train, epochs = 20, batch_size = 50, validation_split = 0.2)
account_circle
Epoch 1/20
10/10 [==============================] - 1s 24ms/step - loss: 0.1408 - val_loss: 0.0675
Epoch 2/20
10/10 [==============================] - 0s 5ms/step - loss: 0.0410 - val_loss: 0.0331
Epoch 3/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0199 - val_loss: 0.0213
Epoch 4/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0135 - val_loss: 0.0176
Epoch 5/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0104 - val_loss: 0.0182
Epoch 6/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0095 - val_loss: 0.0169
Epoch 7/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0076 - val_loss: 0.0156
Epoch 8/20
10/10 [==============================] - 0s 6ms/step - loss: 0.0064 - val_loss: 0.0154
Epoch 9/20
10/10 [==============================] - 0s 8ms/step - loss: 0.0057 - val_loss: 0.0159
Epoch 10/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0053 - val_loss: 0.0148
Epoch 11/20
10/10 [==============================] - 0s 6ms/step - loss: 0.0049 - val_loss: 0.0144
Epoch 12/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0047 - val_loss: 0.0143
Epoch 13/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0043 - val_loss: 0.0143
Epoch 14/20
10/10 [==============================] - 0s 5ms/step - loss: 0.0042 - val_loss: 0.0147
Epoch 15/20
10/10 [==============================] - 0s 5ms/step - loss: 0.0042 - val_loss: 0.0149
Epoch 16/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0041 - val_loss: 0.0152
Epoch 17/20
10/10 [==============================] - 0s 5ms/step - loss: 0.0038 - val_loss: 0.0145
Epoch 18/20
10/10 [==============================] - 0s 5ms/step - loss: 0.0035 - val_loss: 0.0159
Epoch 19/20
10/10 [==============================] - 0s 5ms/step - loss: 0.0035 - val_loss: 0.0153
Epoch 20/20
10/10 [==============================] - 0s 7ms/step - loss: 0.0034 - val_loss: 0.0145
## VII) EVALUATE THE MODEL
[ ]
  1
epochs_hist.history.keys()
account_circle
dict_keys(['loss', 'val_loss'])
[ ]
  1
  2
  3
  4
  5
  6
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
account_circle

[ ]
  1
  2
  3
  4
  5
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color = 'g')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')

account_circle

[ ]
  1
  2
  3
y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)

[ ]
  1
  2
  3
  4
plt.plot(y_test_orig, y_predict_orig, "^", color = 'b')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')

account_circle

[ ]
123
k = X_test.shape[1]
n = len(X_test)
n
account_circle
147
[ ]
12345678910111213


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)= 
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

account_circle
RMSE = 855.646 
MSE = 732130.1093405469 
MAE = 614.8364324245323 
R2 = 0.805684986705684 
Adjusted R2 = 0.744414487018287
Author
Lebede Ngartera
