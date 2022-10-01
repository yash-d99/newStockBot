from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures

from PIL import Image
import pandas as pd
import yfinance as yf
import os, contextlib
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import math
import pandas_datareader as web
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

finviz_url = 'https://finviz.com/quote.ashx?t='

def neuralNetwork(dataframe):

 #Importing data and creating datasets

  #!/usr/bin/env python
# coding: utf-8

# In[1]:


  start_time = time.time()
  plt.style.use('fivethirtyeight')


  # In[2]:

  
  #Reading in the data
  #df = pd.read_csv('AAPL.csv')
  #start = self.start
  #end = self.end
  
  #df=web.DataReader('NASDAQ:AAPL', data_source='google')

  #df=
  # In[3]:

  df=dataframe
  #Create a new dataframe with only the close column
  data = df.filter(['Close'])

  #convert the dataframe to a numpy array
  dataset = data.values

  #Get the number of rows to train the model only
  training_data_len = math.ceil( len(dataset) * .8)


  # In[4]:


  #Scale the data
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)

  #print(scaled_data)


  # In[5]:


  #Create the training dataset
  #Create the scaled training dataset
  train_data = scaled_data[0:training_data_len , :]
  #Split the data into X_train and y_train datasets
  X_train = []
  y_train = []

  for i in range(60, len(train_data)):
      X_train.append(train_data[i-60:i, 0])
      y_train.append(train_data[i, 0])
      if i<= 61:
          print(X_train)
          print(y_train)
          print()


  # In[6]:


  #Convert the x_train and y_train to numpy arrays
  X_train, y_train = np.array(X_train), np.array(y_train)


  # In[7]:


  #Reshape the data
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  X_train.shape


  # In[8]:


  #Build the LSTM model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))


  # In[9]:


  #Compile the model
  model.compile(optimizer='adam', loss='mean_squared_error', metrics =['accuracy'])


  # In[10]:


  #Create the testing dataset
  #Create a new array containing scaled values from index 1543 to 2003
  test_data = scaled_data[training_data_len - 60: , :]
  #Create the data sets X_test and y_test
  X_test = []
  y_test = dataset[training_data_len:, :]
  for i in range(60, len(test_data)):
      X_test.append(test_data[i-60:i, 0])


  # In[11]:


  #Convert the data to a nump array
  X_test = np.array(X_test)


  # In[12]:


  #Reshape the test data
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


  # In[24]:


  #Train the model
  model.fit(X_train, y_train, validation_data=(X_test,y_test),batch_size=1, epochs=1)


  # In[25]:


  #Get the models predicted price values
  predictions = model.predict(X_test)
  predictions = scaler.inverse_transform(predictions)


  # In[26]:


  #Getting the root mean square error (RSME)
  rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
  rmse


  # In[27]:


  train = data[:training_data_len]
  valid = data[training_data_len:]
  valid['Predictions'] = predictions
  #Visualize the model
  plt.figure(figsize=(16,8))
  plt.title('Model')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price USD ($)', fontsize=18)
  plt.plot(train['Close'])
  plt.plot(valid[['Close', 'Predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  plt.show()


  # In[17]:


  #Show the valid and predicted prices
  valid


  # In[18]:


  #Get the quote
  stock_quote = dataframe
  #Create a new dataframe
  new_df = stock_quote.filter(['Close'])
  #Get the last 60 day closing price values and convert the dataframe to an array
  last_60_days = new_df[-60:].values
  print(last_60_days)
  for i in range(30):
  #Scale the data to be values between 0 and 1
      last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
      #Create an empty list
      x_test = []
      #Append the past 60 days 
      x_test.append(last_60_days_scaled)
      #Convert the x_test dataset to a numpy array
      x_test = np.array(x_test)
      #Reshape the data
      x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
      #et the predicted scaled price
      pred_price = model.predict(x_test)
      #Undo scaling
      pred_price = scaler.inverse_transform(pred_price)

      last_60_days=np.append(last_60_days,pred_price)
      

      #print(pred_price)

  final30=last_60_days[-30:]
  #print(final30)
  #print ("My program took", time.time() - start_time, "to run")


  # In[19]:


  apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
  #print(apple_quote2)

  return final30


  # <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=b057f1f9-4d23-4ada-8e2d-d76419dd8e14' target="_blank">
  # <img style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
  # Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>


def sentimentAnalysis(stock):
  news_tables = {}
  
  url = finviz_url + stock[0]

  req = Request(url=url, headers={'user-agent': 'my-app'})
  response = urlopen(req)
  
  html = BeautifulSoup(response, 'html')
  news_table = html.find(id='news-table')
  news_tables[stock] = news_table

  parsed_data = []

  for stock, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
      title = row.a.get_text()
      date_data = row.td.text.split(' ')

      if len(date_data) == 1:
        time = date_data[0]
      else:
        date = date_data[0]
        time = date_data[1]
      
      parsed_data.append([stock, date, time, title])

  df = pd.DataFrame(parsed_data, columns=['stock', 'date', 'time', 'title'])

  vader = SentimentIntensityAnalyzer()

  f = lambda title: vader.polarity_scores(title)['compound']
  df['compound'] = df['title'].apply(f)
  
  sum = 0

  print(df['compound'])
  for value in df['compound']:
    sum = sum + value
  

  average = sum / 100


  return average

def stockInput(stocks):
  df = web.DataReader(stocks[0], data_source='yahoo',start='1800-01-01')
  arrayPredictions = np.around(neuralNetwork(df), 2)
  listPredictions = arrayPredictions.tolist()
  converted_list = [str(element) for element in listPredictions]
  count =0
  day=1
  for values in converted_list:
    converted_list[count]= " Day {}:".format(day) + " $"+values
    count += 1
    day += 1
  stringPredictions = ", \n".join(converted_list)
  sentimentAnalysisAverage = sentimentAnalysis(stocks)
  return stringPredictions, sentimentAnalysisAverage


'''def stockInput(stocks):
  getData([stocks])
  arrayPredictions = np.around(randomForestRegressor('hist/{}.csv'.format(stocks)), 2)
  stringPredictions = np.array_str(arrayPredictions)
  return stringPredictions
  '''