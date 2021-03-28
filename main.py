import pandas as pd
import numpy as np

import pydotplus
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import pickle

from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn import linear_model
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Analyzing Bike Sharing Trends', page_icon="./f.png")
st.title('Analyzing Bike Sharing Trends')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')
st.write("""
## Problem Statement

With environmental issues and health becoming trending topics, usage of bicycles as a mode of transportation has gained traction in recent years. To encourage bike usage, cities across the world have successfully rolled out bike sharing programs. Under such schemes, riders can rent bicycles using manual/ automated kiosks spread across the city for defined periods. In most cases, riders can pick up bikes from one location and return them to any other designated place. The bike sharing platforms from across the world are hotspots of all sorts of data, ranging from travel time, start and end location, demographics of riders, and so on. This data along with alternate sources of information such as weather, traffic, terrain, and so on makes it an attractive proposition for different research areas. The Capital Bike Sharing dataset contains information related to one such bike sharing program underway in Washington DC. Given this augmented (bike sharing details along with weather information) dataset, can we forecast bike rental demand for this program?
""")
hour_df = pd.read_csv("https://raw.githubusercontent.com/ftarantuviez/Analyzing-Bike-Sharing-Trends/master/Datasets/hour.csv")
hour_df.rename(columns={
    "instant": "rec_id",
    "dteday": "datetime",
    "holiday": "is_holiday",
    "workingday": "is_workingday",
    "weathersit": "weather_condition",
    "hum": "humidity",
    "mnth": "month",
    "cnt": "total_count",
    "hr": "hour",
    "yr": "year"
}, inplace=True)
hour_df["datetime"] = pd.to_datetime(hour_df.datetime)


st.write("## Dataset")
st.dataframe(hour_df)


st.write(""" 
## Data Visualization

In the chart below we can see the season wise hourly distribution of counts. We can notice that almost all the seasons have a peak at 18 hours, and other less bigger at 8 am.
""")
fig, ax = plt.subplots()
sn.pointplot(data=hour_df[["hour", "total_count", "season"]], x="hour", y="total_count", hue="season", ax=ax)
ax.set(title="Season wise hourly distribution of counts")
st.pyplot(fig)

st.write("In the below boxplot we can see more in detail the distribution of bikes rented by hour:")
fig, ax = plt.subplots()
sn.boxplot(data=hour_df[["hour", "total_count"]], x="hour", y="total_count", ax=ax)
ax.set(title="Box plot for hourly distribution of counts")
st.pyplot(fig)

st.write("If we analyse the total count distribution but instead of stratify it by season we do by week day, we can observe that the weekends have other distribution to the weekdays. The below chart says that.")
fig, ax = plt.subplots()
sn.pointplot(data=hour_df[["hour", "total_count", "weekday"]], x="hour", y="total_count", hue="weekday", ax=ax)
ax.set(title="Week wise hourly distribution of counts")
st.pyplot(fig)

st.write("And visualizing it by moth, the ditribution of total bikes rented have a very similar across the different months. However, from the fifth month (May) to  the tenth (October) tends to be more number of bicles which have been rented by the clients.")
fig, ax = plt.subplots()
sn.barplot(data=hour_df[["month", "total_count"]], x="month", y="total_count", ax=ax)
ax.set(title="Monthly distribution of counts")
st.pyplot(fig)

st.write("In the below violin plot is easy to note the increase of rented bicles number. In the first year (noted by '0') the distribution roughly pass the 700 bikes. Instead, in the second year ('1') this number reach the 1000 bikes.")
fig = px.violin(hour_df, x="year", y="total_count", color="year", title="Bikes rented by Year")
st.plotly_chart(fig)

st.write("Then, in the below 'Chart 1', we observe that the clients are generally registered user instead of casual ones.")
st.write("Now, if we try to understand if the amount of bikes rented is affected by the temperature and the weather (chart 2 below), we can notice that the major distribution of temperature in celsius fall between .7 and .3 approximatly. This means 'stable' temperature. Also, the windspeed generally is low.")
fig, (ax1, ax2) = plt.subplots(ncols=2)
sn.boxplot(data=hour_df[["total_count", "casual", "registered"]], ax=ax1)
ax1.set_xlabel("Chart 1")
sn.boxplot(data=hour_df[["temp", "windspeed"]], ax=ax2)
ax2.set_xlabel("Chart 2")
st.pyplot(fig)

st.write("Now let's analyse the correlation of the features:")
corrMatt = hour_df[["temp", "atemp", "humidity", "windspeed", "casual", "registered", "total_count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
sn.heatmap(corrMatt, mask=mask, vmax=.8, square=True,annot=True, ax=ax)
st.pyplot(fig)
st.write("There we see that the temperature (temp) and the feeling temperature (atemp) have a relatively strong positive correlation. Also we notice that humidity have a weak negative correlation. This means that usually people tends to rent a bike when there is less humidity. And finally (and logically) the registered users have a strong correlation with rent a bike.")

st.write(""" 
## Prediction

Trough Machine Learning Algorithms, was developed a model to predict amount of bikes that can be rented in determined variables. Filling the below fields, you are able to predict your own results!
The algorithm used is a Decision Tree.
""")

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
seasons = ["Spring", "Summer", "Fall", "Winter"]
yes_no = ["No", "Yes"]

col1, col2 = st.beta_columns(2)

temp = col1.number_input("Temperature (Celsius)", 0.0, 1.0, 0.2)
humidity = col1.number_input("Humidity", 0.0, 1.0, 0.4)
windspeed = col1.number_input("Wind Speed", 0.0, 0.7, 0.2)
weekday = col1.selectbox("Day", weekdays)
weekday_p = weekdays.index(weekday)
hour = col1.slider("Hour", 0, 23, 18)
month = col2.selectbox("Month", months)
month_p = months.index(month)
year = col2.selectbox("Year", [0,1])

season = col2.selectbox("Season", seasons)
season_dummy = [0, 0, 0, 0]
season_dummy[seasons.index(season)] = 1

holiday = col2.selectbox("Is holiday?", yes_no)
holiday_dummy = [0, 0]
holiday_dummy[yes_no.index(holiday)] = 1 

weathersit = col2.slider("Weather *", 1, 4, 1)
weather_dummy = [0, 0, 0, 0]
weather_dummy[weathersit - 1] = 1

is_workingday = st.selectbox("Is working day?", yes_no)
workingday_dummy = [0, 0]
workingday_dummy[yes_no.index(is_workingday)] = 1 

user_list = [temp, humidity, windspeed, weekday_p, hour, month_p, year, season_dummy[0], season_dummy[1], season_dummy[2], season_dummy[3], holiday_dummy[0], holiday_dummy[1], weather_dummy[0], weather_dummy[1], weather_dummy[2], weather_dummy[3], workingday_dummy[0], workingday_dummy[1]]

labels = ["Temperature", "Humidity", "Wind Speed", "Weekday", "Hour", "Month", "Year", "Season", "Holiday", "Weather", "Workingday"]
df_to_show_pred = pd.DataFrame(pd.Series([temp, humidity, windspeed, weekday, hour, month, year, season, holiday, weathersit, is_workingday])).T
df_to_show_pred.columns = labels

st.write("""
\* **Weather list** : 
  - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
  - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
  - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
""")

st.write("### Values to predict")
st.dataframe(df_to_show_pred)

if st.button("Predict"):
  with open("tree.pkl", "rb") as mod:
    st.write("### Prediction result")
    model = pickle.load(mod)
    predict = model.predict([user_list])
    st.dataframe(pd.DataFrame(pd.Series(predict), columns=["Value"]))
    st.write("This means, that given all the above variables, you should rent approximately {} bikes".format(round(predict[0])))



# This app repository

st.write("""
---
## App repository

[Github](https://github.com/ftarantuviez/Analyzing-Bike-Trendings)
""")
# / This app repository