import streamlit as st
import pandas as pd

#loading dataset
coral_df = pd.read_csv('datasets/global_bleaching_environmental.csv', low_memory=False)

#filtering data
coral_df.rename(columns={'Latitude_Degrees': 'latitude', 'Longitude_Degrees': 'longitude'}, inplace=True)
map_data = coral_df[['latitude', 'longitude']]


st.title("Effect of Ocean Acidification on Global Coral Bleaching")
st.map(map_data)