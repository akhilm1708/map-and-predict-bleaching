import streamlit as st
import pandas as pd
import pydeck as pdk
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data():
    coral_df = pd.read_csv('datasets/global_bleaching_environmental.csv', low_memory=False)
    return coral_df

coral_df = load_data()

st.write("### Coral Bleaching Dataset")
st.write(coral_df.head())

coral_df.replace('nd', pd.NA, inplace=True)

coral_df['Temperature_Mean'] = pd.to_numeric(coral_df['Temperature_Mean'], errors='coerce')
coral_df['Percent_Bleaching'] = pd.to_numeric(coral_df['Percent_Bleaching'], errors='coerce')

coral_df = coral_df.dropna(subset=['Temperature_Mean', 'Percent_Bleaching'])

coral_df.rename(columns={
    'Latitude_Degrees': 'latitude',
    'Longitude_Degrees': 'longitude',
    'Percent_Bleaching': 'bleaching',
    'Temperature_Mean': 'temperature',
    'Date_Year': 'year'
}, inplace=True)

min_year = int(coral_df['year'].min())
max_year = 2030  # Extend to 2030 for predictions
selected_year = st.slider("Select Year", min_year, max_year, value=min_year)

def train_regression_model(df):
    X = df[['temperature', 'year']]  # Use 'year' as a feature for future predictions
    y = df['bleaching']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_regression_model(coral_df)

if selected_year > coral_df['year'].max():
    st.write(f"### Predictions for {selected_year}")
    future_data = coral_df.copy()
    future_data['year'] = selected_year
    future_data['predicted_bleaching'] = model.predict(future_data[['temperature', 'year']])
    filtered_df = future_data
else:
    filtered_df = coral_df[coral_df['year'] == selected_year].copy()
    filtered_df['predicted_bleaching'] = model.predict(filtered_df[['temperature', 'year']])

if filtered_df.empty:
    st.error("No data available for the selected year. Please choose a different year.")
else:
    def get_color(bleaching):
        if bleaching < 10:
            return [255, 165, 0]  # Orange (low impact)
        elif 10 <= bleaching < 30:
            return [255, 255, 0]  # Yellow (moderate impact)
        elif 30 <= bleaching < 60:
            return [0, 255, 0]  # Green (high impact)
        else:
            return [255, 0, 0]  # Red (severe impact)

    filtered_df['color'] = filtered_df['predicted_bleaching'].apply(get_color)

    layer = pdk.Layer(
        'ScatterplotLayer',
        filtered_df,
        get_position=['longitude', 'latitude'],
        get_color='color',
        get_radius=50000,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=filtered_df['latitude'].mean(),
        longitude=filtered_df['longitude'].mean(),
        zoom=2
    )

    st.write("### The Effect of Ocean Acidification on Coral Bleaching")
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            'html': '<b>Latitude:</b> {latitude}<br/>'
                    '<b>Longitude:</b> {longitude}<br/>'
                    '<b>Predicted Bleaching:</b> {predicted_bleaching:.2f}%<br/>'
                    '<b>Temperature:</b> {temperature:.2f}°C<br/>'
                    '<b>Year:</b> {year}',
            'style': {
                'backgroundColor': 'steelblue',
                'color': 'white'
            }
        }
    ))

st.write("### Summary Statistics for Bleaching")
st.write(filtered_df['bleaching'].describe())