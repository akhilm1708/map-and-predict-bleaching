# import streamlit as sthttps://www.raisehand.com/desktop_pop_up_window.html#
# import pandas as pd
# import pydeck as pdk
# from sklearn.linear_model import LinearRegression


# coral_df = pd.read_csv('datasets/global_bleaching_environmental.csv', low_memory=False)


# coral_df.rename(columns={'Latitude_Degrees': 'latitude', 'Longitude_Degrees': 'longitude'}, inplace=True)
# coral_df['Percent_Bleaching'] = pd.to_numeric(coral_df['Percent_Bleaching'], errors='coerce')
# coral_df = coral_df.dropna(subset=['Percent_Bleaching'])

# min_year = int(coral_df['Year'].min())
# max_year = int(coral_df['Year'].max())
# my_years = st.slider("Select Year", min_year, max_year+10, value=min_year)

# #Regression model for ph + temp to detect OA:
# def train_regression_model(df, selected_column):
#     df_filtered = df.dropna(subset = ['Year', selected_column])
#     X = df_filtered[['Year']]
#     y = df_filtered[[selected_column]]
#     model = LinearRegression
#     model.fit(X, y)
#     return model

# temp_model = train_regression_model(coral_df, 'Temperature')
# pH_model = train_regression_model(coral_df, 'pH')

# def predict_conditions(my_row, year):
#     predicted_temp = temp_model.predict([[year]])[0] #gives integer val instead of dataframe
#     predicted_pH = pH_model.predict([[year]])[0]
#     return predicted_temp, predicted_pH

# TEMP_LIMIT = 28 # Above 28°C --> bleaching risk
# PH_LIMIT = 7.8 # Below 7.8 --> bleaching risk





# def get_color(row):
#     bleaching_percent = row['Percent_Bleaching']
#     if bleaching_percent < 10:
#         return [0, 255, 0]  # green (low impact)
#     elif bleaching_percent < 30:
#         return [255, 255, 0]  # yellow (moderate impact)
#     elif bleaching_percent < 60:
#         return [255, 165, 0]  # orange (high impact)
#     else:
#         return [255, 0, 0]  # red (severe impact)

# coral_df['color'] = coral_df['Percent_Bleaching'].apply(get_color)
# # print(coral_df['color'])

# #using pydeck instead of basic streamlit map for color optimization
# layer = pdk.Layer(
#     'ScatterplotLayer',
#     coral_df,
#     get_position=['longitude', 'latitude'],
#     get_color='color',
#     get_radius=40000,
#     pickable=True
# )

# # Set up the PyDeck map
# view_state = pdk.ViewState(
#     latitude=coral_df['latitude'].mean(),
#     longitude=coral_df['longitude'].mean(),
#     zoom=2
# )

# # Display the Streamlit title and map
# st.title("Mapping the Effect of Ocean Acidification on Global Coral Bleaching Over Time")
# st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

