import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Ocean Acidification", "Natural Disasters", "Pollution", "Data Analysis"])


# Ocean Acidification Tab
with tab1:
    st.write("### Areas of Bleaching With High Ocean Acidification")
    
    @st.cache_data
    def load_data():
        coral_df = pd.read_csv('datasets/global_bleaching_environmental.csv', low_memory=False)
        return coral_df

    coral_df = load_data()


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

    # Graphs
    st.write("### Temperature vs. Percent Bleaching")
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['temperature'], filtered_df['bleaching'], alpha=0.5)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Percent Bleaching')
    plt.title('Temperature vs. Percent Bleaching')
    st.pyplot(plt)


    


    
#Natural Disasters Tab
with tab2:
    st.write("### Areas of Bleaching Suffering from Natural Disasters")

    @st.cache_data
    def load_data():
        coral_df = pd.read_csv('datasets/global_bleaching_environmental.csv', low_memory=False)
        return coral_df

    coral_df = load_data()

    # Preprocess the data
    coral_df.replace('nd', pd.NA, inplace=True)
    coral_df['Cyclone_Frequency'] = pd.to_numeric(coral_df['Cyclone_Frequency'], errors='coerce')
    coral_df['Windspeed'] = pd.to_numeric(coral_df['Windspeed'], errors='coerce')
    coral_df['Turbidity'] = pd.to_numeric(coral_df['Turbidity'], errors='coerce')
    coral_df['Percent_Bleaching'] = pd.to_numeric(coral_df['Percent_Bleaching'], errors='coerce')

    # Drop rows with missing values in critical columns
    coral_df = coral_df.dropna(subset=['Cyclone_Frequency', 'Windspeed', 'Turbidity', 'Percent_Bleaching'])

    # Rename columns for consistency
    coral_df.rename(columns={
        'Latitude_Degrees': 'latitude',
        'Longitude_Degrees': 'longitude',
        'Percent_Bleaching': 'bleaching',
        'Date_Year': 'year'
    }, inplace=True)

    # Add a slider for year selection
    min_year = int(coral_df['year'].min())
    max_year = 2030  # Extend to 2030 for predictions
    selected_year = st.slider("Select Year", min_year, max_year, value=min_year, key='nd_year')

    # Train a regression model to predict bleaching based on cyclone frequency, windspeed, and turbidity
    def train_regression_model(df):
        X = df[['Cyclone_Frequency', 'Windspeed', 'Turbidity']]
        y = df['bleaching']
        model = LinearRegression()
        model.fit(X, y)
        return model

    model = train_regression_model(coral_df)

    # Handle future years (beyond the dataset's range)
    if selected_year > coral_df['year'].max():
        st.write(f"### Predictions for {selected_year}")
        # Create synthetic data for future years
        future_data = coral_df.copy()
        future_data['year'] = selected_year

        # Simulate realistic values for future years (instead of using the mean)
        # Example: Assume a linear trend for Cyclone_Frequency, Windspeed, and Turbidity
        future_data['Cyclone_Frequency'] = coral_df['Cyclone_Frequency'] + (selected_year - coral_df['year'].max()) * 0.1  # Increase by 0.1 per year
        future_data['Windspeed'] = coral_df['Windspeed'] + (selected_year - coral_df['year'].max()) * 0.05  # Increase by 0.05 per year
        future_data['Turbidity'] = coral_df['Turbidity'] + (selected_year - coral_df['year'].max()) * 0.02  # Increase by 0.02 per year

        # Predict bleaching for future years
        future_data['predicted_bleaching'] = model.predict(future_data[['Cyclone_Frequency', 'Windspeed', 'Turbidity']])
        filtered_df = future_data
    else:
        # Filter data for the selected year
        filtered_df = coral_df[coral_df['year'] == selected_year].copy()
        filtered_df['predicted_bleaching'] = model.predict(filtered_df[['Cyclone_Frequency', 'Windspeed', 'Turbidity']])

    if filtered_df.empty:
        st.error("No data available for the selected year. Please choose a different year.")
    else:
        # Define a function to assign colors based on predicted bleaching
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

        # Create a PyDeck layer for the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            filtered_df,
            get_position=['longitude', 'latitude'],
            get_color='color',
            get_radius=50000,  # Adjust the radius as needed
            pickable=True
        )

        # Set up the PyDeck map view
        view_state = pdk.ViewState(
            latitude=filtered_df['latitude'].mean(),
            longitude=filtered_df['longitude'].mean(),
            zoom=2
        )

        # Display the map
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                'html': '<b>Latitude:</b> {latitude}<br/>'
                        '<b>Longitude:</b> {longitude}<br/>'
                        '<b>Predicted Bleaching:</b> {predicted_bleaching:.2f}%<br/>'
                        '<b>Cyclone Frequency:</b> {Cyclone_Frequency}<br/>'
                        '<b>Windspeed:</b> {Windspeed:.2f}<br/>'
                        '<b>Turbidity:</b> {Turbidity:.2f}',
                'style': {
                    'backgroundColor': 'steelblue',
                    'color': 'white'
                }
            }
        ))

        # Display summary statistics
        st.write("### Summary Statistics for Bleaching")
        st.write(filtered_df['bleaching'].describe())

        # Visualizations
        st.write("### Cyclone Frequency vs. Percent Bleaching")
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_df['Cyclone_Frequency'], filtered_df['bleaching'], alpha=0.5)
        plt.xlabel('Cyclone Frequency')
        plt.ylabel('Percent Bleaching')
        plt.title('Cyclone Frequency vs. Percent Bleaching')
        st.pyplot(plt)

        st.write("### Windspeed vs. Percent Bleaching")
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_df['Windspeed'], filtered_df['bleaching'], alpha=0.5)
        plt.xlabel('Windspeed')
        plt.ylabel('Percent Bleaching')
        plt.title('Windspeed vs. Percent Bleaching')
        st.pyplot(plt)



# Pollution Tab
with tab3:
    st.write("### Impact of Pollution on Coral Bleaching")

    @st.cache_data
    def load_data():
        coral_df = pd.read_csv('datasets/global_bleaching_environmental.csv', low_memory=False)
        return coral_df

    coral_df = load_data()

    # Preprocess the data
    coral_df.replace('nd', pd.NA, inplace=True)
    coral_df['Distance_to_Shore'] = pd.to_numeric(coral_df['Distance_to_Shore'], errors='coerce')
    coral_df['Turbidity'] = pd.to_numeric(coral_df['Turbidity'], errors='coerce')
    coral_df['Percent_Bleaching'] = pd.to_numeric(coral_df['Percent_Bleaching'], errors='coerce')

    # Drop rows with missing values in critical columns
    coral_df = coral_df.dropna(subset=['Distance_to_Shore', 'Turbidity', 'Percent_Bleaching'])

    # Rename columns for consistency
    coral_df.rename(columns={
        'Latitude_Degrees': 'latitude',
        'Longitude_Degrees': 'longitude',
        'Percent_Bleaching': 'bleaching',
        'Date_Year': 'year'
    }, inplace=True)

    # Add a slider for year selection
    min_year = int(coral_df['year'].min())
    max_year = 2030  # Extend to 2030 for predictions
    selected_year = st.slider("Select Year", min_year, max_year, value=min_year, key='ha_year')

    # Train a regression model to predict bleaching based on distance to shore and turbidity
    def train_regression_model(df):
        X = df[['Distance_to_Shore', 'Turbidity']]
        y = df['bleaching']
        model = LinearRegression()
        model.fit(X, y)
        return model

    model = train_regression_model(coral_df)

    # Handle future years (beyond the dataset's range)
    if selected_year > coral_df['year'].max():
        st.write(f"### Predictions for {selected_year}")
        # Create synthetic data for future years
        future_data = coral_df.copy()
        future_data['year'] = selected_year

        # Simulate realistic values for future years (instead of using the mean)
        # Example: Assume a linear trend for Distance_to_Shore and Turbidity
        future_data['Distance_to_Shore'] = coral_df['Distance_to_Shore'] + (selected_year - coral_df['year'].max()) * 0.1  # Increase by 0.1 km per year
        future_data['Turbidity'] = coral_df['Turbidity'] + (selected_year - coral_df['year'].max()) * 0.05  # Increase by 0.05 units per year

        # Predict bleaching for future years
        future_data['predicted_bleaching'] = model.predict(future_data[['Distance_to_Shore', 'Turbidity']])
        filtered_df = future_data
    else:
        # Filter data for the selected year
        filtered_df = coral_df[coral_df['year'] == selected_year].copy()
        filtered_df['predicted_bleaching'] = model.predict(filtered_df[['Distance_to_Shore', 'Turbidity']])

    if filtered_df.empty:
        st.error("No data available for the selected year. Please choose a different year.")
    else:
        # Define a function to assign colors based on predicted bleaching
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

        # Create a PyDeck layer for the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            filtered_df,
            get_position=['longitude', 'latitude'],
            get_color='color',
            get_radius=50000,  # Adjust the radius as needed
            pickable=True
        )

        # Set up the PyDeck map view
        view_state = pdk.ViewState(
            latitude=filtered_df['latitude'].mean(),
            longitude=filtered_df['longitude'].mean(),
            zoom=2
        )

        # Display the map
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                'html': '<b>Latitude:</b> {latitude}<br/>'
                        '<b>Longitude:</b> {longitude}<br/>'
                        '<b>Predicted Bleaching:</b> {predicted_bleaching:.2f}%<br/>'
                        '<b>Distance to Shore:</b> {Distance_to_Shore:.2f} km<br/>'
                        '<b>Turbidity:</b> {Turbidity:.2f}',
                'style': {
                    'backgroundColor': 'steelblue',
                    'color': 'white'
                }
            }
        ))

        # Display summary statistics
        st.write("### Summary Statistics for Bleaching")
        st.write(filtered_df['bleaching'].describe())

        # Visualizations
        st.write("### Distance to Shore vs. Percent Bleaching")
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_df['Distance_to_Shore'], filtered_df['bleaching'], alpha=0.5)
        plt.xlabel('Distance to Shore (km)')
        plt.ylabel('Percent Bleaching')
        plt.title('Distance to Shore vs. Percent Bleaching')
        st.pyplot(plt)

        st.write("### Turbidity vs. Percent Bleaching")
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_df['Turbidity'], filtered_df['bleaching'], alpha=0.5)
        plt.xlabel('Turbidity')
        plt.ylabel('Percent Bleaching')
        plt.title('Turbidity vs. Percent Bleaching')
        st.pyplot(plt)


# Data Analysis Tab
with tab4:
    st.write("### Data Analysis")
    subtab1, subtab2, subtab3 = st.tabs(["Ocean Acidification", "Natural Disasters", "Pollution"])

    # Load and preprocess the data once
    @st.cache_data
    def load_and_preprocess_data():
        coral_df = pd.read_csv('datasets/global_bleaching_environmental.csv', low_memory=False)
        
        # Replace 'nd' with NaN
        coral_df.replace('nd', pd.NA, inplace=True)
        
        # Convert columns to numeric
        numeric_columns = ['Temperature_Mean', 'Percent_Bleaching', 'Cyclone_Frequency', 'Windspeed', 'Turbidity', 'Distance_to_Shore']
        for col in numeric_columns:
            coral_df[col] = pd.to_numeric(coral_df[col], errors='coerce')
        
        # Drop rows with missing values in critical columns
        coral_df.dropna(subset=numeric_columns, inplace=True)
        
        # Rename columns for consistency
        coral_df.rename(columns={
            'Latitude_Degrees': 'latitude',
            'Longitude_Degrees': 'longitude',
            'Percent_Bleaching': 'bleaching',
            'Temperature_Mean': 'temperature',
            'Date_Year': 'year'
        }, inplace=True)
        
        return coral_df

    coral_df = load_and_preprocess_data()



    # Ocean Acidification Subtab
    with subtab1:
        st.write("#### Ocean Acidification Analysis")
        
        # Define feature and target columns
        feature_columns = ['temperature', 'year']  # Ensure this matches the dataset
        target_column = 'bleaching'  # Ensure this matches the dataset
        
        # Check if required columns exist in the dataset
        if all(col in coral_df.columns for col in feature_columns + [target_column]):
            # Drop rows with missing values in the selected columns
            coral_df_clean = coral_df.dropna(subset=feature_columns + [target_column])
            
            if len(coral_df_clean) > 0:
                X = coral_df_clean[feature_columns]
                y = coral_df_clean[target_column]
                
                if X.shape[0] > 0 and y.shape[0] > 0:
                    # Train a Random Forest Regressor
                    model = RandomForestRegressor(random_state=42)
                    model.fit(X, y)
                    
                    # Predict and evaluate
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    
                    st.write("##### Model Accuracy")
                    st.write(f"Mean Squared Error (MSE): {mse:.3f}")
                    st.write(f"R-squared (R²): {r2:.3f}")
                    
                    # Plot predicted vs. actual values
                    st.write("##### Predicted vs. Actual Values")
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y, y_pred, alpha=0.5)
                    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Diagonal line for reference
                    plt.xlabel('Actual Bleaching (%)')
                    plt.ylabel('Predicted Bleaching (%)')
                    plt.title('Predicted vs. Actual Bleaching: Random Forest')
                    st.pyplot(plt)

                    variance_bleaching = coral_df_clean[target_column].var()
                    st.write(f"Variance of Bleaching: {variance_bleaching:.3f}")
                    st.write(f"MSE: {mse:.3f}")

                else:
                    st.error("No data available for model training. Check the dataset.")
            else:
                st.error("No data available after preprocessing. Check for missing values.")
        else:
            st.error("Required columns are missing in the dataset. Check the column names.")




    # Natural Disasters Subtab
    with subtab2:
        st.write("#### Natural Disasters Analysis")
        feature_columns = ['Cyclone_Frequency', 'Windspeed']
        target_column = 'bleaching'

        if all(col in coral_df.columns for col in feature_columns + [target_column]):
            # Drop rows with missing values in the selected columns
            coral_df_clean = coral_df.dropna(subset=feature_columns + [target_column])

            X = coral_df_clean[feature_columns]
            y = coral_df_clean[target_column]

            # Train a Random Forest Regressor
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)

            # Predict and evaluate
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred) 
            r2 = r2_score(y, y_pred) 

            st.write("##### Model Accuracy")
            st.write(f"Mean Squared Error (MSE): {mse:.3f}")
            st.write(f"R-squared (R²): {r2:.3f}")


            # Plot predicted vs. actual values
            st.write("##### Predicted vs. Actual Values")
            plt.figure(figsize=(10, 6))
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Diagonal line for reference
            plt.xlabel('Actual Bleaching (%)')
            plt.ylabel('Predicted Bleaching (%)')
            plt.title('Predicted vs. Actual Bleaching: Random Forest')
            st.pyplot(plt)

            variance_bleaching = coral_df_clean[target_column].var()
            st.write(f"Variance of Bleaching: {variance_bleaching:.3f}")
            st.write(f"MSE: {mse:.3f}")

    # Pollution Subtab
    with subtab3:
        st.write("#### Pollution Analysis")
        feature_columns = ['Distance_to_Shore', 'Turbidity']
        target_column = 'bleaching'

        if all(col in coral_df.columns for col in feature_columns + [target_column]):
            # Drop rows with missing values in the selected columns
            coral_df_clean = coral_df.dropna(subset=feature_columns + [target_column])

            X = coral_df_clean[feature_columns]
            y = coral_df_clean[target_column]

            # Train a Random Forest Regressor
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)

            # Predict and evaluate
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred) 

            st.write("##### Model Accuracy")
            st.write(f"Mean Squared Error (MSE): {mse:.3f}")
            st.write(f"R-squared (R²): {r2:.3f}")

            

            # Plot predicted vs. actual values
            st.write("##### Predicted vs. Actual Values")
            plt.figure(figsize=(10, 6))
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Diagonal line for reference
            plt.xlabel('Actual Bleaching (%)')
            plt.ylabel('Predicted Bleaching (%)')
            plt.title('Predicted vs. Actual Bleaching: Random Forest')
            st.pyplot(plt)

            variance_bleaching = coral_df_clean[target_column].var()
            st.write(f"Variance of Bleaching: {variance_bleaching:.3f}")
            st.write(f"MSE: {mse:.3f}")

# # Printing Mean values for comparison

# # Load the data
# coral_df = pd.read_csv('datasets/global_bleaching_environmental.csv', low_memory=False)

# # Preprocess the data
# coral_df.replace('nd', pd.NA, inplace=True)
# coral_df['Temperature'] = pd.to_numeric(coral_df['Temperature_Mean'], errors='coerce')
# coral_df['Distance_to_Shore'] = pd.to_numeric(coral_df['Distance_to_Shore'], errors='coerce')
# coral_df['Turbidity'] = pd.to_numeric(coral_df['Turbidity'], errors='coerce')
# coral_df['Cyclone_Frequency'] = pd.to_numeric(coral_df['Cyclone_Frequency'], errors='coerce')
# coral_df['Windspeed'] = pd.to_numeric(coral_df['Windspeed'], errors='coerce')

# # Drop rows with missing values in the columns of interest
# coral_df = coral_df.dropna(subset=['Temperature', 'Distance_to_Shore', 'Turbidity', 'Cyclone_Frequency', 'Windspeed'])

# # Print mean values
# print("Mean Temperature:", round(coral_df['Temperature'].mean(), 3))
# print("Mean Distance to Shore:", round(coral_df['Distance_to_Shore'].mean(), 3))
# print("Mean Turbidity:", round(coral_df['Turbidity'].mean(), 3))
# print("Mean Cyclone Frequency:", round(coral_df['Cyclone_Frequency'].mean(), 3))
# print("Mean Windspeed:", round(coral_df['Windspeed'].mean(), 3))


