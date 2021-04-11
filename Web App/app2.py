import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def app():

    st.write("# Find My City")

    df = pd.read_csv('https://tamu-datathon-2020.s3.us-east-2.amazonaws.com/data/country.csv')
    df = df.drop('Unnamed: 0', axis = 1)

    # Helper Function

    def zoom_center(lons: tuple=None, lats: tuple=None, lonlats: tuple=None, format: str='lonlat', projection: str='mercator',
        width_to_height: float=2.0) -> (float, dict):
        
        if lons is None and lats is None:
            if isinstance(lonlats, tuple):
                lons, lats = zip(*lonlats)
            else:
                raise ValueError(
                    'Must pass lons & lats or lonlats'
                )
        
        maxlon, minlon = max(lons), min(lons)
        maxlat, minlat = max(lats), min(lats)
        center = {
            'lon': round((maxlon + minlon) / 2, 6),
            'lat': round((maxlat + minlat) / 2, 6)
        }
        
        # longitudinal range by zoom level (20 to 1)
        # in degrees, if centered at equator
        lon_zoom_range = np.array([
            0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
            0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
            47.5136, 98.304, 190.0544, 360.0
        ])
        
        if projection == 'mercator':
            margin = 1.2
            height = (maxlat - minlat) * margin * width_to_height
            width = (maxlon - minlon) * margin
            lon_zoom = np.interp(width , lon_zoom_range, range(20, 0, -1))
            lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
            zoom = round(min(lon_zoom, lat_zoom), 2)
        else:
            raise NotImplementedError(
                f'{projection} projection is not implemented'
            )
        
        return zoom, center


    # actual plot

    #@title Rate importance of each of the following factors

    def graph_city(cities = [], crime = 'Med', health = 'Med', pol = 'Med', qol = 'Med', pp = 'Med', dem = 'Med', hap = 'Med', rep = 'Med', pop = 'Med', price = 'Med', var = False):

        a = [crime, health, pol, qol, pp, dem, hap, rep, pop, price]
        
        for i in range(len(a)):
            if a[i] <= 2.5:
                a[i] = 'None'
            elif a[i] <= 5:
                a[i] = 'Low'
            elif a[i] <= 7.5:
                a[i] = 'Med'
            elif a[i] <= 10:
                a[i] = 'High'

        df['Price Per Square Foot (USD)'] = [i.replace(',', '') for i in df['Price Per Square Foot (USD)'].astype('str')]
        df['Price Per Square Foot (USD)'] = df['Price Per Square Foot (USD)'].astype('float')

        weights = a
        replace = {'None': 0, 'Low': 1, 'Med': 2, 'High': 3}
        weights = np.array([replace[x] for x in weights])
        weights *= [-1, 1, -1, 1, 1, 1, 1, 1, -1, -1]

        features = ['Crime Rating', 'Health Care', 'Pollution', 'Quality of Life', 'Purchase Power', 'Democracy', 'Happiness',
        'Movehub Rating', 'Population', 'Price Per Square Foot (USD)']
        norm = lambda xs: (xs-xs.min())/(xs.max()-xs.min())

        df['Desirability Index'] = norm(df[features].dot(weights))*10


        # df_subset = df[df.Country.infer_objects().isin(countries)]
        if len(cities) == 0:
            df_subset = df
        else:
            df_subset = df[df.City.infer_objects().isin(cities)]

        zoom, center = zoom_center(lons = list(df_subset.lng), lats = list(df_subset.lat))

        if len(cities) == 1:
            zoom -= 10

        fig = px.scatter_mapbox(df_subset.sort_values('Desirability Index', ascending=False),
                                lat= 'lat', lon='lng', color="Desirability Index", hover_name="City",
                                hover_data=features, opacity = 0.6, size = 'Population',
                                color_continuous_scale=px.colors.sequential.Oranges, zoom = zoom, center = center,
                                mapbox_style="carto-darkmatter")


        # fig.update_geos(fitbounds="locations")
        fig.update_layout(height = 1300, margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_showscale = var)

        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            )
        )

        return fig

        # df.sort_values('Score', ascending=False)[['City', 'Score'] + features].round()

    countries = np.sort(df['Country'].unique())

    file_name= 'kmeans.pkl'
    f=open(file_name, 'rb')
    km = pickle.load(f)
    df['Labels'] =km.labels_

    # Input Preprocessing 

    def input_preprocessing(crime, health, pol, qol, pp, dem, hap, rep, pop, price):
        
        df['Price Per Square Foot (USD)'] = [i.replace(',', '') for i in df['Price Per Square Foot (USD)'].astype('str')]
        df['Price Per Square Foot (USD)'] = df['Price Per Square Foot (USD)'].astype('float')

        crime_q=df['Crime Rating'].quantile(int(crime)/10)
        health_q=df['Health Care'].quantile(int(health)/10)
        pollution_q=df['Pollution'].quantile(int(pol)/10)
        quality_q=df['Quality of Life'].quantile(int(qol)/10)
        purchase_q=df['Purchase Power'].quantile(int(pp)/10)
        democracy_q=df['Democracy'].quantile(int(dem)/10)
        happiness_q=df['Happiness'].quantile(int(hap)/10)
        overall_q=df['Movehub Rating'].quantile(int(rep)/10)
        population_q=df['Population'].quantile(int(pop)/10)
        price_q=df['Price Per Square Foot (USD)'].quantile(int(price)/10)

        # Generate Numpy Array

        inp=np.array([overall_q,purchase_q,health_q,pollution_q,quality_q,crime_q,democracy_q,
                    happiness_q, population_q, price_q]).reshape(1,-1)

        train_X=df.drop(columns= ['City', 'Country','lat', 'lng','Country_Code','paired', 'Labels'])            
        scaler = StandardScaler()
        train_X=scaler.fit_transform(train_X)
        inp=scaler.transform(inp)

        return inp


    # Model Deployment # Assuming inputs is standardised and in a numpy array

    def assign_cities(inputs, countries=[]):
        results = km.transform(inputs)
        i = 0 # The number of cities selected 
        cities_from_cluster = [] # Create a selected list
        city=[]
        for c in countries:
            city.append(int(df['Country'].value_counts().loc[df['Country'].value_counts().index==c]))
        if sum(city) < 4:
            while i < 4: # While the number of cities selected is less than 4 keep selecting
                index = np.where(results == np.amin(results))
                results[index]=100000
                filtered = df[df['Labels']==index[1][0]]
                if len(filtered) < 4-i:
                    city_sample = random.sample(list(filtered.City),len(filtered))
                else: 
                    city_sample=random.sample(list(filtered.City),4-i)
                cities_from_cluster += city_sample
                i += len(city_sample)
        else:
            while i < 4: # While the number of cities selected is less than 4 keep selecting
                index = np.where(results == np.amin(results))
                results[index] = 100000 # Update the value so that we won't keep selecting the same cluster
                filtered = df[df['Labels']==index[1][0]]
                for country in countries:
                    cities_pop=list(filtered.City[filtered['Country'] == country])
                    if len(cities_pop) < 4-i:
                        city_sample=random.sample(cities_pop,len(cities_pop))
                    else:
                        city_sample = random.sample(cities_pop,4-i)
                cities_from_cluster += city_sample
                i += len(city_sample)
        return cities_from_cluster

    def algorithm(crime, health, pol, qol, pp, dem, hap, rep, pop, price, selection):
        return assign_cities(input_preprocessing(crime, health, pol, qol, pp, dem, hap, rep, pop, price), selection)

    selection = st.multiselect('Select Countries', countries)

    st.write("Please indicate on a scale of 1-10 your preference for each of the following factors.")

    crime = st.slider('Crime Rate', 0, 10)
    health = st.slider('Health Care', 0, 10)
    pol = st.slider('Pollution', 0, 10)
    qol = st.slider('Quality of Life', 0, 10)
    pp = st.slider('Purchasing Power', 0, 10)
    dem = st.slider('Democracy', 0, 10)
    hap = st.slider('Happiness', 0, 10)
    rep = st.slider('Overall Reputation', 0, 10)
    pop = st.slider('Population Size', 0, 10)
    price = st.slider('Housing Cost', 0, 10)

    if st.button('Submit'):

        st.balloons()
        
        results = algorithm(crime, health, pol, qol, pp, dem, hap, rep, pop, price, selection)

        col1, col2 = st.beta_columns(2)

        with col1:
            st.header(results[0])
            st.image("https://tamu-datathon-2020.s3.us-east-2.amazonaws.com/data/images/" + results[0] + ".jpg", use_column_width = True, output_format = 'JPEG')
            st.write(graph_city([results[0]], crime, health, pol, qol, pp, dem, hap, rep, pop, price, False), use_column_width = True)

        with col2:
            st.header(results[1])
            st.image("https://tamu-datathon-2020.s3.us-east-2.amazonaws.com/data/images/" + results[1] + ".jpg", use_column_width = True, output_format = 'JPEG')
            st.write(graph_city([results[1]], crime, health, pol, qol, pp, dem, hap, rep, pop, price, True), use_column_width = True)

        col3, col4 = st.beta_columns(2)

        with col3:
            st.header(results[2])
            st.image("https://tamu-datathon-2020.s3.us-east-2.amazonaws.com/data/images/" + results[2] + ".jpg", use_column_width = True, output_format = 'JPEG')
            st.write(graph_city([results[2]], crime, health, pol, qol, pp, dem, hap, rep, pop, price, False), use_column_width = True)

        with col4:
            st.header(results[3])
            st.image("https://tamu-datathon-2020.s3.us-east-2.amazonaws.com/data/images/" + results[3] + ".jpg", use_column_width = True, output_format = 'JPEG')
            st.write(graph_city([results[3]], crime, health, pol, qol, pp, dem, hap, rep, pop, price, True), use_column_width = True)
