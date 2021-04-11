import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def app():

    st.write("# Explore the World")

    df = pd.read_csv('https://tamu-datathon-2020.s3.us-east-2.amazonaws.com/data/country.csv')

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

    def graph_city(cities = [], crime = 'Med', health = 'Med', pol = 'Med', qol = 'Med', pp = 'Med', dem = 'Med', hap = 'Med', rep = 'Med', pop = 'Med', price = 'Med'):

        df['Price Per Square Foot (USD)'] = [i.replace(',', '') for i in df['Price Per Square Foot (USD)'].astype('str')]
        df['Price Per Square Foot (USD)'] = df['Price Per Square Foot (USD)'].astype('float')

        weights = [
        crime, 
        health, 
        pol, 
        qol, 
        pp, 
        dem, 
        hap, 
        rep, 
        pop, 
        price, 
        ]
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
        fig.update_layout(height = 1300, width = 800, margin={"r":0,"t":0,"l":0,"b":0})

        return fig

        # df.sort_values('Score', ascending=False)[['City', 'Score'] + features].round()

    def graph_country(countries = [], crime = 'Med', health = 'Med', pol = 'Med', qol = 'Med', pp = 'Med', dem = 'Med', hap = 'Med', rep = 'Med', pop = 'Med', price = 'Med'):
        
        df['Price Per Square Foot (USD)'] = [i.replace(',', '') for i in df['Price Per Square Foot (USD)'].astype('str')]
        df['Price Per Square Foot (USD)'] = df['Price Per Square Foot (USD)'].astype('float')

        weights = [
        crime, 
        health, 
        pol, 
        qol, 
        pp, 
        dem, 
        hap, 
        rep, 
        pop, 
        price, 
        ]
        replace = {'None': 0, 'Low': 1, 'Med': 2, 'High': 3}
        weights = np.array([replace[x] for x in weights])
        weights *= [-1, 1, -1, 1, 1, 1, 1, 1, -1, -1]

        features = ['Crime Rating', 'Health Care', 'Pollution', 'Quality of Life', 'Purchase Power', 'Democracy', 'Happiness',
        'Movehub Rating', 'Population', 'Price Per Square Foot (USD)']
        norm = lambda xs: (xs-xs.min())/(xs.max()-xs.min())

        df['Desirability Index'] = norm(df[features].dot(weights))*10

        if len(countries) == 0:
            df_subset = df
        else:
            df_subset = df[df.Country.infer_objects().isin(countries)]
        #df_subset = df[df.City.infer_objects().isin(cities)]

        zoom, center = zoom_center(lons = list(df_subset.lng), lats = list(df_subset.lat))

        if len(countries) == 1:
            zoom -= 10

        fig = px.scatter_mapbox(df_subset.sort_values('Desirability Index', ascending=False),
                                lat= 'lat', lon='lng', color="Desirability Index", hover_name="City",
                                hover_data=features, opacity = 0.6, size = 'Population',
                                color_continuous_scale=px.colors.sequential.Oranges, zoom = zoom, center = center,
                                mapbox_style="carto-darkmatter")


        # fig.update_geos(fitbounds="locations")
        fig.update_layout(height = 1300, width = 800, margin={"r":0,"t":0,"l":0,"b":0})

        return fig

        # df.sort_values('Score', ascending=False)[['City', 'Score'] + features].round()

    countries = np.sort(df['Country'].unique())
    cities = np.sort(df['City'].unique())

    st.write("Please indicate your preference for each of the following factors.")

    crime = st.selectbox('Crime Rate', ('None', 'Low', 'Med', 'High'))
    health = st.selectbox('Health Care', ('None', 'Low', 'Med', 'High'))
    pol = st.selectbox('Pollution', ('None', 'Low', 'Med', 'High'))
    qol = st.selectbox('Quality of Life', ('None', 'Low', 'Med', 'High'))
    pp = st.selectbox('Purchasing Power', ('None', 'Low', 'Med', 'High'))
    dem = st.selectbox('Democracy', ('None', 'Low', 'Med', 'High'))
    hap = st.selectbox('Happiness', ('None', 'Low', 'Med', 'High'))
    rep = st.selectbox('Overall Reputation', ('None', 'Low', 'Med', 'High'))
    pop = st.selectbox('Population Size', ('None', 'Low', 'Med', 'High'))
    price = st.selectbox('Housing Cost', ('None', 'Low', 'Med', 'High'))

    selection = st.multiselect('Select a Country/Countries', countries)

    if st.button('Submit', key = '1'):
        st.write(graph_country(selection, crime, health, pol, qol, pp, dem, hap, rep, pop, price), use_column_width = True)

    selection2 = st.multiselect('Select a City/Cities', cities)

    if st.button('Submit', key = '2'):
        st.write(graph_city(selection2, crime, health, pol, qol, pp, dem, hap, rep, pop, price), use_column_width = True)
