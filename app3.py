import streamlit as st

def app():

    st.write("# About Us")

    col1, col2 = st.beta_columns(2)

    with col1:
        st.header("Andrew Liu")
        st.image("https://raw.githubusercontent.com/shailm99/TAMU-Datathon-2020/main/andrew.png", use_column_width = True, output_format = 'JPEG', caption ='''
        Andrew is an undergraduate student at the University of California, Los Angeles (UCLA) majoring in Data Theory ('23).''')

    with col2:
        st.header("Shail Mirpuri")
        st.image("https://raw.githubusercontent.com/shailm99/TAMU-Datathon-2020/main/shail.jpg", use_column_width = True, output_format = 'JPEG', caption ='''
        Shail is an undergraduate student at the University of California, Los Angeles (UCLA) majoring in Data Theory ('23).''')

    col3, col4 = st.beta_columns(2)

    with col3:
        st.header("Anurag Pamuru")
        st.image("https://raw.githubusercontent.com/shailm99/TAMU-Datathon-2020/main/anurag.jpg", use_column_width = True, output_format = 'JPEG', caption ='''
        Anurag is an undergraduate student at the University of California, San Diego (UCSD) majoring in Data Science ('22).''')

    with col4:
        st.header("Adhvaith Vijay")
        st.image("https://raw.githubusercontent.com/shailm99/TAMU-Datathon-2020/main/adhvaith.jpg", use_column_width = True, output_format = 'JPEG', caption ='''
        Adhvaith is an undergraduate student at the University of California, Los Angeles (UCLA) majoring in Data Theory ('22).''')