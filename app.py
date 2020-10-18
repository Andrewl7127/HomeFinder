import app1
import app2
import app3
import app4
import streamlit as st

PAGES = {
    "Explore the World": app1,
    "Find My City": app2,
    "About Us": app3,
    "The Task": app4
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()