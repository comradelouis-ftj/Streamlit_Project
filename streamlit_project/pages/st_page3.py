import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='NLP Project - Preprocessing', 
    layout='wide'
)

st.markdown(
"""
<style>
div[data-testid="stSidebarNav"] {display: none;}
.stButton > button {
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", 
unsafe_allow_html=True)

with st.sidebar:
    st.title('NLP Dengan Neural Network')
    st.divider()

    if st.button('**Profil Proyek**', width="stretch"):
        st.switch_page('st_page1.py')
    
    if st.button('**EDA/Insight**', width="stretch"):
        st.switch_page('pages/st_page2.py')
    
    if st.button('**Tahap Preprocessing**', width="stretch"):
        st.switch_page('pages/st_page3.py')

    if st.button('**Try the Model!**', width="stretch"):
        st.switch_page('pages/st_page4.py')


st.title('Tahapan Preprocessing')