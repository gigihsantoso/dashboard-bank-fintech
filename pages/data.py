import streamlit as st
import helper as help

st.set_page_config(
    page_title="Bank Portugis",
    page_icon="👋",
)

dt = st.radio(
    "Select Dataset",
    ('Train', 'Test'))

if dt =="Train":
    st.dataframe(help.train(),width=None, height=None)
else:
    st.dataframe(help.test(),width=None, height=None)