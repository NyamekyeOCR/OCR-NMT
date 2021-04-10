import streamlit as st
import time

"""
with st.spinner('Wait fo it'):
    time.sleep(5)
st.success("Done")
"""

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)