import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
df = st.cache(pd.read_csv)("new_data.csv")
is_check = st.checkbox("Display Data")
if is_check:
    st.write(df)