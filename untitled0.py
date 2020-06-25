all_columns_names = df.columns.tolist()
type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

# Plot By Streamlit
if type_of_plot == 'area':
    cust_data = df[selected_columns_names]
    st.area_chart(cust_data)

elif type_of_plot == 'bar':
    cust_data = df[selected_columns_names]
    st.bar_chart(cust_data)

elif type_of_plot == 'line':
    cust_data = df[selected_columns_names]
    st.line_chart(cust_data)

# Custom Plot 
elif type_of_plot:
    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
    st.write(cust_plot)
    st.pyplot()
