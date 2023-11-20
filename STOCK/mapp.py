import streamlit as st
from stock_analysis import load_stock_data, plot_stock_data, plot_stock_data_with_100MA, plot_stock_data_with_100MA_200MA, prepare_predictions, plot_predictions, get_latest_price
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Title for the Streamlit app
st.title('STOCKx Stock Analysis App')

# # Initialize session_state to store selected stock
# if 'selected_stock' not in st.session_state:
#     st.session_state.selected_stock = None

# Create a sidebar for navigation
menu = st.sidebar.selectbox('Menu', ['Welcome', 'Data Page', 'Visualization Page', 'Prediction Page', 'App Benefits and Conclusion','Team Members'])

if menu == 'Welcome':
    st.subheader('How the Stock Prediction Model Works')
    st.write("""
        This stock analysis app uses a machine learning model to make predictions based on historical stock data. 
        The stock prediction model works by:

        1. Loading historical stock data for a selected company, including price and volume information.

        2. Analyzing and visualizing this data to identify patterns, trends, and moving averages in the stock's performance over time.

        3. Utilizing a pre-trained machine learning model, such as a Neural Network, to make predictions about future stock prices based on historical patterns and data.

        4. Comparing the model's predictions with actual stock prices to evaluate its accuracy and performance.

        This model is designed to provide insights into possible future stock price trends, but it's important to note that stock markets are influenced by various factors, and predictions are not guaranteed.

        Enjoy using the Stock Analysis App!
    """)

elif menu == 'Data Page':
    # Data Page
            
    st.set_option('deprecation.showPyplotGlobalUse', False)
    all_stock = pd.read_csv(r"stock.csv")
    stock_name = all_stock["Company Name"].tolist()

    st.subheader("Data Page")

    # User input for stock ticker
    st.subheader("Select a stock")
    user_stock = st.selectbox("Stock names", stock_name)

    # Load stock data based on user input
    user_input = all_stock[all_stock["Company Name"] == user_stock]["Symbol"].values[0]
    df = load_stock_data(user_input)

    # Display the latest stock price 

    # Display stock data description
    st.subheader('Data from 2013 to 2023')
    st.write(df.describe())

elif menu == 'Visualization Page':
    # Visualization Page
    st.subheader("Visualization Page")
            
    st.set_option('deprecation.showPyplotGlobalUse', False)
    all_stock = pd.read_csv(r"stock.csv")
    stock_name = all_stock["Company Name"].tolist()
    st.subheader("Select a stock")
    user_stock = st.selectbox("Stock names", stock_name)

    # Load stock data based on user input
    user_input = all_stock[all_stock["Company Name"] == user_stock]["Symbol"].values[0]
    df = load_stock_data(user_input)
    # Plot Closing Price vs Time chart
    st.subheader('Closing Price vs Time chart')
    st.write("The Closing Price vs Time chart indicates how a stock's closing price has changed over a period of time.")
    plot_stock_data(df)
    st.pyplot()  # Use st.pyplot() to display the Matplotlib plot

    # Plot Closing Price vs Time chart with 100MA
    st.subheader('Closing Price vs Time chart with 100MA')
    st.write("Closing Price vs Time chart with 100MA indicates how a stock's closing price has changed over time, and it includes a 100-day Moving Average (100MA).")
    plot_stock_data_with_100MA(df)
    st.pyplot()  # Use st.pyplot() to display the Matplotlib plot

    # Plot Closing Price vs Time chart with 100MA & 200MA
    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    st.write("Red-100MA")
    st.write("Green-200MA")
    st.write("Closing Price vs Time chart with 100MA & 200MA provides a comprehensive view of a stock's historical price movements by including two moving averages.")
    st.write("When the closing price crosses above both the 100MA and 200MA, it may be a strong bullish signal, suggesting potential price strength. Conversely, if it crosses below both moving averages, it may indicate a strong bearish signal, suggesting potential price weakness.")
    plot_stock_data_with_100MA_200MA(df)
    st.pyplot()  # Use st.pyplot() to display the Matplotlib plot

elif menu == 'Prediction Page':
    # Prediction Page
    st.subheader("Prediction Page")
            
    st.set_option('deprecation.showPyplotGlobalUse', False)
    all_stock = pd.read_csv(r"stock.csv")
    stock_name = all_stock["Company Name"].tolist()
    st.subheader("Select a stock")
    user_stock = st.selectbox("Stock names", stock_name)

    # Load stock data based on user input
    user_input = all_stock[all_stock["Company Name"] == user_stock]["Symbol"].values[0]
    df = load_stock_data(user_input)
    # Load your model and scaler here
    model = tf.keras.models.load_model('keras_model.h5')
    scaler = MinMaxScaler(feature_range=(0,1))

# Prepare predictions and get y_test and y_predicted
    y_test, y_predicted = prepare_predictions(model, scaler, df)

# Plot predictions
    st.subheader('Predictions vs Original')
    plot_predictions(y_test, y_predicted)
    st.write("The Predictions vs Original chart is used to compare the predicted stock prices (generated by a machine learning model) with the actual or original stock prices.")
    st.pyplot()  # Use st.pyplot() to display the Matplotlib plot
    st.write("By comparing the two lines, you can identify whether the model captures the general trends and patterns in the stock's price movements. If the predicted line follows the same upward or downward direction as the actual line, it suggests that the model is capable of identifying trends.")

elif menu == 'App Benefits and Conclusion':
    # About Page
    st.subheader("App Benefits and Conclusion")
    st.write("In conclusion, our Stock Analysis Hub app equips users with the essential tools and information to navigate the complex world of stock investments. By allowing users to explore historical stock data, the app provides a wealth of knowledge on how specific companies have fared over the years. It offers insights into trends, price fluctuations, and other key historical metrics, enabling users to make more informed decisions.Users can visualize the stock data through various charts, including the Closing Price vs Time chart, which provides a visual representation of the stock's price movements over time. This helps users quickly identify patterns.")

elif menu == 'Team Members':
     st.subheader("Team Members:")
     st.write("Dwijavanthi J-(RA2211056010068)")
     st.write("Pavithra GV-(RA2211056010046)")
     st.write("Susmitha V-(RA2211056010038)")
     