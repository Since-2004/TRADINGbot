import streamlit as st
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Title and description
st.title("Interactive Stock Analysis Dashboard")
st.markdown("Explore Buy/Sell Charts, Trend Predictions, and Sentiment Analytics for Stocks.")

# Buttons for different functionalities
if st.button("Buy/Sell Charts"):
    st.subheader("Buy/Sell Charts")
    try:
        # Run P1.py and visualize output
        subprocess.run(['python', 'P1.py'], check=True)
        st.success("Buy/Sell charts generated successfully.")
    except Exception as e:
        st.error(f"Error running P1.py: {e}")

if st.button("Trend Prediction"):
    st.subheader("Trend Prediction")
    try:
        # Run P2.py and show summary
        subprocess.run(['python', 'P2.py'], check=True)
        st.success("Trend prediction model executed.")
        st.markdown("Check your local folder for saved model and evaluation metrics.")
    except Exception as e:
        st.error(f"Error running P2.py: {e}")

if st.button("Sentiment Analytics"):
    st.subheader("Sentiment Analytics")
    try:
        # Run P3.py and display sentiment results
        subprocess.run(['python', 'P3.py'], check=True)
        st.success("Sentiment analysis completed.")
        # Display sentiment results from CSV (if saved by P3.py)
        sentiment_file = "sentiment_analysis_results.csv"
        if sentiment_file:
            df = pd.read_csv(sentiment_file)
            st.write("### Sentiment Analysis Results")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error running P3.py: {e}")

# Add sidebar for navigation and additional configurations
st.sidebar.title("Settings")
st.sidebar.markdown("Configure parameters and options here.")
