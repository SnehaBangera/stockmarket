import streamlit as st
from src.fetch_data import fetch_data
from src.preprocess_data import preprocess_data
from src.build_model import build_model
from src.predict import predict_next_20_days
from src.news import fetch_stock_news_alpha_vantage
from src.recommendation import get_recommendation
import matplotlib.pyplot as plt
import pandas as pd

st.markdown("""
    <style>
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(initial_sidebar_state="collapsed")

st.title("📈 Stock Market Prediction")

# Stock Market Prediction Page
def stock_market_prediction():
    st.header("📈 Stock Market Prediction")
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    
    if st.button("Fetch and Predict"):
        with st.spinner("Fetching data..."):
            data = fetch_data(stock_symbol)
            if data is None:
                return
        
        st.success("Data fetched successfully.")
        st.line_chart(data["Close"])
        
        with st.spinner("Processing data..."):
            scaled_data, scaler = preprocess_data(data)
            training_size = int(len(scaled_data) * 0.8)
            train_data, test_data = scaled_data[:training_size], scaled_data[training_size-60:]
        
        with st.spinner("Building model..."):
            model = build_model()
            model.fit(train_data.reshape(train_data.shape[0], train_data.shape[1], 1), train_data, epochs=50, batch_size=32, verbose=0)
        st.success("Model trained successfully.")
        
        with st.spinner("Predicting..."):
            predictions = model.predict(test_data.reshape(test_data.shape[0], test_data.shape[1], 1))
            predictions = scaler.inverse_transform(predictions)
            next_20_days = predict_next_20_days(scaled_data, model, scaler)
        
        last_date = data.index[-1]
        dates = pd.date_range(last_date, periods=21, freq='D')[1:]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, next_20_days, label="Predicted Price", linestyle='--', color='red')
        plt.xticks(rotation=45)
        ax.set_title("Next 20 Days Predicted Stock Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Present Day Value")
        st.write(f"{data['Close'].iloc[-1]:.2f}")
        
        st.subheader("Future Price Range")
        st.write(f"{next_20_days[-1][0]:.2f}")
        
        recommendation = get_recommendation(data['Close'].iloc[-1], next_20_days[-1][0])
        st.subheader(f"Recommendation: {recommendation}")
        
        news = fetch_stock_news_alpha_vantage(stock_symbol)
        st.subheader("Latest News")
        for article in news:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(f"{article['description']}")

if __name__ == "__main__":
    stock_market_prediction()
