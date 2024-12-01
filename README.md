
# Boardmeeting sentimental 
# Stock Price Prediction Using Sentiment Analysis and Financial Data


## Overview
This project is focused on predicting the 7-day stock price growth based on earnings data, sentiment analysis from earnings call transcripts, and financial features. We utilized two models: Random Forest and XGBoost, to predict stock growth and made a comparative analysis between these two models to select the most effective one.

## Steps Involved in the Project

### 1. Data Collection
- **Tickers Selection**: The tickers were selected across various GICS sectors to ensure diversity and represent different industries. Specifically, we chose two tickers from each of the following sectors:
  - **Information Technology**: Apple (AAPL), Microsoft (MSFT)
  - **Healthcare**: Johnson & Johnson (JNJ), Pfizer (PFE)
  - **Financials**: JPMorgan Chase (JPM), Bank of America (BAC)
  - **Consumer Discretionary**: Amazon (AMZN), Tesla (TSLA)
  - **Consumer Staples**: Procter & Gamble (PG), Coca-Cola (KO)
  - **Energy**: ExxonMobil (XOM), Chevron (CVX)
  - **Industrials**: Boeing (BA), Caterpillar (CAT)
  - **Materials**: Linde (LIN), Dow Inc. (DOW)
  - **Real Estate**: American Tower (AMT), Simon Property Group (SPG)
  - **Utilities**: NextEra Energy (NEE), Duke Energy (DUK)
  - **Communication Services**: Alphabet (GOOGL), Meta (META)

- **Timeframe**: Data was collected from 2019 to 2023 for each ticker. This provides a comprehensive historical view of financial data and stock price performance across different economic conditions.

- **Earnings Data**: Collected earnings data for selected companies using the Alpha Vantage API. This data includes quarterly earnings, reported EPS (Earnings Per Share), and estimated EPS.

- **Transcript Data**: Collected earnings call transcripts using the Ninja API. These transcripts are analyzed for sentiment, which is used as a feature for predicting stock growth.

- **Stock Data**: Collected stock price data for the selected companies using the Yahoo Finance API. This data is used to calculate the stock price growth over the specified periods (7 days, 14 days, 30 days).

- **Timeframe**: Data was collected from 2019 to 2023 for each ticker. This provides a comprehensive historical view of financial data and stock price performance across different economic conditions.

## 2. Financial Analysis
We conducted financial analysis to derive the following key insights:
- **EPS Analysis**: We compared the reported EPS with the estimated EPS for each quarter, looking at EPS surprises. A positive surprise (reported EPS greater than estimated EPS) often correlates with a stock price increase.
- **Stock Price Growth**: We calculated stock price growth over multiple timeframes (7-day, 14-day, 30-day) to understand how stock prices move in response to earnings reports and sentiment.
- **Sentiment and EPS Surprise Correlation**: Analyzed the relationship between sentiment (from earnings call transcripts) and EPS surprise, determining how sentiment can impact stock price movement.
  
### 3. Data Cleaning and Preprocessing
- **Text Cleaning**: The earnings call transcripts are cleaned by removing unnecessary characters, punctuation, and stop words. This is essential for the sentiment analysis step.
- **Feature Engineering**: Created features like EPS surprise (difference between reported and estimated EPS), sentiment (positive, neutral, negative), and stock growth (7-day growth, 14-day growth, etc.).
- **Sentiment Analysis**: Sentiment of the earnings call transcripts is analyzed using OpenAI's GPT-3.5 Turbo model. The transcripts are processed through the model to classify the sentiment as either positive, negative, or neutral.

### 4. Model Selection
We evaluated two machine learning models for predicting stock price growth:
- **Random Forest**: A powerful ensemble method that uses multiple decision trees to make predictions. It is less prone to overfitting and handles nonlinear data well.
- **XGBoost**: A gradient boosting framework known for high performance and accuracy in machine learning competitions. It works well on structured/tabular data with a large number of features.

### 5. Model Fine-Tuning
- **Hyperparameter Tuning**: We performed hyperparameter tuning using RandomizedSearchCV to find the best configuration for both models. This step helped optimize parameters like the number of estimators, maximum depth, learning rate, and others.
- **Cross-Validation**: We used 5-fold cross-validation to evaluate the models and ensure they generalize well on unseen data.

### 6. Model Evaluation
- **Random Forest**: The Random Forest model performed well with a low Mean Squared Error (MSE) of approximately 0.08, a Mean Absolute Error (MAE) of 0.24, and an R-squared score of 0.79.
- **XGBoost**: The XGBoost model showed good performance with an MSE of approximately 0.15, an MAE of 0.29, and an R-squared score of 0.72.

We decided to choose **Random Forest** for the final model due to its better generalization and robustness on the dataset.

### 7. Model Testing
After training, we tested the model on a separate dataset (20% of the total data) to ensure it performed well on unseen data. The Random Forest model had better performance in terms of accuracy, which is why we selected it for the final deployment.

### 8. Model Deployment
We deployed the model using Gradio, a Python library that allows for the creation of easy-to-use web interfaces for machine learning models. Users can input a stock ticker, year, and quarter, and the system will predict the 7-day stock growth based on earnings data, sentiment analysis, and stock price data.

### 9. Output
The output from the model includes:
- **Sentiment Analysis**: The sentiment of the earnings call transcript (positive, negative, or neutral).
- **Stock Price Growth**: The predicted 7-day stock growth based on the model's analysis.
- **Stock Price Data**: The stock prices before and after the earnings call to visualize the growth.
- **Recommendation**: The recommendation to buy, sell, or hold based on the sentiment and earnings data.
- **Charts**: Charts showing the growth over 7 days and EPS comparison (reported vs. estimated).

### 10. OpenAI GPT-3.5 Turbo Integration
We used **OpenAI GPT-3.5 Turbo** for sentiment analysis of earnings call transcripts. The transcripts are passed through the model, and the sentiment is classified as positive, neutral, or negative. This sentiment score is then used as an important feature in the stock growth prediction model.

### 11. Gradio Interface
The Gradio interface allows users to input a stock ticker, year, and quarter to get the stock sentiment analysis recommendation to buy, sell, or hold.

## Requirements
- Python 3.9+
- Libraries: pandas, scikit-learn, numpy, yfinance, gradio, requests, openai, matplotlib
- API keys for Alpha Vantage, Ninja API, and OpenAI

## Future Improvements
- **Incorporate more data**: We can expand the dataset to include more features like social media sentiment, historical price trends, or macroeconomic factors.
- **Model Optimization**: Further hyperparameter tuning and feature engineering can be done to improve model accuracy.
- **Real-Time Data**: Integrate real-time stock price and earnings call transcript data for up-to-date predictions.

## Conclusion
This project demonstrates how machine learning models, specifically Random Forest, can be used to predict stock price growth based on financial data and sentiment analysis. The integration of OpenAI's GPT-3.5 Turbo for sentiment analysis provides an additional layer of insight that helps improve prediction accuracy.


# Project3


## Goal
The aim of this project is to predict stock price performance by analyzing the sentiment of public earnings call transcripts. Using advanced generative AI and Natural Language Processing (NLP) techniques, we identified correlations between sentiment, EPS growth, and price movements. By integrating sentiment insights with financial metrics, this project delivers sentiment-driven investment recommendations to aid informed decision-making.

## Solution
1. Data Sources:
- Utilized API Ninjas and Alpha Vantage to gather earnings call data, financial metrics, and historical stock prices.
- Integrated OpenAI’s capabilities to perform sentiment analysis on earnings call transcripts.
2. Analysis Process:
- Conducted a comprehensive analysis of earnings call transcripts to generate overall sentiment scores.
- Leveraged Hugging Face's LLM model for advanced sentiment classification.
3. Stock Price Prediction:
- Combined sentiment scores with financial data to predict stock price growth and identify investment opportunities.
This pipeline provides a robust framework for analyzing earnings call sentiments and linking them to stock performance, empowering investors with actionable insights.

APIs:
- https://api-ninjas.com/api/earningscalltranscript
- https://www.alphavantage.co/query
- https://api.openai.com/v1/chat/completions
