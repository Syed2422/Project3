# Project3


##Goal
The aim of this project is to predict stock price performance by analyzing the sentiment of public earnings call transcripts. Using advanced generative AI and Natural Language Processing (NLP) techniques, we identified correlations between sentiment, EPS growth, and price movements. By integrating sentiment insights with financial metrics, this project delivers sentiment-driven investment recommendations to aid informed decision-making.

##Solution
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
