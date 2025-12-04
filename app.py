
# ==============================================================================
# Step 1: Setup and Configuration
# ==============================================================================
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')
print("--- Libraries Imported ---")

# ==============================================================================
# IMPORTANT: PASTE YOUR NEWSAPI KEY HERE
# ==============================================================================
NEWS_API_KEY = "746762a46cc041e1be7ea53f3a951018"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# ==============================================================================
# Step 2: Data Gathering and Processing Functions
# ==============================================================================
def get_financial_data(ticker_list):
    """Downloads key financial data for a list of tickers."""
    data_list = []
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            data = {
                'Ticker': ticker,
                'MarketCap': info.get('marketCap', 0),
                'Revenue': info.get('totalRevenue', 0),
                'ProfitMargin': info.get('profitMargins', 0),
                'ForwardPE': info.get('forwardPE', 0),
                '52WeekChange': info.get('52WeekChange', 0),
                'Volatility': hist['Close'].pct_change().std() * np.sqrt(252) # Annualized volatility
            }
            data_list.append(data)
        except Exception as e:
            print(f"Could not get data for {ticker}: {e}")
    return pd.DataFrame(data_list)

def get_news_sentiment(company_name):
    """Fetches recent news and calculates a simple sentiment score."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        all_articles = newsapi.get_everything(q=company_name,
                                              from_param=start_date.strftime('%Y-%m-%d'),
                                              to=end_date.strftime('%Y-%m-%d'),
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=20)

        if not all_articles['articles']:
            return 0

        # Simple sentiment: count positive vs. negative words
        positive_words = ['good', 'great', 'up', 'profit', 'strong', 'growth', 'success', 'beat', 'exceed']
        negative_words = ['bad', 'poor', 'down', 'loss', 'weak', 'decline', 'fail', 'miss']

        sentiment_score = 0
        for article in all_articles['articles']:
            content = (article['title'] + ' ' + article['description']).lower() if article['description'] else article['title'].lower()
            sentiment_score += sum(content.count(word) for word in positive_words)
            sentiment_score -= sum(content.count(word) for word in negative_words)

        return sentiment_score / len(all_articles['articles'])
    except Exception as e:
        # Handle common API errors (e.g., key missing)
        print(f"NewsAPI Error for {company_name}: {e}")
        return 0

print("--- Data Functions Defined ---")

# ==============================================================================
# Step 3: AI Model Training
# ==============================================================================
def train_ai_model():
    """Trains a simple AI model to classify investments."""
    # We need a broad list of tickers to train a decent model
    print("Fetching training data for the AI model...")
    sp500_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'JNJ', 'V', 'PG', 'XOM', 'CVX', 'KO', 'PFE']
    training_data = get_financial_data(sp500_tickers)

    # Create a simple 'Is_Good_Investment' target based on rules
    # This is a simplification; in the real world, this would be historical performance.
    training_data['Is_Good_Investment'] = np.where(
        (training_data['ForwardPE'] > 0) & (training_data['ForwardPE'] < 40) & (training_data['ProfitMargin'] > 0.1),
        1, 0
    )

    # Define features (X) and target (y)
    features = ['MarketCap', 'Revenue', 'ProfitMargin', 'ForwardPE', '52WeekChange', 'Volatility']
    X = training_data[features].fillna(0) # Fill missing values
    y = training_data['Is_Good_Investment']

    # Split data and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    print("--- AI Model Trained Successfully ---")
    return model, features

# Train the model once
ai_model, model_features = train_ai_model()
explainer = shap.TreeExplainer(ai_model)

# ==============================================================================
# Step 4: Explanation and Simulation Functions (DEFINITIVE FINAL VERSION)
# ==============================================================================
# Required imports for this step
from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_ai_explanation(model_input):
    """Gets the top reason for an AI prediction using SHAP."""
    shap_values = explainer.shap_values(model_input)
    shap_df = pd.DataFrame(shap_values, columns=model_features)
    abs_shap_values = shap_df.abs().iloc[0]
    top_factor_name = abs_shap_values.idxmax()
    top_factor_value = model_input[top_factor_name].iloc[0]
    top_factor_shap_value = shap_df[top_factor_name].iloc[0]

    if top_factor_shap_value > 0:
        impact = "positively"
    else:
        impact = "negatively"

    return f"{top_factor_name} of {top_factor_value:.2f} impacted the score {impact}."

def run_monte_carlo_simulation(ticker, days=252, simulations=1000):
    """Runs a Monte Carlo simulation using yfinance instead of YahooFinancials."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")['Close']

        if hist.empty or len(hist) < 2:
            print(f"--> Simulation Error for {ticker}: Insufficient data from yfinance.")
            return 0, 0

        log_returns = np.log(1 + hist.pct_change().dropna())
        mu, sigma = log_returns.mean(), log_returns.std()

        if sigma == 0 or pd.isna(sigma):
            print(f"--> Simulation Error for {ticker}: No volatility in historical data.")
            return 0, 0

        last_price = hist.iloc[-1]
        price_paths = np.zeros((days + 1, simulations))
        price_paths[0] = last_price

        for t in range(1, days + 1):
            rand_shocks = np.random.normal(mu, sigma, simulations)
            price_paths[t] = price_paths[t - 1] * np.exp(rand_shocks)

        final_prices = price_paths[-1]
        simulated_roi = (np.mean(final_prices) / last_price - 1) * 100
        value_at_risk = (np.percentile(final_prices, 5) / last_price - 1) * 100

        return simulated_roi, value_at_risk

    except Exception as e:
        print(f"--> An unexpected simulation error occurred for {ticker}: {e}")
        return 0, 0

# ==============================================================================
# Step 5: Main Execution Loop - Analyze Target Companies
# ==============================================================================
# DEFINE THE COMPANIES YOU WANT TO ANALYZE HERE
target_companies = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM',
    'JNJ', 'V', 'PG', 'XOM', 'CVX', 'KO', 'PFE'
]
all_results_list = []

print(f"\n--- Starting Analysis for {len(target_companies)} Target Companies ---")
for ticker in target_companies:
    print(f"\nAnalyzing {ticker}...")

    # 1. Get financial data
    financials = get_financial_data([ticker])
    if financials.empty:
        continue

    # 2. Get news sentiment (Temporarily disabled to avoid API limit)
    # news_score = get_news_sentiment(ticker)
    news_score = 0   # Set a default value

    # 3. Get AI prediction
    model_input = financials[model_features].fillna(0)
    ai_score = ai_model.predict_proba(model_input)[0][1]

    # 4. Get AI explanation
    explanation = get_ai_explanation(model_input)

    # 5. Run simulation
    sim_roi, sim_var = run_monte_carlo_simulation(ticker)

    # 6. Collect results
    result = {
        'Company': ticker,
        'AI_Score': round(ai_score, 2),
        'Key_Factor': explanation,
        'Simulated_ROI_Percent': round(sim_roi, 2),
        'Value_at_Risk_Percent': round(sim_var, 2),
        'News_Sentiment_Score': round(news_score, 2),
        'MarketCap': financials['MarketCap'].iloc[0],
        'ProfitMargin': financials['ProfitMargin'].iloc[0],
        'ForwardPE': financials['ForwardPE'].iloc[0]
    }

    # 7. Append the final result and pause
    all_results_list.append(result)
    time.sleep(1) # Pause after each company is successfully processed

print("\n--- Analysis Complete for All Companies ---")

# ==============================================================================
# Step 6: Save Final Output for Power BI
# ==============================================================================
final_df = pd.DataFrame(all_results_list)

# 1. Paste the full file path to your OneDrive file here
# For Windows, the 'r' before the string is very important.
file_path = r'C:\Users\akshitha\OneDrive\powerBI_dashboards\powerbi_data.csv'

# 2. Save the DataFrame to that specific path
final_df.to_csv(file_path, index=False)

# --- END OF UPDATE ---

print("\n--- Final, Combined Table ---")
print(final_df)
print(f"\nSuccess! The file has been saved to your OneDrive at: {file_path}")
