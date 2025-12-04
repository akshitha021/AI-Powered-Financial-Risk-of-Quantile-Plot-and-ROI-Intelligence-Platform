#Financial Risk Quantile Plot + Project ROI Simulator + AI _ Based Investment Scoring
#Step 1: Getting Data

# Financial Data
# Install the library
!pip install yfinance

# Import the libraries
import yfinance as yf
import pandas as pd

# Choose a stock ticker (e.g., AAPL for Apple)
ticker_symbol = "AAPL"
stock = yf.Ticker(ticker_symbol)

# Get historical price data
price_data = stock.history(period="1y")
print(f"--- Historical Prices for {ticker_symbol} ---")
print(price_data.head())

# Get company information
company_info = stock.info
print(f"\n--- Company Info: Sector for {ticker_symbol} ---")
print(company_info['sector'])

# Get financial statements
financials = stock.financials
print(f"\n--- Recent Financials for {ticker_symbol} ---")
print(financials)

# News Headlines
# Install the library
!pip install newsapi-python

# Import the library
from newsapi import NewsApiClient

# --- IMPORTANT: Replace with your actual API key! ---
# Get your API key from https://newsapi.org/account
api_key = '746762a46cc041e1be7ea53f3a951018'

# Initialize the client
newsapi = NewsApiClient(api_key=api_key)

# Search for news about your company
company_name = "Apple"
all_articles = newsapi.get_everything(q=company_name,
                                      language='en',
                                      sort_by='publishedAt',
                                      page_size=5) # Get the 5 most recent articles

# Print the headlines
print(f"--- Recent News Headlines for {company_name} ---")
for article in all_articles['articles']:
    print(f"- {article['title']}")

import os

# --- PASTE YOUR KAGGLE.JSON CONTENT INSIDE THE QUOTES ---
kaggle_api_key_content = '{"username":"your-username","key":"your-api-key-string"}'

# --- This code sets up the key without uploading any files ---
!mkdir -p ~/.kaggle
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as file:
    file.write(kaggle_api_key_content)
!chmod 600 ~/.kaggle/kaggle.json

print("✅ Kaggle API key configured successfully without file upload.")

# Download the dataset from Kaggle
!kaggle datasets download -d md516/esg-scores-of-1500-us-companies-2018-2022

# Unzip the downloaded file
!unzip esg-scores-of-1500-us-companies-2018-2022.zip
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('company_esg_financial_dataset.csv')

# Display the first 5 rows to confirm it loaded correctly
print(df.head())

 # In Your Notebook: Unzip and Read the Data.
 # Import the pandas library to work with the data
import pandas as pd

# --- IMPORTANT: Change the zip filename below if yours is different! ---
zip_filename = 'esg-scores-of-1500-us-companies-2018-2022.zip'
!unzip {zip_filename} # Uncomment and run this after successfully downloading the zip file

# --- IMPORTANT: Check the unzipped files and change the csv filename below ---
csv_filename = 'company_esg_financial_dataset.csv'
esg_df = pd.read_csv(csv_filename) # Uncomment and run this after successfully unzipping the file

print("--- ESG Data Loaded Successfully! ---")
print(esg_df.head()) # Display the first 5 rows of the table

#Step 2: Build the AI Scoring Model 🧠
#Step 2.1: Load and Clean Your Data 🧹
import pandas as pd

#load he dataset you uploaded
df= pd.read_csv('/content/company_esg_financial_dataset.csv')

#------- Clean the data -----------

# 1. Filling missing value 'GrowthRate' with average of the column.
df['GrowthRate'].fillna(df['GrowthRate'].mean(), inplace = True)

# Also fill any other potential missing value in numeric columns
df.fillna(0, inplace = True)

# 2. Convert text columns into numerical columns using one-hot encoding
df_processed = pd.get_dummies(df, columns = ['Industry', 'Region'], drop_first = True)

print("------------- Data Cleaned and Processesd------------------")
df_processed.head()

#Step 2.2: Define What Makes a "Good Investment" 🎯

# A good investment is a company that has both a high Profit Margin and a high ESG Score.
# Define what a 'good' investment is
# Lets say its a compnay with Profit Margin > 10% AND an ESG score > 60
profit_threshold = 10
esg_threshold = 60

df_processed['Is_Good_Investment'] = ((df_processed['ProfitMargin'] > profit_threshold) & (df_processed['ESG_Overall'] > esg_threshold)).astype(int)

print("--------- 'Is_Good_Investment' Column created ----------------")
print(df_processed[['CompanyName', 'ProfitMargin', 'ESG_Overall', 'Is_Good_Investment']].head())

#Step 2.3: Separate Your Features and Target.

# Features (X): All the data we will use to make a decision (e.g., Revenue, MarketCap, Industry_Finance, etc.).

# Target (y): The one thing we want to predict (Is_Good_Investment).

# 'y' is our target - the column we want to predict
y = df_processed['Is_Good_Investment']

# 'X' are our features - all the columns we use to make the prediction
# We drop the original company info and the target itself
X = df_processed.drop(columns=['CompanyID', 'CompanyName', 'Year', 'Is_Good_Investment'])

print("--- Features (X) ---")
print(X.head())

print("\n--- Target (y) ---")
print(y.head())

#Step 2.4: Train the AI Model 🧠
# Install XGBoost if you haven't already
!pip install xgboost

# Import the necessary tools
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Split data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the XGBoost model (a "Classifier" because we are predicting a category: 0 or 1)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model on the training data
print("--- Training the AI Model... ---")
model.fit(X_train, y_train)
print("--- Model Training Complete! ---")

#Step 2.5: Check How Smart the AI Is 🎓

from sklearn.metrics import accuracy_score

# Make predictions on the unseen test data
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print(f"--- Model Performance ---")
print(f"The model's accuracy on the test data is: {accuracy * 100:.2f}%")

#Step 3.1: Install SHAP and Set It Up
# Install the SHAP library
!pip install shap

# Import the library
import shap

# This line is important for making SHAP plots look nice in the notebook
shap.initjs()

# We already have our trained 'model' and our data 'X_train' and 'X_test' from Step 2.
# We will use them directly in the next steps.
print("--- SHAP is installed and ready to use. ---")

import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- This is the part you are missing ---

# 1. Split your data into training and testing sets
# (Assuming 'X' is your feature data and 'y' is your target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a model
model = RandomForestRegressor().fit(X_train, y_train)

# 3. Create a SHAP explainer object
explainer = shap.Explainer(model)

# 4. Calculate the SHAP values for your test data
shap_values = explainer(X_test)

# --- Now your original code will work ---

# Put initjs() right before the plot command in the SAME cell
shap.initjs()
shap.summary_plot(shap_values, X_test)


#Step 3.2: Create the "Explainer"
#Goal: Connect SHAP to our AI model's "brain."

# Create the SHAP Explainer object, linking it to our trained model
# It looks at the model's structure and the training data to understand its logic
explainer = shap.TreeExplainer(model)

print("--- SHAP Explainer has been created. ---")

#Step 3.3: Calculate the Explanations (SHAP Values)

# Calculate the SHAP values for all the data in our test set
# This can take a moment to run
shap_values = explainer.shap_values(X_test)

print("--- SHAP values have been calculated successfully. ---")

#Step 3.4: Visualize and Understand the AI's Logic 📊
# Create a summary plot to see the overall importance of each feature
print("--- Global Feature Importance Summary ---")
shap.summary_plot(shap_values, X_test)

import shap

# --- This is the fix ---
# Load the SHAP Javascript library (must be in the same cell as the plot)
shap.initjs()

# Now, create the plot
print("\n--- Explanation for a Single Prediction ---")
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

#step 4: Simulate the Future (Stress-Test) 🌪️
import numpy as np
import pandas as pd

# Load the original (unprocessed) dataframe again to get historical data for one company
df = pd.read_csv('company_esg_financial_dataset.csv')

# --- Select a company to analyze (e.g., Company_1) ---
company_id_to_simulate = 1
company_data = df[df['CompanyID'] == company_id_to_simulate]

# --- Define our assumptions from its historical data ---
# Expected return is the average growth rate
expected_return = company_data['GrowthRate'].mean() / 100 # Convert to decimal

# Volatility is the standard deviation of the growth rate
volatility = company_data['GrowthRate'].std() / 100 # Convert to decimal

# Starting investment value (we can use its most recent Market Cap)
initial_investment = company_data['MarketCap'].iloc[-1]

print(f"--- Simulation Assumptions for Company {company_id_to_simulate} ---")
print(f"Initial Investment: ${initial_investment:,.2f}")
print(f"Expected Annual Return: {expected_return:.2%}")
print(f"Annual Volatility: {volatility:.2%}")

#Step 4.2: Build the Simulation Engine ⚙️
# Number of simulations to run
num_simulations = 1000
# Number of years to simulate
num_years = 5

# We'll simulate daily price changes, so we need the number of trading days
num_trading_days = 252

# Create an empty array to store the results of all simulations
simulation_results = np.zeros((num_simulations, num_trading_days * num_years))

print("--- Monte Carlo Simulation Engine is Ready ---")

#Step 4.3: Run Thousands of "What-If" Scenarios 🌪️
print("--- Running Simulations... ---")
for i in range(num_simulations):
    # Start with the initial investment
    price_series = [initial_investment]

    # Simulate day by day for 5 years
    for t in range(1, num_trading_days * num_years):
        # Calculate daily return with a random shock
        daily_return = np.random.normal(expected_return / num_trading_days, volatility / np.sqrt(num_trading_days))
        # Calculate the new price
        new_price = price_series[-1] * (1 + daily_return)
        price_series.append(new_price)

    # Store the full price series for this simulation
    simulation_results[i,:] = price_series

print("--- 1,000 simulations complete! ---")

#Step 4.4: Analyze the Risk and Reward 📉
# Get the final value from the last day of each simulation
final_values = simulation_results[:,-1]

# --- Calculate Key Metrics ---

# Calculate the average final value and ROI
average_final_value = final_values.mean()
average_roi = (average_final_value - initial_investment) / initial_investment

# Calculate Value at Risk (VaR) at the 5% level
# This means we are 95% confident that our losses will not exceed this amount.
var_5_percent = np.percentile(final_values, 5)
loss_at_var = initial_investment - var_5_percent

print(f"--- Simulation Analysis ---")
print(f"Average Final Value after 5 years: ${average_final_value:,.2f}")
print(f"Average ROI: {average_roi:.2%}")
print(f"Value at Risk (5%): We are 95% confident the maximum loss will not exceed ${loss_at_var:,.2f}")

#Step 4.5: Visualize the Risk with a Histogram & Heat Map 🔥
import matplotlib.pyplot as plt
import seaborn as sns

# Set a nice style for the plot
sns.set_style('whitegrid')

# Create the histogram (our 'Risk Heat Map')
plt.figure(figsize=(10, 6))
sns.histplot(final_values, kde=True, bins=50)

# Add lines for key metrics
plt.axvline(initial_investment, color='black', linestyle='--', label='Initial Investment')
plt.axvline(average_final_value, color='red', linestyle='--', label='Average Outcome')
plt.axvline(var_5_percent, color='orange', linestyle='--', label='Value at Risk (5%)')

plt.title(f'Distribution of Possible Investment Outcomes for Company {company_id_to_simulate}')
plt.xlabel('Final Investment Value ($)')
plt.ylabel('Frequency (Number of Simulations)')
plt.legend()
plt.show()

#Quantile Plot

import matplotlib.pyplot as plt
import numpy as np

# Calculate the percentiles for each time step (each day)
# We want to see the 5th, 25th, 50th (median), 75th, and 95th percentiles
p5 = np.percentile(simulation_results, 5, axis=0)
p25 = np.percentile(simulation_results, 25, axis=0)
p50 = np.percentile(simulation_results, 50, axis=0)
p75 = np.percentile(simulation_results, 75, axis=0)
p95 = np.percentile(simulation_results, 95, axis=0)

# Create the plot
plt.figure(figsize=(12, 7))

# Plot the median line
plt.plot(p50, label='Median Outcome (50%)', color='red')

# Fill the area between the percentiles to create the "fan"
plt.fill_between(range(len(p50)), p25, p75, color='blue', alpha=0.3, label='Interquartile Range (25%-75%)')
plt.fill_between(range(len(p50)), p5, p95, color='blue', alpha=0.1, label='Full Range (5%-95%)')

plt.title('Fan Chart of Investment Simulation Over 5 Years')
plt.xlabel('Time (Trading Days)')
plt.ylabel('Investment Value ($)')
plt.legend()
plt.grid(True)
plt.show()