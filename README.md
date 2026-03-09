<H1> Machine Learning Models for Financial Market Forecasting
A Comparative Study and Hybrid Forecasting Framework </H1>


<h2> Overview </h2>

This project investigates the effectiveness of different statistical and machine learning models for stock market forecasting. The objective is to compare multiple forecasting approaches, identify the best-performing model, and enhance its predictive capability by integrating it with machine learning techniques.
The study focuses on combining linear time-series modeling with non-linear machine learning methods to improve prediction accuracy for financial data.

<h2> Project Objectives </h2>

• Compare the performance of multiple forecasting models  
• Evaluate models using standard error metrics  
• Identify the best-performing baseline model  
• Develop a hybrid forecasting framework  
• Validate models using walk-forward validation  
• Visualize results using an analytical dashboard

<h2> Models Implemented </h2>

Five forecasting models were evaluated:

<h3> Autoregressive Model – AR(5): </h3>
Captures linear relationships using previous price values.

<h3> Moving Average Model: </h3>
Smooths the time series and captures short-term trends.

<h3> Random Walk Model: </h3>
Used as a baseline model based on the Efficient Market Hypothesis.

<h3> Random Forest Regressor: </h3>
Ensemble machine learning model capable of capturing non-linear patterns.

<h3> Gradient Boosting Regressor: </h3>
Sequential ensemble model designed to improve prediction accuracy through iterative error correction.

<h3> Hybrid Model Approach: </h3> 
After comparing all models, the best-performing baseline model was combined with machine learning models to create a hybrid forecasting system.

The hybrid approach works by: 
Modeling the linear component of the time series using AR(5).
Learning the remaining non-linear patterns using machine learning models. 
Combining both components to generate the final prediction. 
This approach allows the model to capture both: 
linear market trends, 
complex non-linear relationships present in financial data.

<h2> Validation Method </h2>

The models were evaluated using Walk-Forward Validation, which simulates real-world forecasting conditions.
Process:

Train the model on historical data up to a specific time.
Predict the next time period.
Expand the training window.
Repeat until the end of the dataset.
This method prevents data leakage and provides a realistic estimate of model performance.


<h2> Evaluation Metrics </h2>

Model performance was measured using the following error metrics:

RMSE (Root Mean Squared Error) – Measures prediction accuracy. 
MAE (Mean Absolute Error) – Measures average absolute prediction error. 
MAPE (Mean Absolute Percentage Error) – Measures percentage prediction error. 
R² (Coefficient of Determination) – Indicates goodness of fit. 

<h2> Dashboard Preview </h2>

### Market Trend and Forecast
![Market Trend](Market_Trend_+_Forecast.png)

### Model Comparison
![Model Comparison](Model_Comparison.png)

### Walk Forward Validation
![Walk Forward Validation](Walk_Forward_Validation.png)


<h2> Data </h2>

Source - Kaggle - NIFTY 50 Historical Dataset

The dataset contains historical stock index data including:

Date, 
Opening price, 
Closing price, 
High, 
Low, 
Volume

The dataset was cleaned and sorted chronologically before model training.

<h2> Dashboard </h2>

A Power BI dashboard was created to visualize model performance and compare forecasting accuracy across models.
The dashboard displays:
RMSE comparison, 
MAPE comparison, 
Model performance summary, 
Best performing model identification

<h2> Technologies Used </h2>

Python Libraries:
• Pandas, 
• NumPy, 
• Scikit-learn, 
• Statsmodels, 
• Matplotlib

Visualization:
Power BI

<h2> Key Findings </h2>

Machine learning models demonstrated lower prediction error compared to traditional statistical models.
Gradient Boosting achieved the best performance among the evaluated models.
Hybridizing statistical and machine learning approaches improved forecasting capability by capturing both linear and non-linear patterns.

<h2> Future Improvements </h2>
Incorporate additional features such as macroeconomic indicators and technical indicators. 
Explore deep learning approaches such as LSTM for sequential data. 
Apply the hybrid framework to multiple financial markets. 
Implement real-time prediction pipelines.
