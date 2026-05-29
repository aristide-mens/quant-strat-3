# Quant Strategy 3

A Flask-based quantitative trading strategy that generates real-time market forecasts for AAPL using machine learning models.

## Features

- **Real-time Data**: Fetches live AAPL data via yfinance
- **Technical Indicators**: RSI, MACD, ATR, Bollinger Bands, Volume Ratio
- **Multi-horizon Predictions**: 1, 5, 15, 120, 480, 1440-minute horizons
- **ML Models**: Random Forest classifiers and regressors with scikit-learn
- **Background Updates**: Continuous model updates every 15 seconds
- **CSV Export**: Predictions served as downloadable CSV

## API Endpoint

```
GET /forecast.csv
```

Returns a CSV with the latest predictions including:
- Probability of price moving down/up
- Conditional probability based on first touch
- Predicted direction and likely extreme price

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

The app runs on `http://localhost:8080/forecast.csv`

## Render Deployment

1. Create a new Web Service on Render
2. Connect this GitHub repository
3. Set Root Directory: `/` (default)
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn app:app`
6. Deploy!

The service will automatically:
- Install dependencies from `requirements.txt`
- Use Python 3.11.7 (specified in `runtime.txt`)
- Run with gunicorn via `Procfile`
- Start the background forecasting loop on startup

## Dependencies

- Flask: Web framework
- yfinance: Yahoo Finance data fetching
- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning models
- gunicorn: Production WSGI server

## Architecture

- **app.py**: Main Flask application
- **requirements.txt**: Python dependencies
- **Procfile**: Render deployment configuration
- **runtime.txt**: Python version specification

---
*Built for Render deployment with all production requirements configured.*