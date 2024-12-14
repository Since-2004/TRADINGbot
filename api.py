import yfinance as yf

def fetch_stock_data(symbol, period="1mo", interval="1d"):
    """
    Fetches historical stock data from Yahoo Finance and optionally saves it to a CSV file.

    :param symbol: Stock symbol (e.g., 'AAPL' for Apple).
    :param period: Data period (e.g., '1d', '1mo', '1y', 'max').
    :param interval: Data interval (e.g., '1d', '1wk', '1m').
    """
    try:
        # Fetch the stock data
        stock = yf.Ticker(symbol)
        historical_data = stock.history(period=period, interval=interval)
        
        if not historical_data.empty:
            print(f"Stock data for {symbol}:")
            print(historical_data)
            
            # Save the data to a CSV file
            csv_filename = f"{symbol}_historical_data.csv"
            historical_data.to_csv(csv_filename)
            print(f"Data saved to {csv_filename}")
        else:
            print(f"No historical data found for {symbol}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main block to interact with the user
if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., AAPL for Apple): ").strip()
    period = input("Enter period (e.g., 1mo, 6mo, 1y, max): ").strip() or "1mo"
    interval = input("Enter interval (e.g., 1d, 1wk, 1m): ").strip() or "1d"
    
    fetch_stock_data(symbol, period, interval)
