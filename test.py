from binance.client import Client
import time

# Your API keys
api_key = "Xb9VpOSPolOxyBg7FkvFGwyKwo56dmr5LwB39NYmQ6XSvw3IYyAV8c2MCUDKHkJD"
secret_key = "pNE36TqlqgTahy6RykoooCo7Yfw2E1KzHtHE444Nduxw3m0K6P7tAHAKKwzFIG3R"

# Initialize client
client = Client(api_key, secret_key)

try:
    # Test connection
    status = client.get_system_status()
    print(f"System status: {status}")

    # Get recent trades
    trades = client.get_recent_trades(symbol='BTCUSDT', limit=5)
    print("\nRecent trades:")
    for trade in trades:
        print(trade)

    # Get order book
    depth = client.get_order_book(symbol='BTCUSDT', limit=5)
    print("\nOrder book:")
    print("Bids:", depth['bids'][:2])
    print("Asks:", depth['asks'][:2])

except Exception as e:
    print(f"Error: {e}")