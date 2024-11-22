import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import json

def test_connection():
    try:
        # Load environment variables
        load_dotenv()

        # Get MongoDB URI from environment
        mongodb_uri = os.getenv('MONGODB_URI')
        db_name = os.getenv('DB_NAME', 'crypto_trading')

        print(f"Attempting to connect to database: {db_name}")

        # Connect to MongoDB
        client = MongoClient(mongodb_uri)
        db = client[db_name]

        # Try to insert a test document
        test_collection = db.test_collection
        result = test_collection.insert_one({
            "test": "Connection test",
            "timestamp": datetime.now()
        })

        print(f"Successfully connected to MongoDB and inserted document with id: {result.inserted_id}")

        # Test reading environment variables
        print("\nEnvironment Variables:")
        print(f"MONGODB_URI: {mongodb_uri[:20]}...") # Only show beginning for security
        print(f"DB_NAME: {db_name}")
        print(f"USE_TESTNET: {os.getenv('USE_TESTNET')}")
        try:
            trading_symbols = json.loads(os.getenv('TRADING_SYMBOLS', '["BTCUSDT"]'))
            print(f"TRADING_SYMBOLS: {trading_symbols}")
        except json.JSONDecodeError as e:
            print(f"Error parsing TRADING_SYMBOLS: {e}")

        # Clean up
        test_collection.delete_one({"_id": result.inserted_id})
        client.close()
        print("\nTest completed successfully!")

        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease check that:")
        print("1. You have a .env file in your project root")
        print("2. The .env file contains MONGODB_URI and DB_NAME")
        print("3. The MongoDB URI is correctly formatted")
        print("4. The MongoDB server is accessible")
        return False

if __name__ == "__main__":
    test_connection()