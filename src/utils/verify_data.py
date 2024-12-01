import logging
from typing import Dict
import pandas as pd
from datetime import datetime, timedelta

def verify_ohlcv_data(db_connection) -> Dict:
    """Verify OHLCV data in database"""
    logger = logging.getLogger(__name__)

    try:
        # Get price collection
        collection = db_connection.get_collection('price_data')
        if not collection:
            return {"status": "error", "message": "Could not access price_data collection"}

        # Check recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        query = {
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }

        # Get sample data
        data = list(collection.find(query).limit(10))

        if not data:
            return {
                "status": "error",
                "message": "No recent data found"
            }

        # Verify data structure
        required_fields = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        first_doc = data[0]

        missing_fields = [field for field in required_fields if field not in first_doc]
        if missing_fields:
            return {
                "status": "error",
                "message": f"Missing required fields: {missing_fields}",
                "sample": first_doc
            }

        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)

        return {
            "status": "success",
            "message": "Data verification complete",
            "document_count": collection.count_documents({}),
            "recent_count": len(data),
            "fields_present": list(first_doc.keys()),
            "sample": first_doc,
            "statistics": {
                "mean_volume": df['volume'].mean(),
                "price_range": [df['low'].min(), df['high'].max()]
            }
        }

    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return {"status": "error", "message": str(e)}