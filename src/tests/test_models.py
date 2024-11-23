import os
import logging
from datetime import datetime, timedelta
from src.config import settings
from src.data.database.connection import MongoDBConnection
from src.models.training import ModelTrainer

def test_ml_pipeline():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize database connection
        db_config = {
            'connection_string': settings.mongodb_uri,
            'name': settings.db_name
        }
        db = MongoDBConnection(db_config)

        if not db.connect():
            logger.error("Failed to connect to database")
            return

        # Initialize model trainer
        trainer = ModelTrainer(db)

        # Set time ranges for training and testing
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # Use 30 days of data
        test_start = end_time - timedelta(days=7)   # Last 7 days for testing

        for symbol in settings.trading_symbols:
            logger.info(f"\nTraining models for {symbol}")

            # Train models
            models = trainer.train_models(
                symbol=symbol,
                start_time=start_time,
                end_time=test_start  # Train on data up to test period
            )

            # Generate test features
            test_features = trainer.feature_eng.generate_features(
                symbol=symbol,
                start_time=test_start,
                end_time=end_time
            )

            if test_features.empty:
                logger.warning(f"No test data available for {symbol}")
                continue

            # Evaluate individual models
            results = trainer.evaluate_models(models, test_features)

            # Create and evaluate ensemble
            weights = {'lstm': 0.4, 'lightgbm': 0.6}
            ensemble = trainer.create_ensemble(models, weights)

            ensemble_results = trainer.evaluate_models(
                {'ensemble': ensemble},
                test_features
            )

            # Save models
            os.makedirs(f"models/{symbol}", exist_ok=True)
            for name, model in models.items():
                model.save(f"models/{symbol}/{name}")
            ensemble.save(f"models/{symbol}/ensemble")

            # Perform cross-validation
            logger.info(f"\nPerforming cross-validation for {symbol}")
            cv_results = trainer.cross_validate(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )

    except Exception as e:
        logger.error(f"Error testing ML pipeline: {str(e)}")

if __name__ == "__main__":
    test_ml_pipeline()