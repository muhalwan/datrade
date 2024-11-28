#!/usr/bin/env python3


import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from src.features.pipeline import FeaturePipelineManager
from src.data.database.connection import MongoDBConnection
from src.models.trainer import ModelTrainer
from src.models.optimization import ModelOptimizer
from src.models.analysis import ModelAnalyzer
from src.config import settings

def setup_training():
    """Setup training environment"""
    logger = logging.getLogger(__name__)

    # Setup database connection
    db = MongoDBConnection({
        'connection_string': settings.mongodb_uri,
        'name': settings.db_name
    })

    if not db.connect():
        raise ConnectionError("Failed to connect to database")

    # Initialize feature pipeline
    feature_pipeline = FeaturePipelineManager(db)

    # Initialize model trainer
    model_trainer = ModelTrainer(feature_pipeline)

    # Initialize optimizer and analyzer
    model_optimizer = ModelOptimizer(model_trainer, feature_pipeline)
    model_analyzer = ModelAnalyzer(model_trainer)

    return feature_pipeline, model_trainer, model_optimizer, model_analyzer

def train_models(feature_pipeline, model_trainer, model_optimizer, model_analyzer,
                 start_date: datetime, end_date: datetime,
                 models: List[str] = None):
    """Train and optimize models"""
    logger = logging.getLogger(__name__)

    try:
        # Get training data
        logger.info("Fetching training data...")
        data = feature_pipeline.get_historical_features(start_date, end_date)

        if data.empty:
            raise ValueError("No training data available")

        # Default to all models if none specified
        models = models or ['lstm', 'xgboost', 'prophet', 'ensemble']

        results = {}
        for model_name in models:
            logger.info(f"Training {model_name} model...")

            # Optimize hyperparameters
            best_params = model_optimizer.optimize_model(model_name)
            logger.info(f"Best parameters for {model_name}: {best_params}")

            # Train model with best parameters
            model_trainer.schedule_training(model_name)

            # Analyze feature importance
            importance = model_analyzer.analyze_feature_importance(model_name)

            # Analyze prediction errors
            errors = model_analyzer.analyze_prediction_errors(model_name, data)

            results[model_name] = {
                'best_params': best_params,
                'feature_importance': importance,
                'errors': errors
            }

        return results

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise

def save_training_results(results: Dict, output_dir: str):
    """Save training results"""
    logger = logging.getLogger(__name__)

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, model_results in results.items():
            # Save best parameters
            params_file = output_path / f"{model_name}_params_{timestamp}.json"
            pd.Series(model_results['best_params']).to_json(params_file)

            # Save feature importance
            if not model_results['feature_importance'].empty:
                importance_file = output_path / f"{model_name}_importance_{timestamp}.csv"
                model_results['feature_importance'].to_csv(importance_file)

            # Save error metrics
            errors_file = output_path / f"{model_name}_errors_{timestamp}.json"
            pd.Series(model_results['errors']).to_json(errors_file)

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train crypto trading models')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of historical data to use')
    parser.add_argument('--models', nargs='+',
                        help='Models to train (lstm, xgboost, prophet, ensemble)')
    parser.add_argument('--output', type=str, default='results/training',
                        help='Output directory for results')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Setting up training environment...")
        feature_pipeline, model_trainer, model_optimizer, model_analyzer = setup_training()

        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)

        logger.info(f"Training models with data from {start_date} to {end_date}")
        results = train_models(
            feature_pipeline,
            model_trainer,
            model_optimizer,
            model_analyzer,
            start_date,
            end_date,
            args.models
        )

        logger.info("Saving training results...")
        save_training_results(results, args.output)

        logger.info("Training complete!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()