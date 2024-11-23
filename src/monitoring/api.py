from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict

from src.globals import get_collector

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates
app.mount("/static", StaticFiles(directory="src/monitoring/templates"), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard"""
    template_path = Path("src/monitoring/templates/dashboard.html")
    with open(template_path) as f:
        return f.read()

@app.get("/api/collector/stats")
async def get_stats() -> Dict:
    """Get real-time collection statistics"""
    try:
        collector = get_collector()
        if not collector:
            raise HTTPException(status_code=500, detail="Collector not initialized")

        now = datetime.now()
        timestamps = [(now - timedelta(seconds=x)).strftime('%H:%M:%S')
                      for x in range(60, 0, -1)]

        return {
            "status": {
                "websocket": "active" if collector.websocket.is_alive() else "error",
                "database": "active" if collector.db.validate_collections() else "error",
                "warnings": collector.get_warnings()
            },
            "stats": {
                "trades": collector.stats['trades_processed'],
                "orders": collector.stats['orderbook_updates'],
                "rate": collector.get_collection_rate(),
                "errors": collector.stats['errors']
            },
            "timestamps": timestamps,
            "volumes": collector.get_volume_history(),
            "bidDepth": collector.get_depth_history('bid'),
            "askDepth": collector.get_depth_history('ask')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/status")
async def get_model_status() -> Dict:
    """Get model training status"""
    try:
        # This would be expanded based on your model monitoring needs
        return {
            "status": "active",
            "last_training": datetime.now().isoformat(),
            "models": {
                "lstm": {"accuracy": 0.85},
                "lightgbm": {"accuracy": 0.87},
                "ensemble": {"accuracy": 0.89}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))