from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard"""
    with open(BASE_DIR / "templates/dashboard.html", "r") as f:
        return f.read()

@app.get("/api/collector/stats")
async def get_stats() -> Dict:
    """Get real-time collection statistics"""
    try:
        # Import collector only when needed to avoid circular imports
        from src.globals import collector

        now = datetime.now()
        timestamps = [(now - timedelta(seconds=x)).strftime('%H:%M:%S')
                      for x in range(60, 0, -1)]

        return {
            "status": {
                "websocket": "active" if collector.websocket.is_alive() else "error",
                "database": "active" if collector.db.validate_collections() else "error",
                "warnings": collector.get_warnings() if hasattr(collector, 'get_warnings') else []
            },
            "stats": {
                "trades": collector.stats['trades_processed'],
                "orders": collector.stats['orderbook_updates'],
                "rate": (collector.stats['trades_processed'] + collector.stats['orderbook_updates']) / 60,
                "errors": collector.stats['errors']
            },
            "timestamps": timestamps,
            "volumes": collector.get_volume_history() if hasattr(collector, 'get_volume_history') else [],
            "bidDepth": collector.get_depth_history('bid') if hasattr(collector, 'get_depth_history') else [],
            "askDepth": collector.get_depth_history('ask') if hasattr(collector, 'get_depth_history') else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))