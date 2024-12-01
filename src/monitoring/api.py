from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import asyncio
import logging

from src.globals import get_collector
from src.config import settings
import nest_asyncio
# Enable nested asyncio support
nest_asyncio.apply()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates directory
current_dir = Path(__file__).parent
templates_dir = current_dir / "templates"
static_dir = current_dir / "static"

# Ensure directories exist
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# WebSocket connections
active_connections: List[WebSocket] = []

async def broadcast_metrics():
    """Broadcast metrics to all connected clients"""
    while True:
        try:
            if active_connections:
                # Get collector metrics
                collector = get_collector()
                if collector:
                    metrics = {
                        "timestamp": datetime.now().isoformat(),
                        "status": {
                            "websocket": collector.get_connection_status(),
                            "database": collector.get_db_status(),
                            "warnings": collector.get_warnings()
                        },
                        "stats": {
                            "trades": collector.stats['trades_processed'],
                            "orders": collector.stats['orderbook_updates'],
                            "rate": collector.get_collection_rate(),
                            "errors": collector.stats['errors']
                        },
                        "market": {
                            "volumes": collector.get_volume_history(),
                            "depth": {
                                "bid": collector.get_depth_history('bid'),
                                "ask": collector.get_depth_history('ask')
                            }
                        }
                    }

                    # Broadcast to all connections
                    for connection in active_connections:
                        try:
                            await connection.send_json(metrics)
                        except Exception as e:
                            logging.error(f"WebSocket send error: {str(e)}")
                            active_connections.remove(connection)

        except Exception as e:
            logging.error(f"Broadcast error: {str(e)}")

        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(broadcast_metrics())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection handler"""
    try:
        await websocket.accept()
        active_connections.append(websocket)

        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except Exception:
            active_connections.remove(websocket)

    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard page"""
    try:
        template_path = templates_dir / "dashboard.html"
        with open(template_path) as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error serving dashboard: {e}")
        return f"""
        <html>
            <body>
                <h1>Crypto Trading Platform Monitor</h1>
                <div id="connectionStatus">Initializing...</div>
            </body>
        </html>
        """

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
                "websocket": collector.get_connection_status(),
                "database": collector.get_db_status(),
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

@app.get("/api/collector/health")
async def health_check() -> Dict:
    """System health check endpoint"""
    try:
        collector = get_collector()
        if not collector:
            return {
                "status": "error",
                "message": "Collector not initialized"
            }

        return {
            "status": "healthy",
            "websocket": collector.get_connection_status(),
            "database": collector.get_db_status(),
            "last_update": collector.stats.get('last_update', '').isoformat()
            if collector.stats.get('last_update') else None,
            "errors": collector.stats['errors']
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/collector/config")
async def get_config() -> Dict:
    """Get current configuration"""
    return {
        "symbols": settings.trading_symbols,
        "use_testnet": settings.use_testnet,
        "batch_size": settings.batch_size,
        "history_size": settings.history_size,
        "update_interval": settings.update_interval
    }