"""
Server module for running the FastAPI application
"""

import uvicorn
import logging
import traceback
import os
from .app import app

logger = logging.getLogger(__name__)

def run_server(host: str = "0.0.0.0", port: int = None, reload: bool = False):
    """
    Run the FastAPI server with the specified host and port.
    
    Args:
        host: Host address to bind the server to
        port: Port to run the server on
        reload: Whether to enable auto-reload for development
    
    Raises:
        Exception: If server fails to start
    """
    try:
        port = int(os.environ.get("PORT", 9000)) if port is None else port
        
        logger.info(f"Starting server on {host}:{port} (reload={reload})")
        uvicorn.run("api.app:app", host=host, port=port, reload=reload)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        raise 