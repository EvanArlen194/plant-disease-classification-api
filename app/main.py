"""
Main entry point for the Plant Disease Classification API
This file acts as a trigger for the main() function
"""

import logging
import sys
import traceback
import os
from api.server import run_server
from api.app import app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function that starts the API server
    """
    try:
        port = int(os.environ.get("PORT", 9000))
        
        is_dev = os.environ.get("ENVIRONMENT", "development") == "development"
        
        logger.info(f"Starting Plant Disease Classification API server on port {port}")
        logger.info(f"Environment: {'Development' if is_dev else 'Production'}")
        
        run_server(host="0.0.0.0", port=port, reload=is_dev)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
    