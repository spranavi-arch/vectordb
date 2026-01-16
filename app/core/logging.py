

import logging


def setup_logging():
    """
    Configure Python logging with a standard format.
    
    Format: TIMESTAMP | LEVEL | LOGGER_NAME | MESSAGE
    Example: 2024-01-15 10:30:45,123 | INFO | app.services.vector_service | Indexed document abc123
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
