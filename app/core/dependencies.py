"""
Dependency injection module for FastAPI route handlers.

This module manages the VectorService singleton that gets injected into API
route handlers via FastAPI's Depends() system. The service is instantiated
in main.py and stored here for access by all endpoints.

The pattern:
1. main.py creates all service instances
2. main.py stores the VectorService in this module (deps.vector_service = ...)
3. Route handlers call get_vector_service() which returns the stored instance
4. FastAPI Depends() ensures the same instance is reused across requests
"""

from app.services.vector_service import VectorService

# Global reference to the VectorService instance
# This is set in main.py after service initialization
vector_service: VectorService | None = None


def get_vector_service() -> VectorService:
    """
    Dependency injection function for FastAPI route handlers.
    
    Returns the globally initialized VectorService instance. Can be used
    as a dependency in FastAPI route handlers via:
        @router.post("/endpoint")
        def my_route(vector_service = Depends(get_vector_service)):
            ...
    
    Raises:
        RuntimeError: If VectorService hasn't been initialized yet
        
    Returns:
        VectorService: The initialized service instance
    """
    if vector_service is None:
        raise RuntimeError("VectorService not initialized")
    return vector_service
