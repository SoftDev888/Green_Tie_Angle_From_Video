"""
API routers for the Green Tie Detection API.
"""

from .processing import router as processing_router
from .analysis import router as analysis_router
from .files import router as files_router

__all__ = [
    "processing_router",
    "analysis_router", 
    "files_router"
]