"""
Data models and schemas for the Green Tie Detection API.
"""

from .schemas import (
    ProcessingParameters,
    FrameAnalysisResponse,
    VideoProcessingResponse,
    ProcessingStatusResponse
)

__all__ = [
    "ProcessingParameters",
    "FrameAnalysisResponse", 
    "VideoProcessingResponse",
    "ProcessingStatusResponse"
]