from fastapi import APIRouter
from typing import Dict, Any
import os

from models.schemas import ProcessingParameters
from services.video_processor import video_processor
from config import settings

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.get("/parameters")
async def get_parameters() -> Dict[str, Any]:
    """Get current processing parameters"""
    return {
        "target_width": settings.target_width,
        "max_samples": settings.max_samples,
        "save_no_detect": settings.save_no_detect,
        "line_linking_angle_tolerance": settings.line_linking_angle_tolerance,
        "line_linking_distance_tolerance": settings.line_linking_distance_tolerance,
        "min_line_group_size": settings.min_line_group_size,
        "enable_debug": False
    }

@router.put("/parameters")
async def update_parameters(parameters: ProcessingParameters) -> Dict[str, Any]:
    """Update processing parameters"""
    # Update settings
    settings.target_width = parameters.target_width
    settings.max_samples = parameters.max_samples
    settings.save_no_detect = parameters.save_no_detect
    settings.line_linking_angle_tolerance = parameters.line_linking_angle_tolerance
    settings.line_linking_distance_tolerance = parameters.line_linking_distance_tolerance
    settings.min_line_group_size = parameters.min_line_group_size
    
    return {"message": "Parameters updated successfully", "parameters": parameters.dict()}