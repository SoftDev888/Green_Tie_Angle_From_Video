from fastapi import APIRouter, UploadFile, File
from typing import Optional
import uuid
import os
import time
import cv2

from models.schemas import (
    ProcessingParameters, VideoProcessingResponse, 
    FrameAnalysisResponse, ProcessingStatusResponse
)
from services.video_processor import video_processor
from services.file_manager import file_manager

router = APIRouter(prefix="/api/processing", tags=["processing"])

@router.post("/video", response_model=VideoProcessingResponse)
async def process_video(
    file: UploadFile = File(...),
    parameters: Optional[ProcessingParameters] = None
):
    """Process video synchronously and return results"""
    start_time = time.time()
    
    try:
        # Save uploaded file
        filename = await file_manager.save_upload_file(file)
        file_path = file_manager.get_file_path(filename)
        
        # Process video synchronously
        params_dict = parameters.dict() if parameters else {}
        results = video_processor.process_video_sync(file_path, params_dict)
        
        # Cleanup uploaded file
        os.remove(file_path)
        
        processing_time = (time.time() - start_time) * 1000
        
        return VideoProcessingResponse(
            success=True,
            total_frames=results.get("total_frames", 0),
            processed_frames=results.get("processed_frames", 0),
            best_angle=results.get("best_angle"),
            best_fqs=results.get("best_fqs"),
            results_csv=results.get("csv_path"),
            best_frame=results.get("best_frame_path"),
            frames_zip=results.get("zip_path"),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return VideoProcessingResponse(
            success=False,
            total_frames=0,
            processed_frames=0,
            error=f"Video processing failed: {str(e)}",
            processing_time_ms=processing_time
        )

@router.post("/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(
    file: UploadFile = File(...),
    parameters: Optional[ProcessingParameters] = None
):
    """Analyze single frame synchronously"""
    start_time = time.time()
    
    try:
        # Save uploaded file
        filename = await file_manager.save_upload_file(file)
        file_path = file_manager.get_file_path(filename)
        
        # Process frame
        params_dict = parameters.dict() if parameters else {}
        results = video_processor.process_single_frame(file_path, params_dict)
        
        # Cleanup uploaded file
        os.remove(file_path)
        
        processing_time = (time.time() - start_time) * 1000
        results["processing_time_ms"] = processing_time
        
        return FrameAnalysisResponse(**results)
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return FrameAnalysisResponse(
            success=False,
            error=f"Frame analysis failed: {str(e)}",
            processing_time_ms=processing_time
        )

@router.get("/status")
async def get_processing_status():
    """Get overall processing status (simple health check)"""
    return ProcessingStatusResponse(
        status="ready",
        message="API is ready to process requests",
        progress=100.0
    )