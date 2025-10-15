from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ProcessingParameters(BaseModel):
    target_width: int = Field(default=900, ge=100, le=2000)
    max_samples: Optional[int] = Field(default=None, ge=1)
    save_no_detect: bool = Field(default=True)
    line_linking_angle_tolerance: float = Field(default=1.5, ge=0.1, le=10.0)
    line_linking_distance_tolerance: int = Field(default=5, ge=1, le=50)
    min_line_group_size: int = Field(default=2, ge=1, le=10)
    enable_debug: bool = Field(default=False)

class FrameAnalysisResponse(BaseModel):
    success: bool
    angle_deg: Optional[float] = None
    fqs: Optional[float] = None
    lines_detected: int = 0
    timestamp_ms: Optional[float] = None
    error: Optional[str] = None
    debug_images: Optional[Dict[str, str]] = None
    processing_time_ms: Optional[float] = None

class VideoProcessingResponse(BaseModel):
    success: bool
    total_frames: int
    processed_frames: int
    best_angle: Optional[float] = None
    best_fqs: Optional[float] = None
    results_csv: Optional[str] = None
    best_frame: Optional[str] = None
    frames_zip: Optional[str] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

class ProcessingStatusResponse(BaseModel):
    status: str
    message: str
    progress: Optional[float] = None