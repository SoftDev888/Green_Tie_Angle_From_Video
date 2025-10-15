from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import os
import shutil

from services.file_manager import file_manager

router = APIRouter(prefix="/api/files", tags=["files"])

@router.get("/download/{file_type}")
async def download_file(file_type: str, filename: Optional[str] = None):
    """Download processing results"""
    
    file_paths = {
        "csv": f"temp/results/{filename}" if filename else None,
        "best_frame": f"temp/results/{filename}" if filename else None,
        "zip": f"temp/results/{filename}" if filename else None,
    }
    
    if file_type not in file_paths or not file_paths[file_type]:
        raise HTTPException(400, f"Invalid file type or filename: {file_type}")
    
    file_path = file_paths[file_type]
    if not os.path.exists(file_path):
        raise HTTPException(404, f"File not found: {file_path}")
    
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type='application/octet-stream'
    )

@router.get("/debug/{debug_id}/{step_name}")
async def get_debug_image(debug_id: str, step_name: str):
    """Get debug image from processing"""
    debug_path = f"temp/debug/{debug_id}/{step_name}.png"
    
    if not os.path.exists(debug_path):
        raise HTTPException(404, f"Debug image not found: {debug_path}")
    
    return FileResponse(
        path=debug_path,
        filename=f"{debug_id}_{step_name}.png",
        media_type='image/png'
    )

@router.delete("/debug/{debug_id}")
async def cleanup_debug_files(debug_id: str):
    """Clean up debug files"""
    debug_dir = f"temp/debug/{debug_id}"
    
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        return {"message": f"Debug files cleaned up: {debug_id}"}
    else:
        raise HTTPException(404, f"Debug directory not found: {debug_id}")