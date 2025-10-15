import os
import cv2
import numpy as np
import math
import json
import zipfile
import csv
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import time

from config import settings
from utils.helpers import process_frame_pipeline, validate_parameters

class VideoProcessor:
    def __init__(self):
        self.active_processes = {}
    
    def process_video_sync(self, video_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process video synchronously and return results"""
        start_time = time.time()
        
        try:
            # Validate parameters
            params = validate_parameters(parameters)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(round(fps * 0.1)))  # Sample every 100ms
            
            results = []
            frame_idx = 0
            processed_count = 0
            
            # Create results directory
            result_id = str(uuid.uuid4())
            result_dir = f"temp/results/{result_id}"
            os.makedirs(result_dir, exist_ok=True)
            
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                # Sample frames
                if frame_idx % step == 0:
                    frame_result = process_frame_pipeline(
                        frame, params, frame_idx, 
                        f"temp/debug/{result_id}" if params.get('enable_debug') else None
                    )
                    
                    if frame_result:
                        frame_result['timestamp_ms'] = (frame_idx / fps) * 1000
                        results.append(frame_result)
                        processed_count += 1
                
                frame_idx += 1
            
            cap.release()
            
            # Save results
            result_files = self._save_video_results(results, result_dir, result_id)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Find best result
            best_result = max(results, key=lambda x: x['fqs']) if results else None
            
            return {
                "total_frames": frame_idx,
                "processed_frames": processed_count,
                "best_angle": best_result['angle_deg'] if best_result else None,
                "best_fqs": best_result['fqs'] if best_result else None,
                "csv_path": f"/api/files/download/csv?filename={result_id}/results.csv",
                "best_frame_path": f"/api/files/download/best_frame?filename={result_id}/best_frame.png" if best_result else None,
                "zip_path": f"/api/files/download/zip?filename={result_id}/frames.zip",
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            raise RuntimeError(f"Video processing error: {str(e)}")
    
    def process_single_frame(self, image_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process single frame synchronously"""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return {
                    "success": False, 
                    "error": "Could not load image"
                }
            
            # Validate parameters
            params = validate_parameters(parameters)
            
            # Create debug directory if enabled
            debug_dir = None
            if params.get('enable_debug'):
                debug_id = str(uuid.uuid4())
                debug_dir = f"temp/debug/{debug_id}"
                os.makedirs(debug_dir, exist_ok=True)
            
            result = process_frame_pipeline(frame, params, 0, debug_dir)
            
            if result:
                # Prepare debug images URLs if enabled
                debug_images = None
                if debug_dir and params.get('enable_debug'):
                    debug_images = {}
                    for step in ['1_original', '2_green_mask', '3_outlines', '4_lines_detected', '5_final_result']:
                        step_path = os.path.join(debug_dir, f"{step}_frame_000000.png")
                        if os.path.exists(step_path):
                            debug_id = os.path.basename(os.path.dirname(step_path))
                            debug_images[step] = f"/api/files/debug/{debug_id}/{step}_frame_000000.png"
                
                return {
                    "success": True,
                    "angle_deg": result["angle_deg"],
                    "fqs": result["fqs"],
                    "lines_detected": result["lines_detected"],
                    "timestamp_ms": 0,
                    "debug_images": debug_images
                }
            else:
                return {
                    "success": False,
                    "error": "No green ties detected in frame"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing error: {str(e)}"
            }
    
    def _save_video_results(self, results: List[Dict], result_dir: str, result_id: str) -> Dict[str, str]:
        """Save video processing results to files"""
        if not results:
            return {}
        
        # Save CSV
        csv_path = f"{result_dir}/results.csv"
        if results:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
        
        # Create a simple best frame (in real implementation, save the actual best frame)
        best_frame_path = f"{result_dir}/best_frame.png"
        if results:
            # Create a placeholder best frame
            best_result = max(results, key=lambda x: x['fqs'])
            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Angle: {best_result['angle_deg']:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(placeholder, f"FQS: {best_result['fqs']:.3f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(best_frame_path, placeholder)
        
        # Create zip of results
        zip_path = f"{result_dir}/frames.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(csv_path, 'results.csv')
            if os.path.exists(best_frame_path):
                zipf.write(best_frame_path, 'best_frame.png')
        
        return {
            'csv': csv_path,
            'best_frame': best_frame_path,
            'zip': zip_path
        }

video_processor = VideoProcessor()