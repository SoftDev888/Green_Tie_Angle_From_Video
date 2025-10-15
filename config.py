import os
from typing import Dict, Any

class Settings:
    def __init__(self):
        self.upload_dir = "temp/uploads"
        self.results_dir = "temp/results"
        self.debug_dir = "temp/debug"
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png"}
        
        # Processing parameters (from your original code)
        self.target_width = 900
        self.max_samples = None
        self.save_no_detect = True
        self.line_linking_angle_tolerance = 1.5
        self.line_linking_distance_tolerance = 5
        self.min_line_group_size = 2
        
        # Create directories
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

settings = Settings()