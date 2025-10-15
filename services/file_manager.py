import os
import uuid
import shutil
import aiofiles
from fastapi import UploadFile, HTTPException
from config import settings

class FileManager:
    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.results_dir = settings.results_dir
        self.debug_dir = settings.debug_dir

    async def save_upload_file(self, file: UploadFile) -> str:
        """Save uploaded file and return unique filename"""
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(400, f"File type {file_extension} not allowed")
        
        filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(self.upload_dir, filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return filename

    def get_file_path(self, filename: str) -> str:
        """Get full path for a filename"""
        return os.path.join(self.upload_dir, filename)

    def create_job_directory(self, job_id: str) -> str:
        """Create directory for job results"""
        job_dir = os.path.join(self.results_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    def get_job_directory(self, job_id: str) -> str:
        """Get job directory path"""
        return os.path.join(self.results_dir, job_id)

    def cleanup_job_files(self, job_id: str):
        """Clean up job files after completion"""
        job_dir = self.get_job_directory(job_id)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)

file_manager = FileManager()