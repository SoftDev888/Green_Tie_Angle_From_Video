from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers.processing import router as processing_router
from routers.analysis import router as analysis_router
from routers.files import router as files_router

app = FastAPI(
    title="Green Tie Detection API",
    description="API for detecting and analyzing green ties in videos and images",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(processing_router)
app.include_router(analysis_router)
app.include_router(files_router)

@app.get("/")
async def root():
    return {"message": "Green Tie Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )