import os
import uvicorn
from fastapi import FastAPI
from app.routes import roles, resume, skillassess, job_recommendation_routes
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response





app = FastAPI()

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.head("/")
def head_root():
    """Handle HEAD requests for health checks"""
    return Response(status_code=200)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

app.include_router(roles.router)
app.include_router(job_recommendation_routes.router)
app.include_router(resume.router)
app.include_router(skillassess.router)
# app.include_router(blockchain_certificate.router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default to 8000 if PORT is not set
    uvicorn.run("main:app", host="0.0.0.0", port=port)
