from fastapi import FastAPI
from app.routes import roles, resume, skillassess, job_recommendation_routes
from fastapi.middleware.cors import CORSMiddleware





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


app.include_router(roles.router)
app.include_router(job_recommendation_routes.router)
app.include_router(resume.router)
app.include_router(skillassess.router)
# app.include_router(blockchain_certificate.router)


