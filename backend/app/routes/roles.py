from fastapi import APIRouter

router = APIRouter()

@router.get("/roles")
def list_roles():
    return ["frontend", "backend", "devops", "uiux"]
