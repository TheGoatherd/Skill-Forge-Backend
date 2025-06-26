from fastapi import APIRouter,Depends
from app.dependencies.dependcies import get_current_user

router = APIRouter()
@router.get("/dashboard")
async def get_dashboard(current_user:dict =Depends(get_current_user)):
    return{
        "message":"welcome to the dashboard,{current_user['email']}"
    }