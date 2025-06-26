from fastapi import APIRouter,HTTPException
from pydantic import BaseModel
import os
from jose import jwt
from passlib.hash import bcrypt
from app.database.mongodb import db

class User(BaseModel):
    email:str
    password:str

router= APIRouter()

SECRET_KEY = os.getenv("SECRET_KEY","secret")
ALGORITHM= "HS256"


@router.post("/Register")
async def register(user:User):
    exsiting_user = await db.users.find_one({"email":user.email})
    if exsiting_user:
        raise HTTPException(status_code=400,detail="Email Already Registered")
    
    hash_pass= bcrypt.hash(user.password)
    await db.users.insert_one({
        "email":user.email,
        "password":hash_pass
    })

    return{"message": "User Register successfully"}

@router.post("/Login")
async def login(user:User):
    db_users = await db.users.find_one({"email":user.email})
    if not db_users or not bcrypt.verify(user.password,db_users["password"]):
        raise HTTPException(status_code=400,detail="invaild credentials")
    
    token=jwt.encode({"email":user.email},SECRET_KEY,algorithm=ALGORITHM)
    # Here you would typically return the token to the user
    # For this example, we will just return a success message
    # In a real application, you would return the token in the response
    # and use it for subsequent authenticated requests.
    return {"message":"Login Successful","token":token}

