from fastapi import APIRouter,HTTPException,Depends 
from passlib.hash import bcrypt # hash passsword ko secure krne ke liye 
from jose import jwt # token generate krne ke liye
import os  # enviroment ko acces krne ke liye
from app.schemas.auth import UserRegister,UserLogin #request models ke liye 
from app.database import db # database se connect hone ke liye 


router = APIRouter()
SECRET_KEY = os.gentenv("SECRET","secret")
ALGORITHM = "HS256"

@router.post("/register")
async def register(user:UserRegister):
    # hash krre gye passord ko 
    hashed_password = bcrypt.hash(user.password)

    # save krre gye data base mai 
    await db.users.insert_one({  # db matalb mongi db ka database
        "name" : user.name,
        "email" : user.email,
        "password":user.password,
    })

    return {"message": "User register succcessfully"}

@router.post("/login")
async def Login(user:UserLogin):

    # user ko database mai search krre gye
    db_user = await db.user.find_one({"email":user.email})
    
    if not db_user or not bcrypt.verify(user.password,db_user["password"]):

        #agr usere nahi milla or ya password match nahi hua toh 
        raise HTTPException(status_code=401 ,detail="invalid credentials")
    
    # token gentera krre gye
    token=jwt.encode(
        {"email":user.email},
        SECRET_KEY, #secret key jo hume enviroment se mil rahi hai
        alsgorithm= ALGORITHM # ALGORITHM ko bhi envirmonetnt se le rhe haimn
     )
    
    return{"token":token}
