from fastapi import Request, HTTPException
from jose import jwt
import os

SECRET_KEY = os.getenv("SECRET_KEY","secret") # The secret key used to sign and verify JWTs
ALGORITHM ="HS256"

async def get_current_user(request:Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer"):
        raise HTTPException(status_code=401,details="Authorization header missing")
    token =auth.split(" ")[1] # Extract the token from the Authorization header jab jwt token milta hai toh uske sath bearer likha hota usse hatne ke liye
    try:
        payload =jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
        return payload # Return the decoded payload which contains user information
    except jwt.JWTError:
        raise HTTPException(status_code=401,detail="invalid token")
