from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.encoders import jsonable_encoder
from app.db.mongodb import get_database
from app.models.user import UserCreate, UserInDB
from pymongo.errors import DuplicateKeyError

router = APIRouter()

@router.post("/", response_description="Register a new farmer", response_model=UserInDB)
async def create_user(user: UserCreate = Body(...), db = Depends(get_database)):
    user_data = jsonable_encoder(user)
    
    new_user = UserInDB(**user_data)
    
    try:
        insert_result = await db["users"].insert_one(new_user.model_dump(by_alias=True))
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Phone number already registered")

    created_user = await db["users"].find_one({"_id": insert_result.inserted_id})
    return UserInDB(**created_user)

@router.get("/{phone}", response_description="Get farmer profile", response_model=UserInDB)
async def get_user(phone: str, db = Depends(get_database)):
    if (user := await db["users"].find_one({"phone_number": phone})) is not None:
        return UserInDB(**user)
    
    raise HTTPException(status_code=404, detail=f"User {phone} not found")
