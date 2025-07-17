from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models.user_models import User
from fastapi import HTTPException


async def get_user_by_email(session: AsyncSession, email: str):
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

async def get_user_by_username(session: AsyncSession, username: str):
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()

async def create_user(session: AsyncSession, user_data: dict):
    user = User(**user_data)
    session.add(user)
    try:
        await session.commit()
    except Exception as e:
        await session.rollback()
        # Handle specific errors like IntegrityError here
        raise HTTPException(
            status_code=400, 
            detail="Could not create user"
        ) from e
    await session.refresh(user)
    return user
