from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models.user_models import User


async def get_user_by_email(session: AsyncSession, email: str):
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

async def  create_user(session: AsyncSession, user_data: dict):
    user = User(**user_data)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user
