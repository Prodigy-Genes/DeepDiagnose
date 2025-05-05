import uuid
from datetime import datetime, timezone
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )

    username: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))
    
    #Relationships
    images: Mapped[list["MedicalImage"]] = relationship(
        back_populates="user"
    )
    
    logs: Mapped[list["SystemLog"]] = relationship(
        back_populates="user"
    )
    
    auth_tokens: Mapped[list["AuthToken"]] = relationship(
        back_populates="user"
    )

class AuthToken(Base):
    __tablename__ = "auth_tokens"

    token_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.user_id")
    )

    token: Mapped[str] = mapped_column(
        String(512),
        nullable=False
    )

    expires_at: Mapped[datetime] = mapped_column(
        nullable=False
    )

    last_used_at: Mapped[datetime] = mapped_column(
        nullable=True
    )

    user: Mapped["User"] = relationship(
        back_populates="auth_tokens"
    )



                                                 