"""
Unit tests for database configuration and session management.
"""
import pytest
from pathlib import Path
from sqlmodel import Session, create_engine
from app.core.db import get_session, create_db_and_tables
from app.db.models import Image, Coin


def test_create_db_and_tables():
    """Test database table creation."""
    # This should complete without errors
    create_db_and_tables()


def test_get_session_yields_session():
    """Test that get_session yields a valid session."""
    session_gen = get_session()
    session = next(session_gen)

    assert isinstance(session, Session)

    # Clean up
    try:
        next(session_gen)
    except StopIteration:
        pass  # Expected behavior


def test_session_can_create_and_query_image(tmp_path):
    """Test database operations with session."""
    # Create in-memory database for testing
    from sqlmodel import SQLModel, create_engine
    from sqlmodel.pool import StaticPool

    engine = create_engine(
        "sqlite://",  # In-memory
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create an image
        image = Image(id="test123", filename="test.jpg")
        session.add(image)
        session.commit()

        # Query it back
        retrieved = session.get(Image, "test123")
        assert retrieved is not None
        assert retrieved.filename == "test.jpg"


def test_session_can_create_coin_with_relationship(tmp_path):
    """Test coin creation with image relationship."""
    from sqlmodel import SQLModel, create_engine
    from sqlmodel.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create image
        image = Image(id="img123", filename="test.jpg")
        session.add(image)

        # Create coin linked to image
        coin = Coin(
            id="img123_coin_001",
            image_id="img123",
            center_x=100.0,
            center_y=100.0,
            radius=50.0,
            is_slanted=False,
            bbox_x=50.0,
            bbox_y=50.0,
            bbox_w=100.0,
            bbox_h=100.0,
        )
        session.add(coin)
        session.commit()
        session.refresh(image)

        # Verify relationship
        assert len(image.coins) == 1
        assert image.coins[0].id == "img123_coin_001"
