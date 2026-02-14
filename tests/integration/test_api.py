"""
Integration tests for the Coin Detection API.
Tests all endpoints end-to-end with a real database and test client.
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.main import app
from app.core.db import get_session
from app.core.config import settings

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# Test database setup
@pytest.fixture(name="session")
def session_fixture():
    """Create a fresh in-memory database for each test."""
    engine = create_engine(
        "sqlite://",  # In-memory database
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="client")
def client_fixture(session: Session):
    """Create a test client with overridden database session."""

    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def test_image_path():
    """Path to test image fixture."""
    return Path(__file__).parent / "fixtures" / "test_coins.jpg"


@pytest.fixture
def invalid_file_path():
    """Path to invalid file fixture."""
    return Path(__file__).parent / "fixtures" / "test_invalid.txt"


# ============================================================================
# Image Upload Tests
# ============================================================================


def test_upload_image_success(client: TestClient, test_image_path: Path):
    """Test successful image upload and detection."""
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/images",
            files={"file": ("test_coins.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "id" in data
    assert "filename" in data
    assert "coins" in data
    assert isinstance(data["coins"], list)

    # Verify image ID format (8 characters)
    assert len(data["id"]) == 8

    # Verify filename format
    assert data["filename"].endswith(".jpg")
    assert data["filename"].startswith(data["id"])


def test_upload_image_invalid_file_type(client: TestClient, invalid_file_path: Path):
    """Test upload rejection for invalid file types."""
    with open(invalid_file_path, "rb") as f:
        response = client.post(
            "/api/v1/images",
            files={"file": ("test.txt", f, "text/plain")}
        )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_upload_png_image(client: TestClient, test_image_path: Path):
    """Test that PNG images are also accepted."""
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/images",
            files={"file": ("test.png", f, "image/png")}
        )

    # Should succeed (even though file is actually JPEG, we're testing content_type validation)
    assert response.status_code == 200


def test_upload_returns_request_id_header(client: TestClient, test_image_path: Path):
    """Test that response includes X-Request-ID header for tracing."""
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/images",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )

    assert "x-request-id" in response.headers


# ============================================================================
# Image Retrieval Tests
# ============================================================================


def test_get_image_details_success(client: TestClient, test_image_path: Path):
    """Test retrieving image details by ID."""
    # First upload an image
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    image_id = upload_response.json()["id"]

    # Then retrieve it
    response = client.get(f"/api/v1/images/{image_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == image_id
    assert "coins" in data
    assert isinstance(data["coins"], list)


def test_get_image_not_found(client: TestClient):
    """Test 404 for non-existent image."""
    response = client.get("/api/v1/images/nonexist")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_image_includes_request_id(client: TestClient, test_image_path: Path):
    """Test that image retrieval includes request ID in headers."""
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    image_id = upload_response.json()["id"]

    response = client.get(f"/api/v1/images/{image_id}")
    assert "x-request-id" in response.headers


# ============================================================================
# Coin Retrieval Tests
# ============================================================================


def test_get_coin_details_success(client: TestClient, test_image_path: Path):
    """Test retrieving coin details by ID."""
    # Upload image with coins
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    coins = upload_response.json()["coins"]

    # Skip if no coins detected
    if not coins:
        pytest.skip("No coins detected in test image")

    coin_id = coins[0]["id"]

    # Retrieve coin details
    response = client.get(f"/api/v1/coins/{coin_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == coin_id
    assert "center_x" in data
    assert "center_y" in data
    assert "radius" in data
    assert "is_slanted" in data
    assert "bbox_x" in data
    assert "bbox_y" in data
    assert "bbox_w" in data
    assert "bbox_h" in data


def test_get_coin_not_found(client: TestClient):
    """Test 404 for non-existent coin."""
    response = client.get("/api/v1/coins/nonexistent_coin")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_coin_id_format(client: TestClient, test_image_path: Path):
    """Test that coin IDs follow the expected format: {image_id}_coin_{NNN}."""
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )

    data = upload_response.json()
    image_id = data["id"]
    coins = data["coins"]

    # Skip if no coins
    if not coins:
        pytest.skip("No coins detected in test image")

    # Verify coin ID format
    for idx, coin in enumerate(coins):
        expected_suffix = f"_coin_{idx+1:03d}"
        assert coin["id"].startswith(image_id)
        assert coin["id"].endswith(expected_suffix)


# ============================================================================
# Render/Visualization Tests
# ============================================================================


def test_render_image_success(client: TestClient, test_image_path: Path):
    """Test rendering image with mask overlay."""
    # Upload image
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    image_id = upload_response.json()["id"]

    # Request rendered image
    response = client.get(f"/api/v1/images/{image_id}/render")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_render_image_not_found(client: TestClient):
    """Test 404 for rendering non-existent image."""
    response = client.get("/api/v1/images/nonexist/render")

    assert response.status_code == 404


# ============================================================================
# Health Check Tests
# ============================================================================


def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert "model" in data

    # Verify model status
    model_status = data["model"]
    assert "available" in model_status
    assert "model_path" in model_status
    assert "status" in model_status


def test_health_check_model_status(client: TestClient):
    """Test that health check reports model availability status."""
    response = client.get("/api/v1/health")

    model_status = response.json()["model"]

    # Model status should be one of: loaded, error, not_loaded
    assert model_status["status"] in ["loaded", "error", "not_loaded"]

    # If model unavailable, should have error message or not_loaded status
    if not model_status["available"]:
        assert model_status["status"] in ["error", "not_loaded"]


# ============================================================================
# Data Validation Tests
# ============================================================================


def test_coin_geometric_properties_valid(client: TestClient, test_image_path: Path):
    """Test that coin geometric properties are valid numbers."""
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )

    coins = upload_response.json()["coins"]

    # Skip if no coins
    if not coins:
        pytest.skip("No coins detected in test image")

    for coin in coins:
        # All geometric properties should be valid numbers
        assert coin["center_x"] >= 0
        assert coin["center_y"] >= 0
        assert coin["radius"] > 0
        assert coin["bbox_x"] >= 0
        assert coin["bbox_y"] >= 0
        assert coin["bbox_w"] > 0
        assert coin["bbox_h"] > 0

        # is_slanted should be boolean
        assert isinstance(coin["is_slanted"], bool)


def test_bbox_and_center_consistency(client: TestClient, test_image_path: Path):
    """Test that bbox and center coordinates are consistent."""
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )

    coins = upload_response.json()["coins"]

    if not coins:
        pytest.skip("No coins detected in test image")

    for coin in coins:
        # Center should be within bbox
        expected_center_x = coin["bbox_x"] + coin["bbox_w"] / 2
        expected_center_y = coin["bbox_y"] + coin["bbox_h"] / 2

        # Allow small floating point error
        assert abs(coin["center_x"] - expected_center_x) < 0.1
        assert abs(coin["center_y"] - expected_center_y) < 0.1


# ============================================================================
# Multiple Upload Tests
# ============================================================================


def test_multiple_uploads_have_unique_ids(client: TestClient, test_image_path: Path):
    """Test that multiple uploads generate unique image IDs."""
    image_ids = set()

    for _ in range(3):
        with open(test_image_path, "rb") as f:
            response = client.post(
                "/api/v1/images",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        image_ids.add(response.json()["id"])

    # All IDs should be unique
    assert len(image_ids) == 3


def test_multiple_images_stored_separately(client: TestClient, test_image_path: Path):
    """Test that multiple uploaded images are stored separately."""
    upload_ids = []

    # Upload 2 images
    for _ in range(2):
        with open(test_image_path, "rb") as f:
            response = client.post(
                "/api/v1/images",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        upload_ids.append(response.json()["id"])

    # Retrieve both and verify they're distinct
    for image_id in upload_ids:
        response = client.get(f"/api/v1/images/{image_id}")
        assert response.status_code == 200
        assert response.json()["id"] == image_id
